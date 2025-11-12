import asyncio
import json
import logging
import os
import time
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

# Optional enrichment (translation + entities via AWS Comprehend Medical)
try:
    from services import nlp
except ImportError:
    nlp = None  # graceful degradation

router = APIRouter()
logger = logging.getLogger(__name__)

# -------------------- Configuration / Flags --------------------
DBG = os.getenv("DEBUG_ALIBABA_TRANSCRIBE") == "1"
DO_TRANSLATE = os.getenv("ENABLE_ALIBABA_TRANSLATION") == "1"
DO_ENTITIES = os.getenv("ENABLE_ALIBABA_ENTITIES") == "1"
FAKE_DIARIZATION = os.getenv("ALIBABA_FAKE_DIARIZATION") == "1"  # fallback turn-taking assignment
SPEAKER_COUNT = int(os.getenv("ALIBABA_SPEAKER_COUNT", "2"))
MAX_FAKE_SPEAKERS = max(2, SPEAKER_COUNT)

if DBG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

logging.getLogger("dashscope").setLevel(logging.INFO)

# -------------------- Language Mapping / Heuristics --------------------
CANTONESE_HINT_CHARS = set("嘅咩喺咗冇哋啲呢啦噉仲咪睇醫")  # crude heuristic for Cantonese

def map_lang_to_ui(lang_raw: Optional[str], text: str) -> str:
    """
    Map Paraformer language token (if present) OR infer from text.
    Preference order:
      - Explicit lang_raw
      - CJK detection + Cantonese char heuristic
      - Default English
    """
    if lang_raw:
        lr = lang_raw.lower()
        if lr.startswith("en"):
            return "en-US"
        if "yue" in lr or "cant" in lr:
            return "zh-HK"
        if lr.startswith("zh"):
            return "zh-TW"
    # Heuristic fallback
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        if any(ch in CANTONESE_HINT_CHARS for ch in text):
            return "zh-HK"
        return "zh-TW"
    return "en-US"

# -------------------- Speaker Extraction --------------------
WORD_SPEAKER_KEYS = ("speaker", "spk", "speaker_id", "spk_id")

def _norm_spk(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    if s.startswith("spk_"):
        return s
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return f"spk_{int(digits)}"
    # Accept simple integer-like strings
    if s.isdigit():
        return f"spk_{int(s)}"
    return s.lower()

def extract_majority_word_speaker(words: Any) -> Optional[str]:
    if not isinstance(words, list) or not words:
        return None
    speakers: List[str] = []
    for w in words:
        if not isinstance(w, dict):
            continue
        raw = None
        for k in WORD_SPEAKER_KEYS:
            if k in w:
                raw = w[k]
                break
        sp = _norm_spk(raw)
        if sp:
            speakers.append(sp)
    if not speakers:
        return None
    counts = Counter(speakers)
    winner, _ = counts.most_common(1)[0]
    return winner

def extract_sentence_speaker(sentence: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """
    Attempt speaker extraction from sentence-level fields first, then word-level majority.
    Returns (speaker_label, source_tag)
    source_tag: 'sentence', 'words', 'fake', or 'none'
    """
    for key in ("speaker", "speaker_id", "spk", "spk_id"):
        raw = sentence.get(key)
        sp = _norm_spk(raw)
        if sp:
            return sp, "sentence"

    words = sentence.get("words") or sentence.get("word") or sentence.get("tokens") or sentence.get("Tokens")
    sp_words = extract_majority_word_speaker(words)
    if sp_words:
        return sp_words, "words"
    return None, "none"

# -------------------- AWS-Shaped Payload Construction --------------------
def build_aws_payload(
    *,
    text: str,
    is_partial: bool,
    result_id: str,
    speaker: Optional[str],
    language_code: str,
    start_ms: Optional[int],
    end_ms: Optional[int],
    translated_text: Optional[str],
    entities: Optional[List[dict]],
    debug: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    items = []
    if speaker:
        items.append({"Type": "pronunciation", "Speaker": speaker})

    result_obj: Dict[str, Any] = {
        "ResultId": result_id,
        "IsPartial": is_partial,
        "Alternatives": [
            {
                "Transcript": text,
                "Items": items,
            }
        ],
        "LanguageCode": language_code,
    }
    if speaker:
        result_obj["Speaker"] = speaker
    if start_ms is not None:
        result_obj["StartTimeMs"] = start_ms
    if end_ms is not None:
        result_obj["EndTimeMs"] = end_ms

    payload: Dict[str, Any] = {
        "Transcript": {"Results": [result_obj]},
        "DisplayText": text,
        "TranslatedText": translated_text,
        "ComprehendEntities": entities or [],
    }
    if debug and DBG:
        payload["_debug"] = debug
    return payload

# -------------------- Callback --------------------
class ParaformerCallback(RecognitionCallback):
    """
    Translate DashScope events (RecognitionResult or raw JSON) into AWS-like streaming payloads.
    Handles:
      - Unique result IDs
      - Partial vs final distinction
      - Language mapping
      - Speaker diarization (real or fake fallback)
      - Optional translation & entity extraction
      - Timing based on SDK's begin_time / end_time fields (preferred) else wall-clock
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self.loop = loop
        self.queue = queue
        self.session_start_monotonic = time.monotonic()
        self.final_counter = 0
        self.current_partial_id: Optional[str] = None
        self.last_final_speaker: Optional[str] = None
        self.fake_turn_state: List[str] = []  # maintain encountered speakers if fake diarization engaged

    def _emit(self, data: Dict[str, Any]) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, data)

    def on_open(self):
        logger.info("[Alibaba] Connection opened")

    def on_error(self, message: str):
        logger.error("[Alibaba] Error: %s", message)
        self._emit({"_error": True, "message": message})

    def on_close(self):
        logger.info("[Alibaba] Connection closed")

    def on_complete(self):
        logger.info("[Alibaba] Recognition completed")
        self._emit({"_complete": True})

    # Utility for fallback timing if begin_time is missing
    def _rel_ms(self) -> int:
        return int((time.monotonic() - self.session_start_monotonic) * 1000)

    def _assign_fake_speaker(self) -> str:
        """
        Simple turn-taking heuristic:
        - First speaker => spk_0
        - Next distinct turn => spk_1
        - If >2 required, cycle spk_2, spk_3 ...
        """
        if not self.fake_turn_state:
            self.fake_turn_state.append("spk_0")
            return "spk_0"
        if len(self.fake_turn_state) == 1:
            self.fake_turn_state.append("spk_1")
            return "spk_1"
        # Cycle if more needed
        next_index = len(self.fake_turn_state)
        spk_label = f"spk_{next_index % MAX_FAKE_SPEAKERS}"
        self.fake_turn_state.append(spk_label)
        return spk_label

    def _parse_raw_json(self, raw_str: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], bool]]:
        try:
            obj = json.loads(raw_str)
        except Exception:
            return None
        header = obj.get("header", {}) or {}
        payload = obj.get("payload", {}) or {}
        name = header.get("name")
        if name == "TranscriptionResultChanged":
            is_final = False
        elif name == "SentenceEnd":
            is_final = True
        else:
            return None
        return header, payload, is_final

    def on_event(self, message: Any):
        try:
            # Normalize into "sentence" dict and final flag.
            sentence: Dict[str, Any] = {}
            is_final = False
            raw_debug: Dict[str, Any] = {}
            if isinstance(message, RecognitionResult):
                # SDK object
                sent = message.get_sentence()
                if not isinstance(sent, dict):
                    return
                sentence = sent
                raw_debug["raw_sentence_keys"] = list(sent.keys())
                try:
                    is_final = message.is_sentence_end(sent)
                except Exception:
                    is_final = bool(sent.get("sentence_end") or sent.get("is_sentence_end"))
            elif isinstance(message, str):
                parsed = self._parse_raw_json(message)
                if not parsed:
                    return
                header, payload, is_final = parsed
                sentence = payload
                raw_debug["raw_header"] = header
                raw_debug["raw_payload_keys"] = list(payload.keys())
            else:
                return

            text = sentence.get("text") or sentence.get("result") or ""
            if not text:
                return

            # Times (prefer SDK begin_time / end_time)
            begin_time_raw = sentence.get("begin_time")
            end_time_raw = sentence.get("end_time")
            try:
                start_ms = int(begin_time_raw) if begin_time_raw is not None else self._rel_ms()
            except Exception:
                start_ms = self._rel_ms()
            try:
                end_ms = int(end_time_raw) if (is_final and end_time_raw is not None) else None
            except Exception:
                end_ms = self._rel_ms() if is_final else None

            sentence_id = sentence.get("sentence_id")
            lang_raw = sentence.get("language") or sentence.get("lang") or sentence.get("language_code")
            language_code = map_lang_to_ui(lang_raw, text)

            # Speaker extraction
            speaker_label, source_tag = extract_sentence_speaker(sentence)

            # Fake diarization fallback if absent
            if speaker_label is None and is_final and FAKE_DIARIZATION:
                speaker_label = self._assign_fake_speaker()
                source_tag = "fake"

            # Maintain last_final_speaker (could assist other heuristics later)
            if is_final and speaker_label:
                self.last_final_speaker = speaker_label

            # Unique IDs
            if is_final:
                unique_suffix = uuid.uuid4().hex[:8]
                if sentence_id is not None:
                    result_id = f"alibaba-seg-{sentence_id}-{unique_suffix}"
                else:
                    result_id = f"alibaba-seg-{self.final_counter}-{unique_suffix}"
                self.final_counter += 1
                self.current_partial_id = None
            else:
                if self.current_partial_id is None:
                    base = sentence_id if sentence_id is not None else self.final_counter
                    self.current_partial_id = f"alibaba-temp-{base}"
                result_id = self.current_partial_id

            translated_text: Optional[str] = None
            entities: Optional[List[dict]] = None
            if is_final:
                if DO_TRANSLATE and nlp and not language_code.startswith("en"):
                    translated_text = nlp.to_english(text, language_code)
                if DO_ENTITIES and nlp:
                    analysis_text = translated_text or text
                    entities = nlp.detect_entities(analysis_text)

            if DBG:
                raw_debug.update(
                    {
                        "sentence_id": sentence_id,
                        "is_final": is_final,
                        "speaker_extracted": speaker_label,
                        "speaker_source": source_tag,
                        "language_raw": lang_raw,
                        "language_final": language_code,
                        "begin_time": begin_time_raw,
                        "end_time": end_time_raw,
                        "result_id": result_id,
                        "final_counter": self.final_counter,
                        "fake_diarization": FAKE_DIARIZATION,
                    }
                )

            payload = build_aws_payload(
                text=text,
                is_partial=not is_final,
                result_id=result_id,
                speaker=speaker_label,
                language_code=language_code,
                start_ms=start_ms,
                end_ms=end_ms,
                translated_text=translated_text,
                entities=entities,
                debug=raw_debug if DBG else None,
            )
            self._emit(payload)

        except Exception as exc:
            logger.exception("Error in on_event: %s", exc)

# -------------------- WebSocket Endpoint --------------------
@router.websocket("/alibaba")
async def transcribe_alibaba(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected (/transcribe/alibaba)")
    os.environ.setdefault("DASHSCOPE_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    callback = ParaformerCallback(loop, queue)
    recognizer: Optional[Recognition] = None

    async def forward_events():
        try:
            while True:
                item = await queue.get()
                if isinstance(item, dict):
                    if item.get("_error"):
                        if websocket.client_state == WebSocketState.CONNECTED:
                            await websocket.close(code=1011, reason=item.get("message") or "Transcription service error")
                        break
                    if item.get("_complete"):
                        if websocket.client_state == WebSocketState.CONNECTED:
                            await websocket.close()
                        break
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text(json.dumps(item))
                    else:
                        break
        except asyncio.CancelledError:
            if DBG:
                logger.debug("forward_events cancelled")
        except Exception as e:
            logger.error("forward_events error: %s", e)

    async def receive_audio():
        try:
            while True:
                data = await websocket.receive_bytes()
                if not data:
                    # zero-length frame signals end
                    try:
                        if recognizer:
                            recognizer.send_end_flag()
                    except Exception:
                        pass
                    break
                try:
                    if recognizer:
                        recognizer.send_audio_frame(data)
                except Exception as e:
                    logger.error("send_audio_frame error: %s", e)
                    break
        except WebSocketDisconnect:
            logger.info("Client disconnected (WebSocketDisconnect)")
            try:
                if recognizer:
                    recognizer.send_end_flag()
            except Exception:
                pass
        except asyncio.CancelledError:
            if DBG:
                logger.debug("receive_audio cancelled")
        except Exception as e:
            logger.error("receive_audio error: %s", e)

    try:
        # Use correct diarization parameter name speaker_count (newer SDK)
        try:
            recognizer = Recognition(
                model="paraformer-realtime-v2",
                format="pcm",
                sample_rate=16000,
                diarization_enabled=True,
                speaker_count=SPEAKER_COUNT,
                language_hints=["en", "yue", "zh"],
                callback=callback,
                semantic_punctuation_enabled=False,
                punctuation_prediction_enabled=True,
                inverse_text_normalization_enabled=True,
            )
        except TypeError:
            # Fallback if SDK doesn't support speaker_count
            logger.warning("DashScope SDK missing speaker_count parameter; continuing without it.")
            recognizer = Recognition(
                model="paraformer-realtime-v2",
                format="pcm",
                sample_rate=16000,
                diarization_enabled=True,
                language_hints=["en", "yue", "zh"],
                callback=callback,
                semantic_punctuation_enabled=False,
                punctuation_prediction_enabled=True,
                inverse_text_normalization_enabled=True,
            )

        recognizer.start()
        logger.info(
            "DashScope recognizer started (diarization_enabled=True, speaker_count_hint=%s, fake_diarization=%s)",
            SPEAKER_COUNT,
            FAKE_DIARIZATION,
        )

        sender_task = asyncio.create_task(forward_events())
        audio_task = asyncio.create_task(receive_audio())

        done, pending = await asyncio.wait(
            {sender_task, audio_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except Exception:
                pass
        for task in done:
            if task.exception():
                logger.error("Task finished with exception: %s", task.exception())

    except Exception as e:
        logger.exception("Transcription session error: %s", e)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1011, reason=str(e))
            except Exception:
                pass
    finally:
        if recognizer:
            try:
                recognizer.stop()
            except Exception:
                pass
        logger.info("Alibaba transcription session ended")