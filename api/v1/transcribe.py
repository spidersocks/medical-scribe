import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

# Optional enrichment (translation + entities via AWS services)
try:
    from services import nlp
except ImportError:
    nlp = None

router = APIRouter()
logger = logging.getLogger(__name__)

DBG = os.getenv("DEBUG_ALIBABA_TRANSCRIBE") == "1"
DO_TRANSLATE = os.getenv("ENABLE_ALIBABA_TRANSLATION") == "1"
DO_ENTITIES = os.getenv("ENABLE_ALIBABA_ENTITIES") == "1"

if DBG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

logging.getLogger("dashscope").setLevel(logging.INFO)

# --- Language mapping / heuristic (unchanged) ---
CANTONESE_HINT_CHARS = set("嘅咩喺咗冇哋啲呢啦噉仲咪睇醫")
def map_lang_to_ui(lang_raw: Optional[str], text: str) -> str:
    if lang_raw:
        lr = lang_raw.lower()
        if lr.startswith("en"):
            return "en-US"
        if "yue" in lr or "cant" in lr:
            return "zh-HK"
        if lr.startswith("zh"):
            return "zh-TW"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        if any(ch in CANTONESE_HINT_CHARS for ch in text):
            return "zh-HK"
        return "zh-TW"
    return "en-US"

# --- AWS-shaped payload builder ---
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
        "Alternatives": [{"Transcript": text, "Items": items}],
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

# --- Callback without diarization ---
class ParaformerCallback(RecognitionCallback):
    """
    Streams Alibaba recognition events into an asyncio.Queue as AWS-shaped payloads.
    Since the model does not support speaker diarization, we assign placeholder
    speaker labels only to FINAL sentences: sentence_1, sentence_2, ...
    """
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self.loop = loop
        self.queue = queue
        self.session_start_monotonic = time.monotonic()
        self.final_counter = 0
        self.current_partial_id: Optional[str] = None

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

    def _rel_ms(self) -> int:
        return int((time.monotonic() - self.session_start_monotonic) * 1000)

    def _parse_raw_json(self, raw_str: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(raw_str)
        except Exception:
            return None
        header = obj.get("header") or {}
        payload = obj.get("payload") or {}
        name = header.get("name")
        if name not in ("TranscriptionResultChanged", "SentenceEnd"):
            return None
        payload["_is_final_name"] = (name == "SentenceEnd")
        return payload

    def on_event(self, message: Any):
        try:
            sentence: Dict[str, Any] = {}
            is_final = False
            dbg: Dict[str, Any] = {}

            if isinstance(message, RecognitionResult):
                sent = message.get_sentence()
                if not isinstance(sent, dict):
                    return
                sentence = sent
                dbg["raw_sentence_keys"] = list(sent.keys())
                try:
                    is_final = message.is_sentence_end(sent)
                except Exception:
                    is_final = bool(sent.get("sentence_end") or sent.get("is_sentence_end"))
            elif isinstance(message, str):
                parsed = self._parse_raw_json(message)
                if not parsed:
                    return
                sentence = parsed
                is_final = bool(sentence.get("_is_final_name"))
                dbg["raw_payload_keys"] = [k for k in sentence.keys() if k != "_is_final_name"]
            else:
                return

            text = sentence.get("text") or sentence.get("result") or ""
            if not text:
                return

            # Timing
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

            # --- NEW: Convert to Traditional Chinese if applicable ---
            if is_final and language_code.startswith("zh") and nlp:
                text = nlp.to_traditional_chinese(text)

            # Placeholder speaker only for finalized sentences
            speaker_label: Optional[str] = None
            if is_final:
                self.final_counter += 1
                speaker_label = f"sentence_{self.final_counter}"

            # Unique IDs
            if is_final:
                rid = f"alibaba-seg-{sentence_id if sentence_id is not None else self.final_counter}-{uuid.uuid4().hex[:8]}"
                self.current_partial_id = None
            else:
                if self.current_partial_id is None:
                    base = sentence_id if sentence_id is not None else self.final_counter
                    self.current_partial_id = f"alibaba-temp-{base}"
                rid = self.current_partial_id

            translated_text: Optional[str] = None
            entities: Optional[List[dict]] = None
            if is_final:
                if DO_TRANSLATE and nlp and not language_code.startswith("en"):
                    translated_text = nlp.to_english(text, language_code)
                if DO_ENTITIES and nlp:
                    analysis = translated_text or text
                    entities = nlp.detect_entities(analysis)

            if DBG:
                dbg.update({
                    "sentence_id": sentence_id,
                    "is_final": is_final,
                    "placeholder_speaker": speaker_label,
                    "language_raw": lang_raw,
                    "language_final": language_code,
                    "result_id": rid,
                    "final_counter": self.final_counter,
                })

            payload = build_aws_payload(
                text=text,
                is_partial=not is_final,
                result_id=rid,
                speaker=speaker_label,
                language_code=language_code,
                start_ms=start_ms,
                end_ms=end_ms,
                translated_text=translated_text,
                entities=entities,
                debug=dbg if DBG else None,
            )
            self._emit(payload)

        except Exception as exc:
            logger.exception("Error in on_event: %s", exc)

# --- WebSocket endpoint ---
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
        recognizer = Recognition(
            model="paraformer-realtime-v2",
            format="pcm",
            sample_rate=16000,
            # diarization removed – model unsupported
            diarization_enabled=False,
            language_hints=["en", "yue", "zh"],
            callback=callback,
            semantic_punctuation_enabled=False,
            punctuation_prediction_enabled=True,
            inverse_text_normalization_enabled=True,
        )
        recognizer.start()
        logger.info("DashScope recognizer started (placeholder speakers)")

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