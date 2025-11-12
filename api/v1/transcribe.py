import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

# Lightweight translation / entities helpers (wrap AWS)
try:
    from services import nlp
except ImportError:
    nlp = None  # allow running without enrichment

logger = logging.getLogger(__name__)
router = APIRouter()

DBG = os.getenv("DEBUG_ALIBABA_TRANSCRIBE") == "1"
DO_TRANSLATE = os.getenv("ENABLE_ALIBABA_TRANSLATION") == "1"
DO_ENTITIES = os.getenv("ENABLE_ALIBABA_ENTITIES") == "1"

if DBG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

logging.getLogger("dashscope").setLevel(logging.INFO)


# --- Language mapping & heuristics -------------------------------------------------
CANTONESE_HINT_CHARS = set("嘅咩喺咗冇哋啲呢啦噉仲咪睇醫")  # crude heuristic

def _map_lang_to_ui(language: Optional[str], text: str = "") -> str:
    if language:
        lang = language.lower()
        if lang.startswith("en"):
            return "en-US"
        if "yue" in lang or "cant" in lang:
            return "zh-HK"
        if lang.startswith("zh"):
            # prefer Traditional for generic "zh"
            return "zh-TW"
    # Heuristic based on character sets
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        # Try differentiate Cantonese vs Mandarin
        if any(ch in CANTONESE_HINT_CHARS for ch in text):
            return "zh-HK"
        return "zh-TW"
    return "en-US"


# --- AWS-shaped payload builder ----------------------------------------------------
def _aws_shape(
    text: str,
    is_partial: bool,
    result_id: str,
    speaker: Optional[str],
    language_code: str,
    start_ms: Optional[int],
    end_ms: Optional[int],
    translated_text: Optional[str],
    entities: Optional[List[dict]],
    debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the AWS-like payload consumed by existing frontend logic.
    """
    items = []
    if speaker:
        # Provide a simple single "pronunciation" item to retain speaker label parity
        items.append({"Type": "pronunciation", "Speaker": speaker})

    result_obj = {
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
    if start_ms is not None:
        result_obj["StartTimeMs"] = start_ms
    if end_ms is not None:
        result_obj["EndTimeMs"] = end_ms
    # Duplicate speaker at top level of result for simpler extraction (new)
    if speaker:
        result_obj["Speaker"] = speaker

    payload = {
        "Transcript": {"Results": [result_obj]},
        "DisplayText": text,
        "TranslatedText": translated_text,
        "ComprehendEntities": entities or [],
    }
    if debug and DBG:
        payload["_debug"] = debug
    return payload


# --- Callback ----------------------------------------------------------------------
class ParaformerCallback(RecognitionCallback):
    """
    Converts DashScope events into AWS-shaped JSON.
    Handles both RecognitionResult objects and raw JSON strings.
    Accumulates timing info.
    """
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self.loop = loop
        self.queue = queue
        self.final_counter = 0
        self.current_partial_id: Optional[str] = None
        self.session_start = time.monotonic()
        self.sentence_start_monotonic: Optional[float] = None

    def _emit(self, data: Dict[str, Any]) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, data)

    def on_open(self):
        logger.info("[DashScope] Connection opened")

    def on_error(self, message: str):
        logger.error("[DashScope] Error: %s", message)
        self._emit({"_error": True, "message": message})

    def on_close(self):
        logger.info("[DashScope] Connection closed")

    def on_complete(self):
        logger.info("[DashScope] Recognition completed")
        self._emit({"_complete": True})

    # Utility to compute ms relative to session start
    def _rel_ms(self, monotonic_time: float) -> int:
        return int((monotonic_time - self.session_start) * 1000)

    def on_event(self, message: Any):
        try:
            raw_debug: Dict[str, Any] = {}
            speaker = None
            language = None
            sentence_id = None
            is_final = False
            text = ""

            # Extract from RecognitionResult objects
            if isinstance(message, RecognitionResult):
                sentence = message.get_sentence()
                raw_debug["raw_sentence"] = sentence
                if not isinstance(sentence, dict):
                    return
                text = sentence.get("text") or sentence.get("result") or ""
                speaker = sentence.get("speaker") or sentence.get("speaker_id")
                language = (
                    sentence.get("language")
                    or sentence.get("lang")
                    or sentence.get("language_code")
                )
                sentence_id = sentence.get("sentence_id")

                try:
                    is_final = message.is_sentence_end(sentence)
                except Exception:
                    is_final = bool(sentence.get("is_sentence_end") or sentence.get("sentence_end"))

            # Extract from raw JSON strings
            elif isinstance(message, str):
                try:
                    parsed = json.loads(message)
                except Exception:
                    return
                raw_debug["raw_json"] = parsed
                header = parsed.get("header", {})
                payload = parsed.get("payload", {})
                name = header.get("name")
                if name == "SentenceEnd":
                    is_final = True
                elif name == "TranscriptionResultChanged":
                    is_final = False
                else:
                    return

                text = payload.get("result") or payload.get("text") or ""
                speaker = payload.get("speaker_id") or payload.get("speaker")
                language = (
                    payload.get("language")
                    or payload.get("lang")
                    or payload.get("language_code")
                )
                sentence_id = payload.get("sentence_id")

            else:
                return  # unknown type

            if not text:
                return

            # Timing handling
            now_mono = time.monotonic()
            if self.sentence_start_monotonic is None:
                self.sentence_start_monotonic = now_mono  # first partial of sentence

            start_ms: Optional[int] = self._rel_ms(self.sentence_start_monotonic)
            end_ms: Optional[int] = None

            if is_final:
                end_ms = self._rel_ms(now_mono)

                # Unique id for every final
                unique = uuid.uuid4().hex[:8]
                if sentence_id is not None:
                    result_id = f"alibaba-seg-{sentence_id}-{unique}"
                else:
                    result_id = f"alibaba-seg-{self.final_counter}-{unique}"
                self.final_counter += 1
                self.current_partial_id = None
                # Reset sentence timing for next one
                self.sentence_start_monotonic = None
            else:
                # Stable temp id
                if self.current_partial_id is None:
                    base = sentence_id if sentence_id is not None else self.final_counter
                    self.current_partial_id = f"alibaba-temp-{base}"
                result_id = self.current_partial_id

            language_code = _map_lang_to_ui(language, text)

            translated_text: Optional[str] = None
            entities: Optional[List[dict]] = None

            if is_final:
                # Optional translation
                if DO_TRANSLATE and nlp and not language_code.startswith("en"):
                    translated_text = nlp.to_english(text, language_code)
                # Optional entities (on English text)
                if DO_ENTITIES and nlp:
                    analysis_text = translated_text or text
                    entities = nlp.detect_entities(analysis_text)
                raw_debug.update(
                    {
                        "final_counter": self.final_counter,
                        "sentence_id": sentence_id,
                        "language_inferred": language_code,
                        "speaker": speaker,
                    }
                )
            else:
                raw_debug.update(
                    {
                        "partial_id": result_id,
                        "sentence_id": sentence_id,
                        "language_inferred": language_code,
                        "speaker": speaker,
                    }
                )

            payload = _aws_shape(
                text=text,
                is_partial=not is_final,
                result_id=result_id,
                speaker=speaker,
                language_code=language_code,
                start_ms=start_ms,
                end_ms=end_ms,
                translated_text=translated_text,
                entities=entities,
                debug=raw_debug if DBG else None,
            )
            self._emit(payload)

        except Exception as e:
            logger.exception("Error in on_event: %s", e)


# --- WebSocket endpoint -------------------------------------------------------------
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
                if isinstance(item, dict) and item.get("_error"):
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.close(
                            code=1011,
                            reason=item.get("message") or "Transcription service error",
                        )
                    break
                if isinstance(item, dict) and item.get("_complete"):
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.close()
                    break
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(item))
                else:
                    break
        except asyncio.CancelledError:
            # Normal during shutdown
            if DBG:
                logger.debug("forward_events task cancelled")
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
                logger.debug("receive_audio task cancelled")
        except Exception as e:
            logger.error("receive_audio error: %s", e)

    try:
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
        logger.info("DashScope recognizer started")

        sender_task = asyncio.create_task(forward_events())
        audio_task = asyncio.create_task(receive_audio())

        done, pending = await asyncio.wait(
            {sender_task, audio_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
            try:
                await t
            except Exception:
                pass

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