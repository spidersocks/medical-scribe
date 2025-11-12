import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

logger = logging.getLogger(__name__)
router = APIRouter()

# Enable verbose logging for this module only when needed.
if os.getenv("DEBUG_ALIBABA_TRANSCRIBE") == "1":
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

logging.getLogger("dashscope").setLevel(logging.INFO)


def _map_lang_to_ui(language: Optional[str]) -> str:
    if not language:
        return "en-US"
    lang = language.lower()
    if lang.startswith("en"):
        return "en-US"
    if "yue" in lang or "cant" in lang:
        return "zh-HK"
    if lang.startswith("zh"):
        return "zh-TW"
    return "en-US"


def _aws_shape(text: str,
               is_partial: bool,
               result_id: str,
               speaker: Optional[str],
               language: Optional[str],
               debug: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "Transcript": {
            "Results": [
                {
                    "ResultId": result_id,
                    "IsPartial": is_partial,
                    "Alternatives": [
                        {
                            "Transcript": text,
                            "Items": (
                                [{"Type": "pronunciation", "Speaker": speaker}]
                                if speaker else []
                            ),
                        }
                    ],
                    "LanguageCode": _map_lang_to_ui(language),
                }
            ]
        },
        "DisplayText": text,
        "TranslatedText": None,
        "ComprehendEntities": [],
    }
    if debug and os.getenv("DEBUG_ALIBABA_TRANSCRIBE") == "1":
        payload["_debug"] = debug
    return payload


class ParaformerCallback(RecognitionCallback):
    """
    Handles both RecognitionResult objects and (if the SDK sends them) raw JSON strings.
    Emits AWS-shaped payloads onto an asyncio.Queue consumed by the websocket coroutine.
    """
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self.loop = loop
        self.queue = queue
        self.final_counter = 0
        self.current_partial_id: Optional[str] = None

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

    def on_event(self, message: Any):
        """
        message may be:
          - RecognitionResult
          - str (JSON)
        """
        try:
            raw_debug: Dict[str, Any] = {}
            sentence: Dict[str, Any] = {}
            speaker = None
            language = None
            sentence_id = None
            is_final = False
            text = ""

            # Case 1: RecognitionResult instance
            if isinstance(message, RecognitionResult):
                sentence = message.get_sentence()
                raw_debug["raw_sentence"] = sentence
                if not isinstance(sentence, dict):
                    return
                text = sentence.get("text") or sentence.get("result") or ""
                speaker = sentence.get("speaker") or sentence.get("speaker_id")
                language = sentence.get("language")
                sentence_id = sentence.get("sentence_id")

                # Try official method first
                try:
                    is_final = message.is_sentence_end(sentence)
                except Exception:
                    # Fallback flags sometimes appear
                    is_final = bool(sentence.get("is_sentence_end") or sentence.get("sentence_end"))

            # Case 2: raw JSON string (old style header/payload from earlier version)
            elif isinstance(message, str):
                try:
                    parsed = json.loads(message)
                    raw_debug["raw_json"] = parsed
                    header = parsed.get("header", {})
                    payload = parsed.get("payload", {})
                    text = payload.get("result") or payload.get("text") or ""
                    speaker = payload.get("speaker_id") or payload.get("speaker")
                    language = payload.get("language")
                    sentence_id = payload.get("sentence_id")

                    name = header.get("name")
                    # Names from docs: TranscriptionResultChanged (partial), SentenceEnd (final)
                    if name == "SentenceEnd":
                        is_final = True
                    elif name == "TranscriptionResultChanged":
                        is_final = False
                    else:
                        # If we don't recognize the header name, ignore
                        return
                except Exception as exc:
                    logger.debug("Failed to parse raw string event: %s", exc)
                    return
            else:
                # Unknown type
                return

            if not text:
                return  # nothing to emit

            # ID generation
            if is_final:
                # ALWAYS unique: uuid4 hex + counter (counter is just for debug ordering)
                unique_part = uuid.uuid4().hex[:8]
                if sentence_id is not None:
                    result_id = f"alibaba-seg-{sentence_id}-{unique_part}"
                else:
                    result_id = f"alibaba-seg-{self.final_counter}-{unique_part}"
                self.final_counter += 1
                self.current_partial_id = None
            else:
                # Stable temp id for partials of the current sentence
                if self.current_partial_id is None:
                    base = sentence_id if sentence_id is not None else self.final_counter
                    self.current_partial_id = f"alibaba-temp-{base}"
                result_id = self.current_partial_id

            debug_block = {
                "is_final": is_final,
                "result_id": result_id,
                "sentence_id": sentence_id,
                "final_counter": self.final_counter,
                "speaker": speaker,
                "language": language,
            }
            debug_block.update(raw_debug)

            logger.debug(
                "[DashScope] Emit %s id=%s sentence_id=%s text=%r",
                "FINAL" if is_final else "PARTIAL",
                result_id,
                sentence_id,
                text[:120],
            )

            aws_payload = _aws_shape(
                text=text,
                is_partial=not is_final,
                result_id=result_id,
                speaker=speaker,
                language=language,
                debug=debug_block,
            )
            self._emit(aws_payload)

        except Exception as e:
            logger.exception("Error in on_event: %s", e)


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
                        await websocket.close(code=1011, reason=item.get("message") or "Transcription service error")
                    break
                if isinstance(item, dict) and item.get("_complete"):
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.close()
                    break
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(item))
                else:
                    break
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