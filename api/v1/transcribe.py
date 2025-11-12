import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

logger = logging.getLogger(__name__)
router = APIRouter()

# Make DashScope a bit quieter unless debugging
logging.getLogger("dashscope").setLevel(logging.INFO)


def _map_lang_to_ui(language: Optional[str]) -> str:
    # Map Alibaba language tokens to the UI's expected codes
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


def _aws_shape(text: str, is_partial: bool, result_id: str, speaker: Optional[str], language: Optional[str]) -> Dict[str, Any]:
    return {
        "Transcript": {
            "Results": [
                {
                    "ResultId": result_id,
                    "IsPartial": is_partial,
                    "Alternatives": [
                        {
                            "Transcript": text,
                            "Items": (
                                [
                                    {
                                        "Type": "pronunciation",
                                        "Speaker": speaker,
                                    }
                                ]
                                if speaker
                                else []
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


class ParaformerCallback(RecognitionCallback):
    """
    DashScope invokes these synchronously; we push transformed events onto an asyncio.Queue
    which the WebSocket coroutine drains.
    """
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self.loop = loop
        self.queue = queue
        self.final_counter = 0
        self.current_partial_id: Optional[str] = None

    def on_open(self):
        logger.info("[DashScope] Connection opened")

    def on_error(self, message: str):
        logger.error("[DashScope] Error: %s", message)
        self.loop.call_soon_threadsafe(self.queue.put_nowait, {"_error": True, "message": message})

    def on_close(self):
        logger.info("[DashScope] Connection closed")

    def on_complete(self):
        logger.info("[DashScope] Recognition completed")
        self.loop.call_soon_threadsafe(self.queue.put_nowait, {"_complete": True})

    def on_event(self, message: Any):
        try:
            if not isinstance(message, RecognitionResult):
                return

            sentence = message.get_sentence()
            if not isinstance(sentence, dict):
                return

            # Prefer normalized 'text'; fall back to 'result' if present
            text = sentence.get("text") or sentence.get("result") or ""
            if not text:
                return

            # Determine final vs partial
            try:
                is_final = message.is_sentence_end(sentence)
            except Exception:
                # Fallback heuristic if SDK differs
                is_final = bool(sentence.get("is_sentence_end") or sentence.get("sentence_end"))

            speaker = sentence.get("speaker") or sentence.get("speaker_id")
            language = sentence.get("language")
            sent_id = sentence.get("sentence_id")

            if is_final:
                # Unique id for EVERY finalized utterance
                if sent_id is not None:
                    result_id = f"alibaba-seg-{sent_id}"
                else:
                    result_id = f"alibaba-seg-{self.final_counter}"
                self.final_counter += 1
                self.current_partial_id = None
            else:
                # Stable temp id across partial updates of the same sentence
                if self.current_partial_id is None:
                    temp_index = sent_id if sent_id is not None else self.final_counter
                    self.current_partial_id = f"alibaba-temp-{temp_index}"
                result_id = self.current_partial_id

            payload = _aws_shape(
                text=text,
                is_partial=not is_final,
                result_id=result_id,
                speaker=speaker,
                language=language,
            )

            # For quick debugging of overwrites:
            logger.debug("[DashScope] Emit %s id=%s text=%r", "FINAL" if is_final else "PARTIAL", result_id, text[:80])

            self.loop.call_soon_threadsafe(self.queue.put_nowait, payload)

        except Exception as e:
            logger.exception("Error in on_event: %s", e)


@router.websocket("/alibaba")
async def transcribe_alibaba(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected (/transcribe/alibaba)")

    # Ensure the DashScope API key is available to the SDK; harmless if already set in env
    os.environ.setdefault("DASHSCOPE_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    callback = ParaformerCallback(loop, queue)
    recognizer = None

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
                    # End-of-stream from client
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
            language_hints=["en", "yue", "zh"],  # auto-detect among English, Cantonese, Mandarin
            callback=callback,
            semantic_punctuation_enabled=False,
            punctuation_prediction_enabled=True,
            inverse_text_normalization_enabled=True,
        )
        recognizer.start()
        logger.info("DashScope recognizer started")

        sender_task = asyncio.create_task(forward_events())
        audio_task = asyncio.create_task(receive_audio())

        done, pending = await asyncio.wait({sender_task, audio_task}, return_when=asyncio.FIRST_COMPLETED)
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