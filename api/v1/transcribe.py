import asyncio
import json
import logging
import os
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from dashscope.audio.asr import Recognition

from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


def _to_aws_shape_from_alibaba(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Transform Alibaba Paraformer realtime message into the AWS-like structure
    expected by the frontend:
      { Transcript: { Results: [ { ResultId, IsPartial, Alternatives: [ { Transcript, Items: [...] } ], LanguageCode } ] },
        DisplayText, TranslatedText, ComprehendEntities }
    Returns None if the message doesn't contain a usable result.
    """
    header = msg.get("header") or {}
    payload = msg.get("payload") or {}

    text = payload.get("result")
    if not text:
        return None

    is_partial = (header.get("name") == "TranscriptionResultChanged")
    speaker_id = payload.get("speaker_id")
    sentence_id = payload.get("sentence_id", 0)
    language = payload.get("language")

    aws_payload = {
        "Transcript": {
            "Results": [
                {
                    "ResultId": f"alibaba-seg-{sentence_id}",
                    "IsPartial": is_partial,
                    "Alternatives": [
                        {
                            "Transcript": text,
                            "Items": (
                                [
                                    {
                                        "Type": "pronunciation",
                                        "Speaker": speaker_id,
                                    }
                                ]
                                if speaker_id
                                else []
                            ),
                        }
                    ],
                    "LanguageCode": language,
                }
            ]
        },
        "DisplayText": text,
        "TranslatedText": None,
        "ComprehendEntities": [],
    }
    return aws_payload


@router.websocket("/alibaba")
async def transcribe_alibaba(ws: WebSocket):
    """
    WebSocket proxy for Alibaba Paraformer real-time transcription.
    - Accepts 16 kHz 16-bit PCM frames from the browser.
    - Sends AWS-shaped JSON messages back to the browser.
    - Auto-detects among English, Cantonese, and Mandarin Traditional.
    """
    await ws.accept()

    # Always auto-detect between English, Cantonese, Mandarin
    language_hints: List[str] = ["en", "yue", "zh"]
    logger.info("Browser connected to Alibaba endpoint. Auto-detecting languages: %s", language_hints)

    if not settings.dashscope_api_key:
        logger.error("DASHSCOPE_API_KEY is not set.")
        await ws.close(code=1011, reason="Server not configured for this transcription service.")
        return

    # Ensure the SDK sees the key (dashscope reads env var)
    os.environ.setdefault("DASHSCOPE_API_KEY", settings.dashscope_api_key)

    # Queue that the sync callbacks will push into; our async task will consume and send to client
    event_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    class WsCallback:
        """
        DashScope calls these methods synchronously (no await).
        Use loop.call_soon_threadsafe to hand off work to the asyncio loop.
        """

        def __init__(self):
            self.task_id = None

        def on_open(self) -> None:
            logger.info("Alibaba Recognizer connection opened.")

        def on_error(self, message: str) -> None:
            logger.error("Alibaba Recognizer error: %s", message)
            # Signal the sender to close
            loop.call_soon_threadsafe(event_queue.put_nowait, {"_close": True, "_reason": "Transcription service error"})

        def on_close(self) -> None:
            logger.info("Alibaba Recognizer connection closed.")
            loop.call_soon_threadsafe(event_queue.put_nowait, {"_close": True, "_reason": "Transcription service closed"})

        def on_event(self, message: str) -> None:
            try:
                data = json.loads(message)
                transformed = _to_aws_shape_from_alibaba(data)
                if transformed:
                    loop.call_soon_threadsafe(event_queue.put_nowait, transformed)
            except Exception as e:
                logger.error("Error processing Alibaba message: %s", e)

    recognizer = None

    async def sender_task():
        # Consume transformed events and forward to browser
        try:
            while True:
                item = await event_queue.get()
                if isinstance(item, dict) and item.get("_close"):
                    # Close requested by callback
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.close(code=1011, reason=item.get("_reason", "Service closed"))
                    break
                if ws.client_state != WebSocketState.CONNECTED:
                    break
                await ws.send_text(json.dumps(item))
        except Exception as e:
            logger.error("Sender task error: %s", e)

    async def audio_task():
        # Forward audio from browser to Alibaba
        try:
            while True:
                chunk = await ws.receive_bytes()
                if not chunk:
                    break
                # Send PCM frame to recognizer
                if recognizer:
                    recognizer.send_audio_frame(chunk)
        except WebSocketDisconnect:
            logger.info("Client disconnected (WebSocketDisconnect).")
        except Exception as e:
            logger.error("Audio task error: %s", e)

    try:
        callback = WsCallback()
        recognizer = Recognition(
            model="paraformer-realtime-v2",
            format="pcm",
            sample_rate=16000,
            callback=callback,
            diarization_enabled=True,
            language_hints=language_hints,
        )

        recognizer.start()

        # Run audio forwarder and sender in parallel
        send_task = asyncio.create_task(sender_task())
        mic_task = asyncio.create_task(audio_task())

        done, pending = await asyncio.wait(
            {send_task, mic_task}, return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel the remaining task
        for t in pending:
            t.cancel()
            try:
                await t
            except Exception:
                pass

    except Exception as exc:
        logger.error("Alibaba WebSocket proxy failed: %s", exc)
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close(code=1011, reason="Internal server error.")
    finally:
        try:
            if recognizer:
                recognizer.stop()
        except Exception:
            pass
        logger.info("Alibaba browser session ended.")