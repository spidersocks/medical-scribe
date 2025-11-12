import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from dashscope.audio.asr import Recognition

from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Environment flag for verbose event logging
ALIBABA_DEBUG = os.getenv("ALIBABA_DEBUG", "0") == "1"

PARTIAL_EVENT_NAMES = {"TranscriptionResultChanged"}
FINAL_EVENT_NAMES = {"SentenceEnd"}
IGNORED_EVENT_NAMES = {"SentenceBegin"}
FAIL_EVENT_NAMES = {"TaskFailed", "RecognitionFailed"}

def _transform_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a DashScope event (header + payload) into AWS Transcribe-like JSON.
    Returns None if the event is not a transcription result.
    Expected shapes per docs:
      {
        "header": {"name": "...", ...},
        "payload": {
           "result": "text",
           "sentence_id": 3,
           "speaker_id": "spk_0",
           "language": "en",
           ...
        }
      }
    """
    if not isinstance(event, dict):
        return None

    header = event.get("header") or {}
    payload = event.get("payload") or {}
    if not payload:
        return None

    text = payload.get("result")
    if not isinstance(text, str) or not text.strip():
        return None

    name = header.get("name") or ""
    is_partial = name in PARTIAL_EVENT_NAMES
    is_final = name in FINAL_EVENT_NAMES

    # We treat partial vs final via override; for final segments IsPartial=False
    if not (is_partial or is_final):
        return None  # Not a usable transcript-bearing event

    sentence_id = payload.get("sentence_id", 0)
    speaker_id = payload.get("speaker_id")  # can be None for interim
    language = payload.get("language")

    # Build Items array (minimal: one pronunciation item for speaker labeling)
    items = []
    if speaker_id:
        items.append({"Type": "pronunciation", "Speaker": speaker_id})

    aws_payload = {
        "Transcript": {
            "Results": [
                {
                    "ResultId": f"alibaba-seg-{sentence_id}",
                    "IsPartial": is_partial,
                    "Alternatives": [
                        {
                            "Transcript": text,
                            "Items": items,
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


def _safe_to_dict(message: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize incoming message into a dict with header/payload if possible.
    Accepts:
      - JSON string
      - bytes (JSON)
      - dict already
      - object with .header /.payload
    """
    if message is None:
        return None
    if isinstance(message, (bytes, bytearray)):
        try:
            message = message.decode("utf-8", errors="ignore")
        except Exception:
            return None
    if isinstance(message, str):
        try:
            parsed = json.loads(message)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    if isinstance(message, dict):
        return message
    # Generic object path
    try:
        header = getattr(message, "header", None)
        payload = getattr(message, "payload", None)
        if isinstance(header, dict) and isinstance(payload, dict):
            return {"header": header, "payload": payload}
    except Exception:
        return None
    return None


@router.websocket("/alibaba")
async def transcribe_alibaba(ws: WebSocket):
    """
    WebSocket proxy: Browser sends 16 kHz 16-bit PCM frames.
    We forward frames to DashScope Paraformer real-time model and translate
    its callback events into AWS-like transcript JSON.
    """
    await ws.accept()

    language_hints = ["en", "yue", "zh"]
    logger.info("Alibaba WS connected. language_hints=%s", language_hints)

    if not settings.dashscope_api_key:
        logger.error("Missing DASHSCOPE_API_KEY.")
        await ws.close(code=1011, reason="Missing API key.")
        return

    os.environ.setdefault("DASHSCOPE_API_KEY", settings.dashscope_api_key)

    loop = asyncio.get_running_loop()
    outbound_queue: asyncio.Queue = asyncio.Queue()
    running = True

    class Callback:
        def on_open(self):
            logger.info("[DashScope] on_open")

        def on_error(self, message: str):
            logger.error("[DashScope] on_error: %s", message)
            loop.call_soon_threadsafe(
                outbound_queue.put_nowait,
                {"_close": True, "_reason": f"Service error: {message}"},
            )

        def on_close(self):
            logger.info("[DashScope] on_close")
            loop.call_soon_threadsafe(
                outbound_queue.put_nowait,
                {"_close": True, "_reason": "Service closed"},
            )

        def on_event(self, message: Any):
            """
            Core event dispatcher for dashscope 1.25.0 (recognition.py invokes this).
            """
            try:
                evt = _safe_to_dict(message)
                if not evt:
                    if ALIBABA_DEBUG:
                        logger.debug("[DashScope] Ignored non-parsable event: %r", type(message).__name__)
                    return

                name = (evt.get("header") or {}).get("name")
                if ALIBABA_DEBUG:
                    logger.debug("[DashScope] Event name=%s keys=%s", name, list((evt.get("payload") or {}).keys())[:10])

                if name in IGNORE_EVENT_NAMES:
                    if ALIBABA_DEBUG:
                        logger.debug("[DashScope] Ignored event %s", name)
                    return

                if name in FAIL_EVENT_NAMES:
                    reason = (evt.get("payload") or {}).get("error_message") or name
                    logger.error("[DashScope] Failure event: %s", reason)
                    loop.call_soon_threadsafe(
                        outbound_queue.put_nowait,
                        {"_close": True, "_reason": f"Recognition failed: {reason}"},
                    )
                    return

                transformed = _transform_event(evt)
                if not transformed:
                    return

                loop.call_soon_threadsafe(outbound_queue.put_nowait, transformed)
            except Exception as e:
                logger.error("on_event processing error: %s", e)

    # NOTE: small typo correction (IGNORE_EVENT_NAMES variable was used above)
    IGNORE_EVENT_NAMES = IGNORED_EVENT_NAMES

    recognizer: Optional[Recognition] = None

    async def forward_events():
        try:
            while running:
                msg = await outbound_queue.get()
                if isinstance(msg, dict) and msg.get("_close"):
                    if ws.client_state == WebSocketState.CONNECTED:
                        # Send an info message before closing so frontend can surface error
                        try:
                            info_msg = {"error": msg.get("_reason", "Service closed")}
                            await ws.send_text(json.dumps(info_msg))
                        except Exception:
                            pass
                        await ws.close(code=1011, reason=msg.get("_reason", "Service closed"))
                    break
                if ws.client_state != WebSocketState.CONNECTED:
                    break
                await ws.send_text(json.dumps(msg))
        except Exception as e:
            logger.error("forward_events error: %s", e)

    async def pump_audio():
        """
        Receive PCM frames from browser and forward to recognizer.
        Each Float32Array is converted to 16-bit PCM in the frontend already.
        """
        frame_count = 0
        try:
            while running:
                chunk = await ws.receive_bytes()
                if not chunk:
                    logger.info("Received empty chunk; treating as end-of-stream trigger.")
                    break
                frame_count += 1
                if recognizer:
                    recognizer.send_audio_frame(chunk)
                if ALIBABA_DEBUG and frame_count % 50 == 0:
                    logger.debug("[DashScope] Sent %d audio frames", frame_count)
        except WebSocketDisconnect:
            logger.info("Client disconnected (WebSocket).")
        except Exception as e:
            logger.error("pump_audio error: %s", e)

    try:
        cb = Callback()
        recognizer = Recognition(
            model="paraformer-realtime-v2",
            format="pcm",
            sample_rate=16000,
            callback=cb,
            diarization_enabled=True,
            language_hints=language_hints,
        )
        recognizer.start()

        events_task = asyncio.create_task(forward_events())
        audio_task = asyncio.create_task(pump_audio())

        done, pending = await asyncio.wait(
            {events_task, audio_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
            try:
                await t
            except Exception:
                pass

    except Exception as exc:
        logger.exception("Alibaba realtime session failed: %s", exc)
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close(code=1011, reason="Internal server error.")
    finally:
        running = False
        try:
            if recognizer:
                recognizer.stop()
        except Exception:
            pass
        logger.info("Alibaba browser session ended.")