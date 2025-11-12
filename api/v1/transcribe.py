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

# Try to import RecognitionResult type for isinstance checks (may vary by SDK version)
try:
    from dashscope.audio.asr.recognition import RecognitionResult  # type: ignore
except Exception:  # pragma: no cover
    RecognitionResult = None  # type: ignore


def _extract_sentence(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Extract the 'sentence' dict from:
      - RecognitionResult object (obj.sentence)
      - Plain dict with 'sentence'
      - Already a sentence dict
    Returns None if not found.
    """
    if obj is None:
        return None
    # RecognitionResult object
    if RecognitionResult and isinstance(obj, RecognitionResult):
        return getattr(obj, "sentence", None)
    # Dict wrapper
    if isinstance(obj, dict):
        if "sentence" in obj and isinstance(obj["sentence"], dict):
            return obj["sentence"]
        # Maybe it's already the sentence
        sentence_keys = {"text", "begin_time", "end_time", "words"}
        if sentence_keys.intersection(obj.keys()):
            return obj
    return None


def _to_aws_shape(sentence: Dict[str, Any], result_counter: int) -> Dict[str, Any]:
    """
    Convert a DashScope 'sentence' into the AWS-like structure your frontend expects.
    sentence_end == True -> final result (IsPartial = False)
    sentence_end == False -> partial (IsPartial = True)
    """
    if not sentence:
        return {}

    text = sentence.get("text") or ""
    if not text.strip():
        return {}

    sentence_end = bool(sentence.get("sentence_end"))
    is_partial = not sentence_end  # invert

    # Speaker extraction: prefer sentence-level speaker, else first word with speaker
    speaker = sentence.get("speaker")
    if not speaker:
        for w in sentence.get("words") or []:
            w_s = w.get("speaker")
            if w_s:
                speaker = w_s
                break

    # Build Items from words (optional timing)
    items: List[Dict[str, Any]] = []
    for w in sentence.get("words") or []:
        w_text = w.get("text")
        if not w_text:
            continue
        begin = w.get("begin_time")
        end = w.get("end_time")
        # AWS Items expect Type + Transcript-like text per word (we only need Speaker for UI)
        itm: Dict[str, Any] = {
            "Type": "pronunciation",
            "Content": w_text,
        }
        if speaker:
            itm["Speaker"] = speaker
        if isinstance(begin, int):
            itm["StartTimeMs"] = begin
        if isinstance(end, int):
            itm["EndTimeMs"] = end
        items.append(itm)

    if speaker and not items:
        # Fallback single item so UI can still read speaker
        items.append({"Type": "pronunciation", "Speaker": speaker})

    aws_payload = {
        "Transcript": {
            "Results": [
                {
                    "ResultId": f"alibaba-seg-{result_counter}",
                    "IsPartial": is_partial,
                    "Alternatives": [
                        {
                            "Transcript": text,
                            "Items": items,
                        }
                    ],
                    # LanguageCode is unknown here; Paraformer may not return it per sentence.
                    "LanguageCode": None,
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

    Incoming audio: raw 16-bit PCM (16000 Hz).
    Outgoing messages: AWS Transcribe-compatible JSON blocks consumed by existing frontend.
    """
    await ws.accept()

    language_hints: List[str] = ["en", "yue", "zh"]  # English, Cantonese, Mandarin
    logger.info("Browser connected to Alibaba endpoint. Auto-detecting languages: %s", language_hints)

    if not settings.dashscope_api_key:
        logger.error("DASHSCOPE_API_KEY is not set.")
        await ws.close(code=1011, reason="Server not configured for this transcription service.")
        return

    # Ensure DashScope sees the key
    os.environ.setdefault("DASHSCOPE_API_KEY", settings.dashscope_api_key)

    event_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # Counter for creating stable ResultIds
    result_counter = 0

    class WsCallback:
        """
        DashScope invokes these synchronously.
        We push work into the asyncio event loop via call_soon_threadsafe.
        """

        def on_open(self) -> None:
            logger.info("Alibaba Recognizer connection opened.")

        def on_error(self, message: str) -> None:
            logger.error("Alibaba Recognizer error: %s", message)
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                {"_close": True, "_reason": f"Service error: {message}"},
            )

        def on_close(self) -> None:
            logger.info("Alibaba Recognizer connection closed (callback).")
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                {"_close": True, "_reason": "Service closed"},
            )

        def on_event(self, message: Any) -> None:
            """
            message may be:
              - RecognitionResult object
              - dict with 'sentence'
              - dict already shaped like your example
              - JSON string (older versions / alternate mode)
            """
            nonlocal result_counter
            try:
                parsed: Any = message
                if isinstance(message, str):
                    # JSON string path
                    parsed = json.loads(message)

                sentence = _extract_sentence(parsed)
                if not sentence:
                    # Could be usage/events we ignore
                    return

                # Increment counter only when we have a sentence with text
                if sentence.get("text"):
                    result_counter += 1

                transformed = _to_aws_shape(sentence, result_counter)
                if transformed:
                    loop.call_soon_threadsafe(event_queue.put_nowait, transformed)
            except Exception as e:
                logger.error("Error processing Alibaba message: %s", e)

    recognizer: Optional[Recognition] = None

    async def sender_task():
        try:
            while True:
                item = await event_queue.get()
                if isinstance(item, dict) and item.get("_close"):
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.close(code=1011, reason=item.get("_reason", "Service closed"))
                    break
                if ws.client_state != WebSocketState.CONNECTED:
                    break
                await ws.send_text(json.dumps(item))
        except Exception as e:
            logger.error("Sender task error: %s", e)

    async def audio_task():
        try:
            while True:
                chunk = await ws.receive_bytes()
                if not chunk:
                    break
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

        send_task = asyncio.create_task(sender_task())
        mic_task = asyncio.create_task(audio_task())

        done, pending = await asyncio.wait({send_task, mic_task}, return_when=asyncio.FIRST_COMPLETED)
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