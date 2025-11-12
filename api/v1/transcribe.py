import asyncio
import json
import logging
import os
from typing import List, Optional, Dict, Any
from collections.abc import Mapping

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


def _safe_get(d: Any, key: str, default=None):
    """Get key from dict-like object without raising KeyError."""
    try:
        if isinstance(d, dict):
            return d.get(key, default)
        if isinstance(d, Mapping):
            return d.get(key, default)
        # Some SDK objects implement .get
        if hasattr(d, "get") and callable(getattr(d, "get")):
            return d.get(key, default)
    except Exception:
        pass
    return default


def _word_to_dict(w: Any) -> Dict[str, Any]:
    """Normalize a word object/dict into a plain dict with text/times/speaker."""
    if isinstance(w, dict):
        return {
            "text": w.get("text") or w.get("content") or w.get("word") or "",
            "begin_time": w.get("begin_time") or w.get("start") or w.get("start_time") or w.get("StartTimeMs"),
            "end_time": w.get("end_time") or w.get("end") or w.get("end_time") or w.get("EndTimeMs"),
            "speaker": w.get("speaker") or w.get("spk") or w.get("Speaker"),
        }
    # object with attributes
    out = {
        "text": getattr(w, "text", "") or getattr(w, "content", "") or getattr(w, "word", ""),
        "begin_time": getattr(w, "begin_time", None) or getattr(w, "start", None) or getattr(w, "start_time", None),
        "end_time": getattr(w, "end_time", None) or getattr(w, "end", None) or getattr(w, "end_time", None),
        "speaker": getattr(w, "speaker", None) or getattr(w, "spk", None),
    }
    return out


def _sentence_to_dict(s: Any) -> Optional[Dict[str, Any]]:
    """Normalize a sentence object/dict into a plain dict with the expected keys."""
    if s is None:
        return None
    if isinstance(s, dict):
        # already dict-like
        words = s.get("words") or []
        words_norm = [_word_to_dict(w) for w in words] if isinstance(words, list) else []
        return {
            "text": s.get("text") or s.get("result") or "",
            "begin_time": s.get("begin_time"),
            "end_time": s.get("end_time"),
            "words": words_norm,
            "sentence_end": bool(s.get("sentence_end") or s.get("is_final") or s.get("final")),
            "speaker": s.get("speaker"),
            "language": s.get("language"),
        }
    # object path
    try:
        words = getattr(s, "words", None) or []
        words_norm = [_word_to_dict(w) for w in words] if isinstance(words, list) else []
        return {
            "text": getattr(s, "text", "") or getattr(s, "result", "") or "",
            "begin_time": getattr(s, "begin_time", None),
            "end_time": getattr(s, "end_time", None),
            "words": words_norm,
            "sentence_end": bool(getattr(s, "sentence_end", False) or getattr(s, "is_final", False)),
            "speaker": getattr(s, "speaker", None),
            "language": getattr(s, "language", None),
        }
    except Exception:
        return None


def _extract_sentence(msg: Any) -> Optional[Dict[str, Any]]:
    """
    Extract a normalized 'sentence' dict from:
      - RecognitionResult object (msg.sentence)
      - Plain dict with 'sentence' OR already a sentence dict
      - JSON string
    Returns None if not found or not usable.
    """
    # JSON string path
    if isinstance(msg, (str, bytes, bytearray)):
        try:
            parsed = json.loads(msg.decode() if isinstance(msg, (bytes, bytearray)) else msg)
        except Exception:
            return None
        return _extract_sentence(parsed)

    # RecognitionResult object path
    if RecognitionResult and isinstance(msg, RecognitionResult):
        return _sentence_to_dict(getattr(msg, "sentence", None))

    # Generic object with .sentence attribute
    if hasattr(msg, "sentence"):
        return _sentence_to_dict(getattr(msg, "sentence"))

    # Mapping/dict path
    if isinstance(msg, dict) or isinstance(msg, Mapping):
        maybe_sentence = _safe_get(msg, "sentence")
        if isinstance(maybe_sentence, (dict, Mapping)) or hasattr(maybe_sentence, "__dict__"):
            return _sentence_to_dict(maybe_sentence)
        # Some SDK variants flatten the sentence at top-level
        keys = set(getattr(msg, "keys", lambda: [])())
        if keys & {"text", "begin_time", "end_time", "words"}:
            return _sentence_to_dict(msg)
        # Some send {"result": "...", "sentence_end": bool}
        if "result" in keys or "sentence_end" in keys:
            return _sentence_to_dict(msg)

    return None


def _to_aws_shape(sentence: Dict[str, Any], result_counter: int) -> Dict[str, Any]:
    """
    Convert a DashScope 'sentence' into the AWS-like structure your frontend expects.
    sentence_end == True -> final result (IsPartial = False)
    sentence_end == False -> partial (IsPartial = True)
    """
    if not sentence:
        return {}

    text = (sentence.get("text") or "").strip()
    if not text:
        return {}

    sentence_end = bool(sentence.get("sentence_end"))
    is_partial = not sentence_end  # invert
    language = sentence.get("language")

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
              - dict already shaped like {sentence:{...}}
              - JSON string
            """
            nonlocal result_counter
            try:
                # Minimal structural debug without logging PHI
                try:
                    cls = type(message).__name__
                    debug_info = {}
                    if isinstance(message, Mapping):
                        keys = list(getattr(message, "keys", lambda: [])())
                        debug_info["keys"] = keys[:8]
                    else:
                        debug_info["has_sentence_attr"] = hasattr(message, "sentence")
                        debug_info["has_to_dict"] = hasattr(message, "to_dict")
                    logger.debug("DashScope on_event type=%s info=%s", cls, debug_info)
                except Exception:
                    pass

                sentence = _extract_sentence(message)
                if not sentence:
                    return

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