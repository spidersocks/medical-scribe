import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from dashscope.audio.asr import Recognition

from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Attempt type import (not strictly required)
try:
    from dashscope.audio.asr.recognition import RecognitionResult  # type: ignore
except Exception:  # pragma: no cover
    RecognitionResult = None  # type: ignore


def _normalize_sentence(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Accepts:
      - RecognitionResult with .sentence
      - dict with 'sentence'
      - already a sentence-shaped dict
    Returns normalized dict or None.
    """
    if obj is None:
        return None

    # RecognitionResult path
    if RecognitionResult and isinstance(obj, RecognitionResult):
        raw = getattr(obj, "sentence", None)
        return _normalize_sentence(raw)

    # dict path with nested 'sentence'
    if isinstance(obj, dict):
        if "sentence" in obj and isinstance(obj["sentence"], dict):
            return _normalize_sentence(obj["sentence"])
        # Flattened structure
        keys = set(obj.keys())
        if {"text", "begin_time", "end_time"}.intersection(keys):
            # It may or may not have words / sentence_end
            return {
                "text": obj.get("text") or obj.get("result") or "",
                "begin_time": obj.get("begin_time"),
                "end_time": obj.get("end_time"),
                "words": obj.get("words") or [],
                "sentence_end": bool(obj.get("sentence_end") or obj.get("is_final") or obj.get("final")),
                "speaker": obj.get("speaker"),
                "language": obj.get("language"),
            }
        if "result" in obj or "sentence_end" in obj:
            return {
                "text": obj.get("result") or "",
                "begin_time": obj.get("begin_time"),
                "end_time": obj.get("end_time"),
                "words": obj.get("words") or [],
                "sentence_end": bool(obj.get("sentence_end")),
                "speaker": obj.get("speaker"),
                "language": obj.get("language"),
            }
        return None

    # Generic object with .text, .words, etc.
    try:
        text = getattr(obj, "text", "") or getattr(obj, "result", "") or ""
        begin_time = getattr(obj, "begin_time", None)
        end_time = getattr(obj, "end_time", None)
        words = getattr(obj, "words", None) or []
        sentence_end = bool(getattr(obj, "sentence_end", False) or getattr(obj, "is_final", False))
        speaker = getattr(obj, "speaker", None)
        language = getattr(obj, "language", None)
        # If no text, treat as unusable
        if not text and not words:
            return None
        return {
            "text": text,
            "begin_time": begin_time,
            "end_time": end_time,
            "words": words,
            "sentence_end": sentence_end,
            "speaker": speaker,
            "language": language,
        }
    except Exception:
        return None


def _words_to_items(words: List[Dict[str, Any]], speaker: Optional[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for w in words or []:
        try:
            w_text = w.get("text") or w.get("content") or w.get("word")
            if not w_text:
                continue
            itm: Dict[str, Any] = {"Type": "pronunciation", "Content": w_text}
            if speaker:
                itm["Speaker"] = speaker
            # Optional timings if present
            bt = w.get("begin_time")
            et = w.get("end_time")
            if isinstance(bt, int):
                itm["StartTimeMs"] = bt
            if isinstance(et, int):
                itm["EndTimeMs"] = et
            out.append(itm)
        except Exception:
            continue
    return out


def _to_aws_shape(sentence: Dict[str, Any], seq: int, is_partial_override: Optional[bool] = None) -> Dict[str, Any]:
    """
    Build AWS-like payload:
      IsPartial: True for incremental (on_result_changed), False for final (on_sentence_end)
      We invert sentence['sentence_end'] unless override is provided.
    """
    text = (sentence.get("text") or "").strip()
    if not text:
        return {}

    speaker = sentence.get("speaker")
    if not speaker:
        # Try find first word with speaker field
        for w in sentence.get("words") or []:
            sp = w.get("speaker")
            if sp:
                speaker = sp
                break

    is_final = bool(sentence.get("sentence_end"))
    is_partial = is_partial_override if is_partial_override is not None else (not is_final)

    items = _words_to_items(sentence.get("words") or [], speaker)
    if speaker and not items:
        items.append({"Type": "pronunciation", "Speaker": speaker})

    return {
        "Transcript": {
            "Results": [
                {
                    "ResultId": f"alibaba-seg-{seq}",
                    "IsPartial": is_partial,
                    "Alternatives": [
                        {
                            "Transcript": text,
                            "Items": items,
                        }
                    ],
                    "LanguageCode": sentence.get("language"),
                }
            ]
        },
        "DisplayText": text,
        "TranslatedText": None,
        "ComprehendEntities": [],
    }


@router.websocket("/alibaba")
async def transcribe_alibaba(ws: WebSocket):
    """
    WebSocket proxy: Browser PCM (16kHz 16-bit) -> DashScope Paraformer -> AWS-shaped JSON.
    """
    await ws.accept()

    language_hints = ["en", "yue", "zh"]
    logger.info("Alibaba WS connected. language_hints=%s", language_hints)

    if not settings.dashscope_api_key:
        logger.error("Missing DASHSCOPE_API_KEY.")
        await ws.close(code=1011, reason="Missing API key.")
        return

    # Ensure env var for SDK
    os.environ.setdefault("DASHSCOPE_API_KEY", settings.dashscope_api_key)

    loop = asyncio.get_running_loop()
    outbound_queue: asyncio.Queue = asyncio.Queue()
    result_counter = 0
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

        def on_result_changed(self, result: Any):
            """
            Partial / incremental update.
            """
            nonlocal result_counter
            try:
                sentence = _normalize_sentence(result)
                if not sentence:
                    logger.debug("[DashScope] on_result_changed: no usable sentence object")
                    return
                # Do not increment counter for partial updates unless text changed meaningfully
                result_counter += 1
                payload = _to_aws_shape(sentence, result_counter, is_partial_override=True)
                if payload:
                    loop.call_soon_threadsafe(outbound_queue.put_nowait, payload)
            except Exception as e:
                logger.error("[DashScope] on_result_changed error: %s", e)

        def on_sentence_end(self, result: Any):
            """
            Finalized sentence.
            """
            nonlocal result_counter
            try:
                sentence = _normalize_sentence(result)
                if not sentence:
                    logger.debug("[DashScope] on_sentence_end: no usable sentence object")
                    return
                result_counter += 1
                payload = _to_aws_shape(sentence, result_counter, is_partial_override=False)
                if payload:
                    loop.call_soon_threadsafe(outbound_queue.put_nowait, payload)
            except Exception as e:
                logger.error("[DashScope] on_sentence_end error: %s", e)

        # Some SDK versions may also emit on_sentence_begin; safe to implement
        def on_sentence_begin(self, result: Any):
            logger.debug("[DashScope] on_sentence_begin received (ignored for UI).")

    recognizer: Optional[Recognition] = None

    async def forward_events():
        try:
            while running:
                msg = await outbound_queue.get()
                if isinstance(msg, dict) and msg.get("_close"):
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_text(json.dumps({"info": msg.get("_reason", "closed")}))
                        await ws.close(code=1011, reason=msg.get("_reason", "Service closed"))
                    break
                if ws.client_state != WebSocketState.CONNECTED:
                    break
                await ws.send_text(json.dumps(msg))
        except Exception as e:
            logger.error("forward_events error: %s", e)

    async def pump_audio():
        try:
            while running:
                chunk = await ws.receive_bytes()
                if not chunk:
                    logger.info("Received empty chunk; treating as end-of-stream trigger.")
                    break
                if recognizer:
                    recognizer.send_audio_frame(chunk)
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