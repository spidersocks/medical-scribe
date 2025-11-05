import asyncio
import base64
import json
import logging
import os
import queue
import re
import threading
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketDisconnect as StarletteWebSocketDisconnect
from starlette.websockets import WebSocketState

from google.api_core import exceptions as gcp_exceptions  # noqa: F401
from google.api_core.client_options import ClientOptions  # noqa: F401
from google.cloud import speech_v2 as speech
from google.cloud import translate_v3 as translate
import google.auth

from services import nlp

logger = logging.getLogger("stethoscribe-gcp-v2")

# Toggle high-volume debug logs by setting this env var.
GCP_STREAMING_DEBUG = os.getenv("GCP_STREAMING_DEBUG", "").lower() in {"1", "true", "yes"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _spk_label_from_any(word: object) -> Optional[str]:
    tag = getattr(word, "speaker_tag", None)
    lab = getattr(word, "speaker_label", None)
    if isinstance(tag, int) and tag > 0:
        return f"spk_{tag - 1}"
    if isinstance(lab, str) and lab:
        if lab.startswith("spk_"):
            return lab
        try:
            return f"spk_{max(0, int(lab) - 1)}"
        except Exception:
            digits = re.findall(r"\d+", lab)
            if digits:
                return f"spk_{max(0, int(digits[-1]) - 1)}"
    return None


def _dur_to_seconds(ts) -> float:
    if ts is None:
        return 0.0
    try:
        return float(ts.total_seconds())
    except Exception:
        pass
    return float(getattr(ts, "seconds", 0) or 0) + float(getattr(ts, "nanos", 0) or 0) / 1e9


_translate_client: Optional[translate.TranslationServiceClient] = None


def _get_translate_client() -> translate.TranslationServiceClient:
    global _translate_client
    if _translate_client is None:
        _translate_client = translate.TranslationServiceClient()
    return _translate_client


def _map_asr_lang_to_translate_source(asr_code: Optional[str]) -> str:
    if not asr_code:
        return "auto"
    code = asr_code.lower()
    if code.startswith("en"):
        return "en"
    if code.startswith("yue") or "hant" in code:
        return "zh-TW"
    if "hans" in code:
        return "zh-CN"
    if code.startswith("zh"):
        return "zh-TW"
    return "auto"


def _translate_to_english(text: str, project_id: Optional[str], source_lang: Optional[str]) -> str:
    if not text.strip():
        return text
    parent = f"projects/{project_id or '-'}/locations/global"
    request = {
        "parent": parent,
        "contents": [text],
        "mime_type": "text/plain",
        "source_language_code": source_lang or "auto",
        "target_language_code": "en",
    }
    try:
        response = _get_translate_client().translate_text(request=request)
        if response and response.translations:
            return response.translations[0].translated_text or text
    except Exception as exc:
        logger.warning("Translate v3 failed (src=%s). Error: %s", source_lang or "auto", exc)
    return text


def _resolve_gcp_project_id(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    env_proj = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    if env_proj:
        return env_proj
    try:
        _, proj = google.auth.default()
        return proj
    except Exception:
        return None


class _QueueBytesSource:
    def __init__(self) -> None:
        self._q: "queue.Queue[Optional[bytes]]" = queue.Queue()

    async def put(self, data: Optional[bytes]) -> None:
        # queue.Queue.put is non-blocking with infinite size; still call in executor to be safe.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._q.put, data)

    def audio_requests(self) -> Iterable[speech.StreamingRecognizeRequest]:
        while True:
            item = self._q.get()
            if item is None:
                break
            yield speech.StreamingRecognizeRequest(audio=item)


def _build_v2_recognition_config(language_codes: List[str], model: str) -> speech.RecognitionConfig:
    """
    Build a minimal RecognitionConfig compatible with the default recognizer (`recognizers/_`).
    Optional features (diarization, punctuation, word offsets) are managed by Google; requesting
    unsupported ones yields a 400.
    """
    try:
        auto_decoding = speech.AutoDetectDecodingConfig()
    except Exception:
        auto_decoding = {}

    return speech.RecognitionConfig(
        auto_decoding_config=auto_decoding,
        language_codes=language_codes,
        model=model,
    )


def _words_to_items(words) -> List[Dict]:
    return [
        {
            "StartTime": round(_dur_to_seconds(getattr(w, "start_offset", None)), 3),
            "EndTime": round(_dur_to_seconds(getattr(w, "end_offset", None)), 3),
            "Type": "pronunciation",
            "Content": getattr(w, "word", ""),
            "Speaker": _spk_label_from_any(w),
        }
        for w in list(words or [])
    ]


def _fallback_items(transcript: str) -> List[Dict]:
    if not transcript:
        return []
    return [
        {
            "StartTime": 0.0,
            "EndTime": 0.0,
            "Type": "pronunciation",
            "Content": transcript,
            "Speaker": None,
        }
    ]


def _normalize_to_aws_like_payload(result: speech.StreamingRecognitionResult, project_id: Optional[str]) -> Dict:
    is_final = bool(result.is_final)
    alt = result.alternatives[0] if result.alternatives else None
    transcript_text = getattr(alt, "transcript", "") if alt else ""
    detected_lang = getattr(alt, "language_code", None) or ("en-US" if not _has_cjk(transcript_text) else "yue-Hant-HK")

    items = _words_to_items(getattr(alt, "words", []))
    if not items and transcript_text:
        items = _fallback_items(transcript_text)

    payload: Dict[str, object] = {
        "Transcript": {
            "Results": [
                {
                    "Alternatives": [{"Transcript": transcript_text, "Items": items}],
                    "ResultId": str(uuid.uuid4()),
                    "IsPartial": not is_final,
                    "LanguageCode": detected_lang,
                }
            ]
        },
        "_engine": "gcp-v2",
        "_detected_language": detected_lang,
    }

    if is_final and transcript_text.strip():
        payload["DisplayText"] = transcript_text
        english_text = (
            transcript_text
            if detected_lang.lower().startswith("en")
            else _translate_to_english(
                transcript_text,
                project_id,
                _map_asr_lang_to_translate_source(detected_lang),
            )
        )
        if english_text and english_text != transcript_text:
            payload["TranslatedText"] = english_text
        try:
            payload["ComprehendEntities"] = nlp.detect_entities(english_text)
        except Exception:
            payload["ComprehendEntities"] = []
    elif is_final:
        payload["Transcript"]["Results"][0]["IsPartial"] = True

    return payload


def _normalize_lang_list_param(raw: str) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[,\s&]+", raw.split("?", 1)[0].strip())
    codes = ["yue-Hant-HK" if p.strip() == "zh-HK" else p.strip() for p in parts if p.strip()]
    seen = set()
    deduped: List[str] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            deduped.append(code)
    return deduped


def _normalize_v2_langs_and_location(codes: List[str], loc: str) -> Tuple[List[str], str]:
    normalized_codes: List[str] = []

    for code in codes:
        cl = code.lower()
        if cl in {"zh-tw", "cmn-hant-tw"} or cl.startswith("zh"):
            normalized_codes.append("cmn-Hant-TW")
        elif cl.startswith("yue") or cl == "zh-hk":
            normalized_codes.append("yue-Hant-HK")
        else:
            normalized_codes.append(code)

    final_loc = (loc or "global").strip().lower() or "global"
    if final_loc not in {
        "global",
        "us",
        "us-central1",
        "us-west1",
        "us-west2",
        "us-east1",
        "us-east4",
        "europe-west1",
        "europe-west2",
        "asia-southeast1",
    }:
        logger.warning("Unsupported location '%s' for Speech v2; defaulting to 'global'.", final_loc)
        final_loc = "global"

    seen = set()
    deduped: List[str] = []
    for code in normalized_codes:
        if code not in seen:
            seen.add(code)
            deduped.append(code)
    return deduped, final_loc


def _default_recognizer_name(project: str, location: str) -> str:
    return f"projects/{project}/locations/{location}/recognizers/_"


@dataclass
class _StreamStats:
    audio_messages: int = 0
    audio_bytes: int = 0
    last_log_at: int = 0
    first_audio_received: bool = False
    start_event_seen: bool = False
    start_metadata: Dict[str, object] = field(default_factory=dict)

    def bump(self, chunk_len: int) -> None:
        self.audio_messages += 1
        self.audio_bytes += chunk_len
        if not self.first_audio_received:
            self.first_audio_received = True
            logger.info("First audio frame received: %d bytes.", chunk_len)
        elif GCP_STREAMING_DEBUG and self.audio_messages <= 10:
            logger.debug("Audio frame #%d: %d bytes.", self.audio_messages, chunk_len)


def _parse_ws_text_message(text: str) -> Tuple[str, Optional[bytes], Dict[str, object]]:
    """
    Returns (event_name, audio_bytes, raw_json).
    audio_bytes is None when no audio is present; b"" signals a stop request.
    """
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Dropped non-JSON text frame: %s", text[:160])
        return "invalid", None, {}

    event = (msg.get("event") or "").lower()
    if not event:
        logger.debug("Ignoring JSON frame without 'event': %s", msg)
        return "unknown", None, msg

    if event == "start":
        return event, None, msg

    if event in {"stop", "speech_stopped"}:
        return event, b"", msg

    if event == "media":
        media = msg.get("media")
        if not isinstance(media, dict):
            logger.warning("Media frame missing 'media' object: %s", msg)
            return "media", None, msg
        payload = media.get("payload")
        if not isinstance(payload, str) or not payload:
            logger.warning("Media frame missing 'payload': %s", msg)
            return "media", None, msg
        padding = "=" * (-len(payload) % 4)
        try:
            audio_bytes = base64.b64decode(payload + padding, validate=False)
            return event, audio_bytes, msg
        except Exception:
            logger.warning("Failed to base64-decode media payload (length %d).", len(payload))
            return "media", None, msg

    return event, None, msg


async def _run_watchdog(stats: _StreamStats, ws: WebSocket) -> None:
    """
    Emit warnings if no audio arrives within 5s, or if the stream stalls.
    """
    idle_checks = 0
    prev_audio_count = 0
    try:
        while True:
            await asyncio.sleep(5)
            if ws.client_state != WebSocketState.CONNECTED:
                break
            if stats.audio_messages == 0:
                idle_checks += 1
                if idle_checks >= 2:
                    logger.warning(
                        "No audio frames received after %d seconds. "
                        "Verify the browser is sending audio.",
                        idle_checks * 5,
                    )
            else:
                if stats.audio_messages == prev_audio_count:
                    logger.warning(
                        "No new audio frames in the last 5 seconds "
                        "(total frames: %d, bytes: %d).",
                        stats.audio_messages,
                        stats.audio_bytes,
                    )
                prev_audio_count = stats.audio_messages
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def register_gcp_streaming_v2_routes(
    app: FastAPI,
    *,
    gcp_project_id: Optional[str] = None,
    gcp_location: str = "global",
) -> None:
    @app.websocket("/client-transcribe-gcp-v2")
    async def client_transcribe_gcp_v2(ws: WebSocket) -> None:
        await ws.accept()

        raw_langs = (ws.query_params.get("languages") or "").strip()
        req_loc = (ws.query_params.get("location") or gcp_location or "global").strip()
        req_codes = _normalize_lang_list_param(raw_langs) or ["en-US", "yue-Hant-HK", "zh-TW"]

        language_codes, location = _normalize_v2_langs_and_location(req_codes, req_loc)
        project = _resolve_gcp_project_id(gcp_project_id)

        if not project:
            logger.error("GCP v2: Could not resolve project id.")
            await ws.close(code=1011, reason="Missing GCP project id.")
            return

        logger.info(
            "GCP v2 WS connected. languages=%s | location=%s | project=%s",
            language_codes,
            location,
            project,
        )

        speech_client = speech.SpeechClient()
        recognition_config = _build_v2_recognition_config(language_codes, model="latest_long")
        streaming_config = speech.StreamingRecognitionConfig(config=recognition_config)
        recognizer_name = _default_recognizer_name(project, location)

        loop = asyncio.get_running_loop()

        bytes_src = _QueueBytesSource()
        out_q: "asyncio.Queue[Optional[object]]" = asyncio.Queue()
        stats = _StreamStats()

        def _request_generator():
            yield speech.StreamingRecognizeRequest(
                recognizer=recognizer_name,
                streaming_config=streaming_config,
            )
            yield from bytes_src.audio_requests()

        def _gcp_streaming_call():
            try:
                responses = speech_client.streaming_recognize(requests=_request_generator())
                for resp in responses:
                    asyncio.run_coroutine_threadsafe(out_q.put(resp), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(out_q.put(exc), loop)
            finally:
                asyncio.run_coroutine_threadsafe(out_q.put(None), loop)

        async def _writer_to_client():
            while True:
                resp = await out_q.get()
                if resp is None:
                    break
                if isinstance(resp, Exception):
                    logger.error("GCP v2 streaming_recognize error: %s", resp)
                    if "404" in str(resp) and location != "global":
                        logger.error(
                            "Streaming recognizer failed for location '%s'. "
                            "Reconnect without specifying a region to use the global endpoint.",
                            location,
                        )
                    break
                try:
                    payload = _normalize_to_aws_like_payload(resp, project)
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_text(json.dumps(payload))
                except Exception as exc:
                    logger.error("Error processing GCP response: %s", exc)

        writer_task = asyncio.create_task(_writer_to_client(), name="gcp-v2-writer")
        audio_watchdog = asyncio.create_task(_run_watchdog(stats, ws), name="gcp-v2-audio-watchdog")
        gcp_future = loop.run_in_executor(None, _gcp_streaming_call)

        try:
            while ws.client_state == WebSocketState.CONNECTED:
                try:
                    message = await ws.receive()
                except (WebSocketDisconnect, StarletteWebSocketDisconnect):
                    logger.info("Client disconnected from /client-transcribe-gcp-v2.")
                    break
                except RuntimeError as exc:
                    logger.error("WebSocket receive runtime error: %s", exc)
                    break

                msg_type = message.get("type")
                if msg_type == "websocket.disconnect":
                    logger.info("Received WebSocket disconnect frame.")
                    break

                data = message.get("bytes")
                text = message.get("text")

                if data:
                    stats.bump(len(data))
                    await bytes_src.put(data)
                    if GCP_STREAMING_DEBUG:
                        logger.debug("Binary frame received (%d bytes).", len(data))
                    continue

                if text:
                    event, audio_bytes, raw_msg = _parse_ws_text_message(text)

                    if event == "start":
                        stats.start_event_seen = True
                        stats.start_metadata = raw_msg.get("start", {})
                        logger.info("Received start event: %s", stats.start_metadata)
                        continue

                    if event == "stop":
                        logger.info("Received stop event; closing audio stream.")
                        break

                    if event == "media":
                        if audio_bytes:
                            stats.bump(len(audio_bytes))
                            await bytes_src.put(audio_bytes)
                        else:
                            logger.warning("Media event without audio bytes. Raw: %s", raw_msg)
                        continue

                    if event in {"connected", "keepalive", "ping"}:
                        if event == "keepalive":
                            logger.debug("Keepalive event received.")
                        continue

                    if event == "invalid":
                        continue

                    logger.debug("Ignoring unsupported event '%s': %s", event, raw_msg)
                    continue

                logger.debug("Ignored frame without text/bytes: %s", message)

        except Exception as exc:
            logger.error("Unexpected error receiving audio from browser: %s", exc)
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1011, reason="Audio receive error.")
        finally:
            await bytes_src.put(None)

            writer_task.cancel()
            audio_watchdog.cancel()
            with suppress(asyncio.CancelledError):
                await writer_task
            with suppress(asyncio.CancelledError):
                await audio_watchdog

            try:
                await asyncio.wrap_future(gcp_future)
            except Exception as exc:
                logger.error("GCP streaming future raised: %s", exc)

            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close()
            logger.info(
                "GCP v2 WebSocket session ended. Frames=%d, bytes=%d, start_event=%s",
                stats.audio_messages,
                stats.audio_bytes,
                stats.start_event_seen,
            )