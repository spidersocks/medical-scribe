import asyncio
import json
import logging
import re
import uuid
import queue
import os
from typing import Dict, List, Optional, Iterable, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from google.cloud import speech_v2 as speech
from google.cloud import translate_v3 as translate
from google.api_core.client_options import ClientOptions
import google.auth

from services import nlp

logger = logging.getLogger("stethoscribe-gcp-v2")

# --- Helper Functions ---
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

def _has_cjk(text: str) -> bool:
    """Check if a string contains Chinese/Japanese/Korean characters."""
    return bool(_CJK_RE.search(text or ""))

def _spk_label_from_any(word: object) -> Optional[str]:
    """Safely extract speaker label from a v2 word object."""
    tag = getattr(word, "speaker_tag", None)
    lab = getattr(word, "speaker_label", None)
    if isinstance(tag, int) and tag > 0:
        return f"spk_{tag - 1}"
    if isinstance(lab, str) and lab:
        if lab.startswith("spk_"):
            return lab
        try:
            return f"spk_{max(0, int(lab) - 1)}"
        except (ValueError, TypeError):
            digits = re.findall(r"\d+", lab)
            if digits:
                return f"spk_{max(0, int(digits[-1]) - 1)}"
    return None

def _dur_to_seconds(ts) -> float:
    """Convert a duration object to seconds."""
    if ts is None:
        return 0.0
    try:
        return float(ts.total_seconds())
    except (AttributeError, TypeError):
        return float(getattr(ts, "seconds", 0) or 0) + float(getattr(ts, "nanos", 0) or 0) / 1e9

_translate_client: Optional[translate.TranslationServiceClient] = None
def _get_translate_client() -> translate.TranslationServiceClient:
    global _translate_client
    if _translate_client is None:
        _translate_client = translate.TranslationServiceClient()
    return _translate_client

def _map_asr_lang_to_translate_source(asr_code: Optional[str]) -> str:
    """Map ASR language code to a compatible Translation API code."""
    if not asr_code:
        return "auto"
    code = asr_code.lower()
    if code.startswith("en"):
        return "en"
    if code.startswith("yue") or "hant" in code:
        return "zh-TW"  # Use Traditional Chinese for translation
    if "hans" in code:
        return "zh-CN"
    if code.startswith("zh"):
        return "zh-TW"
    return "auto"

def _translate_to_english(text: str, project_id: Optional[str], source_lang: Optional[str]) -> str:
    """Translate text to English using Google Translate API."""
    if not text.strip():
        return text
    parent = f"projects/{project_id or '-'}/locations/global"
    request = {"parent": parent, "contents": [text], "mime_type": "text/plain", "source_language_code": source_lang or "auto", "target_language_code": "en"}
    try:
        response = _get_translate_client().translate_text(request=request)
        return response.translations[0].translated_text if response and response.translations else text
    except Exception as exc:
        logger.warning("Translate v3 failed (src=%s). Error: %s", source_lang or "auto", exc)
    return text

def _resolve_gcp_project_id(explicit: Optional[str]) -> Optional[str]:
    """Resolve GCP project ID from explicit arg, environment, or auth context."""
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
    """A thread-safe, async-compatible byte queue to bridge web server and gRPC threads."""
    def __init__(self):
        self._q: "queue.Queue[Optional[bytes]]" = queue.Queue()

    async def put(self, data: Optional[bytes]):
        """Put data into the queue from an async context."""
        await asyncio.get_running_loop().run_in_executor(None, self._q.put, data)

    def audio_requests(self) -> Iterable[speech.StreamingRecognizeRequest]:
        """Generator that yields audio requests for the gRPC thread."""
        while True:
            item = self._q.get()
            if item is None:
                break
            yield speech.StreamingRecognizeRequest(audio=item)

def _build_v2_recognition_config(language_codes: List[str], model: str) -> speech.RecognitionConfig:
    """Build the RecognitionConfig for v2, enabling all necessary features."""
    features_kwargs = {"enable_automatic_punctuation": True, "enable_word_time_offsets": True}
    try:
        features = speech.RecognitionFeatures(**features_kwargs, diarization_config=speech.SpeakerDiarizationConfig(min_speaker_count=2, max_speaker_count=2))
    except TypeError:
        logger.warning("GCP v2: diarization_config not supported by client library version. Continuing without speaker labels.")
        features = speech.RecognitionFeatures(**features_kwargs)
    
    return speech.RecognitionConfig(
        auto_decoding_config=speech.AutoDetectDecodingConfig(),
        language_codes=language_codes,
        model=model,
        features=features
    )

def _words_to_items(words: List) -> List[Dict]:
    """Convert GCP word objects to the AWS-like 'Items' format for the frontend."""
    return [{"StartTime": round(_dur_to_seconds(getattr(w, "start_offset", None)), 3), "EndTime": round(_dur_to_seconds(getattr(w, "end_offset", None)), 3), "Type": "pronunciation", "Content": getattr(w, "word", ""), "Speaker": _spk_label_from_any(w)} for w in words]

def _normalize_to_aws_like_payload(result: speech.StreamingRecognitionResult, project_id: Optional[str]) -> Dict:
    """Convert a GCP v2 result to the AWS-like JSON structure the frontend expects."""
    is_final, alt = bool(result.is_final), result.alternatives[0] if result.alternatives else None
    transcript_text = getattr(alt, "transcript", "") if alt else ""
    detected_lang = getattr(alt, "language_code", None) or ("en-US" if not _has_cjk(transcript_text) else "yue-Hant-HK")
    
    payload: Dict[str, Any] = {
        "Transcript": {"Results": [{"Alternatives": [{"Transcript": transcript_text, "Items": _words_to_items(getattr(alt, "words", []))}], "ResultId": str(uuid.uuid4()), "IsPartial": not is_final, "LanguageCode": detected_lang}]},
        "_engine": "gcp-v2",
        "_detected_language": detected_lang,
    }
    
    if is_final and transcript_text.strip():
        payload["DisplayText"] = transcript_text
        english_text = transcript_text if detected_lang.lower().startswith("en") else _translate_to_english(transcript_text, project_id, _map_asr_lang_to_translate_source(detected_lang))
        if english_text and english_text != transcript_text:
            payload["TranslatedText"] = english_text
        try:
            payload["ComprehendEntities"] = nlp.detect_entities(english_text)
        except Exception:
            payload["ComprehendEntities"] = []
    elif is_final: # Mark empty finals as partial to prevent UI flicker
        payload["Transcript"]["Results"][0]["IsPartial"] = True
        
    return payload

def _normalize_lang_list_param(raw: str) -> List[str]:
    """Parse comma-separated language codes from a URL query parameter."""
    if not raw:
        return []
    parts = re.split(r"[,\s&]+", raw.split("?", 1)[0].strip())
    codes = ["yue-Hant-HK" if p.strip() == "zh-HK" else p.strip() for p in parts if p.strip()]
    seen = set()
    return [c for c in codes if c not in seen and not seen.add(c)]

def _normalize_v2_langs_and_location(codes: List[str], loc: str) -> Tuple[List[str], str]:
    """Normalize language codes for v2 and select a compatible region if needed."""
    normalized_codes, has_cjk = [], False
    for c in codes:
        cl = c.lower()
        if cl in ("zh-tw", "cmn-hant-tw") or cl.startswith("zh"):
            normalized_codes.append("cmn-Hant-TW")
            has_cjk = True
        elif cl.startswith("yue") or cl == "zh-hk":
            normalized_codes.append("yue-Hant-HK")
            has_cjk = True
        else:
            normalized_codes.append(c)
    
    final_loc = (loc or "global").strip().lower() or "global"
    if has_cjk and final_loc == "global":
        final_loc = "asia-east1" # Default to a region that supports these languages
    
    seen = set()
    return [c for c in normalized_codes if c not in seen and not seen.add(c)], final_loc

# --- Main Route Registration ---
def register_gcp_streaming_v2_routes(app: FastAPI, *, gcp_project_id: Optional[str] = None, gcp_location: str = "global") -> None:
    @app.websocket("/client-transcribe-gcp-v2")
    async def client_transcribe_gcp_v2(ws: WebSocket):
        await ws.accept()
        
        # --- Configuration ---
        raw_langs = (ws.query_params.get("languages") or "").strip()
        req_loc = (ws.query_params.get("location") or gcp_location or "global").strip()
        req_codes = _normalize_lang_list_param(raw_langs) or ["en-US", "yue-Hant-HK", "zh-TW"]

        language_codes, location = _normalize_v2_langs_and_location(req_codes, req_loc)
        project = _resolve_gcp_project_id(gcp_project_id)

        if not project:
            logger.error("GCP v2: Could not resolve project id."); await ws.close(); return

        logger.info("GCP v2 WS connected. languages=%s | location=%s | project=%s", language_codes, location, project)
        
        # KEY FIX: Use a regional client if the location is not global.
        client_options = ClientOptions(api_endpoint=f"{location}-speech.googleapis.com") if location != "global" else None
        speech_client = speech.SpeechClient(client_options=client_options)
        
        model = "latest_long"
        recognition_config = _build_v2_recognition_config(language_codes, model=model)
        streaming_config = speech.StreamingRecognitionConfig(config=recognition_config)

        # --- Streaming Orchestration ---
        bytes_src = _QueueBytesSource()
        out_q: "asyncio.Queue[Optional[object]]" = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _request_generator():
            # The first request must contain the configuration. Subsequent requests contain audio.
            yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
            yield from bytes_src.audio_requests()

        def _gcp_streaming_call():
            """Runs in a separate thread to bridge async and gRPC."""
            try:
                responses = speech_client.streaming_recognize(requests=_request_generator())
                for resp in responses:
                    asyncio.run_coroutine_threadsafe(out_q.put(resp), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(out_q.put(exc), loop)
            finally:
                asyncio.run_coroutine_threadsafe(out_q.put(None), loop)

        async def _writer_to_client():
            """Sends processed results back to the WebSocket client."""
            while True:
                resp = await out_q.get()
                if resp is None: break
                if isinstance(resp, Exception):
                    logger.error("GCP v2 streaming error: %s", resp); break
                try:
                    payload = _normalize_to_aws_like_payload(resp, project)
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_text(json.dumps(payload))
                except Exception as e:
                    logger.error("Error processing message: %s", e)
        
        writer_task = asyncio.create_task(_writer_to_client())
        loop.run_in_executor(None, _gcp_streaming_call)
        
        # --- Main loop: Receive audio from client ---
        try:
            while ws.client_state == WebSocketState.CONNECTED:
                # This implementation expects raw binary audio, which matches your frontend.
                data = await ws.receive_bytes()
                await bytes_src.put(data)
        except WebSocketDisconnect:
            logger.info("Browser disconnected.")
        finally:
            await bytes_src.put(None) # Signal end of audio to the gRPC stream
            writer_task.cancel()
            try:
                await writer_task
            except asyncio.CancelledError:
                pass # Expected cancellation
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close()
            logger.info("GCP v2 WS session ended.")