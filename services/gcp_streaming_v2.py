import asyncio
import json
import logging
import re
import uuid
import queue
import os
import hashlib
from typing import Dict, List, Optional, Iterable, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from google.cloud import speech_v2 as speech
from google.cloud import translate_v3 as translate
from google.api_core import exceptions as gax
import google.auth

from services import nlp

logger = logging.getLogger("stethoscribe-gcp-v2")

# --- Helper Functions ---
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))

def _spk_label_from_any(word: object) -> Optional[str]:
    tag = getattr(word, "speaker_tag", None)
    lab = getattr(word, "speaker_label", None)
    if isinstance(tag, int) and tag > 0: return f"spk_{tag-1}"
    if isinstance(lab, str) and lab:
        if lab.startswith("spk_"): return lab
        try: return f"spk_{max(0, int(lab)-1)}"
        except Exception:
            digits = re.findall(r"\d+", lab)
            if digits: return f"spk_{max(0, int(digits[-1]) - 1)}"
    return None

def _dur_to_seconds(ts) -> float:
    if ts is None: return 0.0
    try: return float(ts.total_seconds())
    except Exception: pass
    return float(getattr(ts, "seconds", 0) or 0) + float(getattr(ts, "nanos", 0) or 0) / 1e9

_translate_client: Optional[translate.TranslationServiceClient] = None
def _get_translate_client() -> translate.TranslationServiceClient:
    global _translate_client
    if _translate_client is None: _translate_client = translate.TranslationServiceClient()
    return _translate_client

def _map_asr_lang_to_translate_source(asr_code: Optional[str]) -> str:
    if not asr_code: return "auto"
    code = asr_code.lower()
    if code.startswith("en"): return "en"
    if code.startswith("yue") or "hant" in code: return "zh-TW"
    if "hans" in code: return "zh-CN"
    if code.startswith("zh"): return "zh-TW"
    return "auto"

def _translate_to_english(text: str, project_id: Optional[str], source_lang: Optional[str]) -> str:
    if not text.strip(): return text
    parent = f"projects/{project_id or '-'}/locations/global"
    req = {"parent": parent, "contents": [text], "mime_type": "text/plain", "source_language_code": source_lang or "auto", "target_language_code": "en"}
    try:
        resp = _get_translate_client().translate_text(request=req)
        if resp and resp.translations: return resp.translations[0].translated_text or text
    except Exception as exc:
        logger.warning("Translate v3 failed (src=%s). Error: %s", source_lang or "auto", exc)
    return text

def _resolve_gcp_project_id(explicit: Optional[str]) -> Optional[str]:
    if explicit: return explicit
    env_proj = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    if env_proj: return env_proj
    try: _, proj = google.auth.default(); return proj
    except Exception: return None

class _QueueBytesSource:
    def __init__(self): self._q: "queue.Queue[Optional[bytes]]" = queue.Queue()
    async def put(self, data: Optional[bytes]): self._q.put(data)
    def audio_requests(self) -> Iterable[speech.StreamingRecognizeRequest]:
        while True:
            item = self._q.get()
            if item is None: break
            yield speech.StreamingRecognizeRequest(audio=item)

def _build_v2_recognizer_config(language_codes: List[str], model: str) -> speech.RecognitionConfig:
    features_kwargs = {"enable_automatic_punctuation": True, "enable_word_time_offsets": True}
    try:
        features = speech.RecognitionFeatures(**features_kwargs, diarization_config=speech.SpeakerDiarizationConfig(min_speaker_count=2, max_speaker_count=2))
    except Exception:
        logger.warning("GCP v2: diarization_config unsupported by client library; continuing without diarization.")
        features = speech.RecognitionFeatures(**features_kwargs)
    return speech.RecognitionConfig(auto_decoding_config={}, language_codes=language_codes, model=model, features=features)

def _safe_recognizer_id(language_codes: List[str], model: str) -> str:
    key = "|".join(sorted(language_codes)) + "|" + model
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    model_short = model.replace("_", "").replace("-", "")
    return f"stethoscribe-{h}-{model_short}"[:63]

def _ensure_v2_recognizer(client: speech.SpeechClient, project: str, location: str, language_codes: List[str], model: str) -> str:
    parent = f"projects/{project}/locations/{location}"
    recognizer_id = _safe_recognizer_id(language_codes, model)
    name = f"{parent}/recognizers/{recognizer_id}"
    try:
        client.get_recognizer(request={"name": name})
        logger.info("GCP v2: Found existing recognizer: %s", name)
        return name
    except gax.NotFound:
        logger.info("GCP v2: Recognizer not found. Creating: %s", name)
        cfg = _build_v2_recognizer_config(language_codes, model)
        op = client.create_recognizer(request={"parent": parent, "recognizer_id": recognizer_id, "recognizer": speech.Recognizer(default_recognition_config=cfg)})
        op.result(timeout=60)
        logger.info("GCP v2: Recognizer created successfully: %s", name)
        return name

def _words_to_items(words) -> List[Dict]:
    return [{"StartTime": round(_dur_to_seconds(getattr(w, "start_offset", None)), 3), "EndTime": round(_dur_to_seconds(getattr(w, "end_offset", None)), 3), "Type": "pronunciation", "Content": getattr(w, "word", ""), "Speaker": _spk_label_from_any(w)} for w in list(words or [])]

def _normalize_to_aws_like_payload(result: speech.StreamingRecognitionResult, project_id: Optional[str]) -> Dict:
    is_final, alt = bool(result.is_final), result.alternatives[0] if result.alternatives else None
    transcript_text = getattr(alt, "transcript", "") if alt else ""
    detected_lang = getattr(alt, "language_code", None) or ("en-US" if not _has_cjk(transcript_text) else "yue-Hant-HK")
    payload = {"Transcript": {"Results": [{"Alternatives": [{"Transcript": transcript_text, "Items": _words_to_items(getattr(alt, "words", []))}], "ResultId": str(uuid.uuid4()), "IsPartial": not is_final, "LanguageCode": detected_lang}]}, "_engine": "gcp-v2", "_detected_language": detected_lang}
    if is_final and transcript_text.strip():
        payload["DisplayText"] = transcript_text
        english_text = transcript_text if detected_lang.lower().startswith("en") else _translate_to_english(transcript_text, project_id, _map_asr_lang_to_translate_source(detected_lang))
        if english_text and english_text != transcript_text: payload["TranslatedText"] = english_text
        try: payload["ComprehendEntities"] = nlp.detect_entities(english_text)
        except Exception: payload["ComprehendEntities"] = []
    elif is_final:
        payload["Transcript"]["Results"][0]["IsPartial"] = True
    return payload

def _normalize_lang_list_param(raw: str) -> List[str]:
    if not raw: return []
    parts = re.split(r"[,\s&]+", raw.split("?", 1)[0].strip())
    codes = ["yue-Hant-HK" if p.strip() == "zh-HK" else p.strip() for p in parts if p.strip()]
    seen = set(); return [c for c in codes if not (c in seen or seen.add(c))]

def _normalize_v2_langs_and_location(codes: List[str], loc: str) -> Tuple[List[str], str]:
    normalized_codes, has_cjk = [], False
    for c in codes:
        cl = c.lower()
        if cl in ("zh-tw", "cmn-hant-tw") or cl.startswith("zh"): normalized_codes.append("cmn-Hant-TW"); has_cjk = True
        elif cl.startswith("yue") or cl == "zh-hk": normalized_codes.append("yue-Hant-HK"); has_cjk = True
        else: normalized_codes.append(c)
    final_loc = (loc or "global").strip().lower() or "global"
    if has_cjk and final_loc == "global": final_loc = "asia-east1"
    seen = set(); return [c for c in normalized_codes if not (c in seen or seen.add(c))], final_loc

# --- Main Route Registration ---
def register_gcp_streaming_v2_routes(app: FastAPI, *, gcp_project_id: Optional[str] = None, gcp_location: str = "global") -> None:
    @app.websocket("/client-transcribe-gcp-v2")
    async def client_transcribe_gcp_v2(ws: WebSocket):
        await ws.accept()
        raw_langs = (ws.query_params.get("languages") or "").strip()
        req_loc = (ws.query_params.get("location") or gcp_location or "global").strip()
        req_codes = _normalize_lang_list_param(raw_langs) or ["en-US", "yue-Hant-HK", "zh-TW"]

        language_codes, location = _normalize_v2_langs_and_location(req_codes, req_loc)
        project = _resolve_gcp_project_id(gcp_project_id)

        if not project:
            logger.error("GCP v2: Could not resolve project id."); await ws.close(); return

        logger.info("GCP v2 WS connecting. languages=%s | location=%s | project=%s", language_codes, location, project)
        
        # Always use the global client. The recognizer's name will handle routing.
        speech_client = speech.SpeechClient()
        
        try:
            model = "latest_long"
            recognizer_name = _ensure_v2_recognizer(speech_client, project, location, language_codes, model)
        except Exception as exc:
            logger.error("Failed to prepare recognizer in %s: %s", location, exc)
            await ws.close(code=1011, reason="Recognizer preparation failed.")
            return

        bytes_src = _QueueBytesSource()
        out_q: "asyncio.Queue[Optional[object]]" = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _request_generator():
            # First request contains the recognizer. No streaming_config is needed here.
            # The config is part of the recognizer resource itself.
            yield speech.StreamingRecognizeRequest(recognizer=recognizer_name)
            yield from bytes_src.audio_requests()

        def _gcp_streaming_call():
            try:
                responses = speech_client.streaming_recognize(requests=_request_generator())
                for resp in responses: asyncio.run_coroutine_threadsafe(out_q.put(resp), loop)
            except Exception as exc: asyncio.run_coroutine_threadsafe(out_q.put(exc), loop)
            finally: asyncio.run_coroutine_threadsafe(out_q.put(None), loop)

        async def _writer_to_client():
            while True:
                resp = await out_q.get()
                if resp is None: break
                if isinstance(resp, Exception): logger.error("GCP v2 streaming error: %s", resp); break
                try:
                    payload = _normalize_to_aws_like_payload(resp, project)
                    if ws.client_state == WebSocketState.CONNECTED: await ws.send_text(json.dumps(payload))
                except Exception as e: logger.error("Error processing message: %s", e)
        
        writer_task = asyncio.create_task(_writer_to_client())
        loop.run_in_executor(None, _gcp_streaming_call)
        
        try:
            while ws.client_state == WebSocketState.CONNECTED:
                data = await ws.receive_bytes()
                await bytes_src.put(data)
        except WebSocketDisconnect:
            logger.info("Browser disconnected.")
        finally:
            await bytes_src.put(None)
            writer_task.cancel()
            try: await writer_task
            except asyncio.CancelledError: pass
            if ws.client_state == WebSocketState.CONNECTED: await ws.close()
            logger.info("GCP v2 WS session ended.")