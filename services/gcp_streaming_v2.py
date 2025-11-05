import asyncio
import json
import logging
import re
import uuid
import queue
import os
import hashlib
from typing import Dict, List, Optional, Iterable, Tuple

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketState

from google.cloud import speech_v2 as speech
from google.cloud import translate_v3 as translate
from google.api_core.client_options import ClientOptions
import google.auth

from services import nlp

logger = logging.getLogger("stethoscribe-gcp-v2")

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))

def _spk_label_from_any(word: object) -> Optional[str]:
    tag = getattr(word, "speaker_tag", None)
    lab = getattr(word, "speaker_label", None)
    if isinstance(tag, int) and tag > 0:
        return f"spk_{tag-1}"
    if isinstance(lab, str) and lab:
        if lab.startswith("spk_"):
            return lab
        try:
            n = int(lab)
            return f"spk_{max(0, n-1)}"
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
    secs = float(getattr(ts, "seconds", 0) or 0)
    nanos = float(getattr(ts, "nanos", 0) or 0)
    return secs + nanos / 1e9

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
    if code.startswith("yue"):
        return "zh-TW"
    if "hant" in code:
        return "zh-TW"
    if "hans" in code:
        return "zh-CN"
    if code.startswith("zh"):
        return "zh-TW"
    return "auto"

def _translate_to_english(
    text: str, project_id: Optional[str] = None, location: str = "global", source_lang: Optional[str] = None
) -> str:
    if not text.strip():
        return text
    parent = f"projects/{project_id or '-'}/locations/{location}"
    req = {
        "parent": parent, "contents": [text], "mime_type": "text/plain",
        "source_language_code": source_lang or "auto", "target_language_code": "en",
    }
    try:
        resp = _get_translate_client().translate_text(request=req)
        if resp and resp.translations:
            return resp.translations[0].translated_text or text
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
    def __init__(self):
        self._q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._closed = False

    async def put(self, data: Optional[bytes]):
        if self._closed: return
        try: self._q.put_nowait(data)
        except queue.Full: self._q.put(data)

    def close(self): self._closed = True

    def audio_requests(self) -> Iterable[speech.StreamingRecognizeRequest]:
        while True:
            item = self._q.get()
            if item is None: break
            yield speech.StreamingRecognizeRequest(audio=item)

def _build_v2_recognition_config(language_codes: List[str], model: str = "latest_long") -> speech.RecognitionConfig:
    """Builds the main RecognitionConfig for v2."""
    base_features_kwargs = dict(enable_automatic_punctuation=True, enable_word_time_offsets=True)
    try:
        features = speech.RecognitionFeatures(
            **base_features_kwargs,
            diarization_config=speech.SpeakerDiarizationConfig(min_speaker_count=2, max_speaker_count=2),
        )
    except Exception:
        logger.warning("GCP v2: diarization_config unsupported; continuing without diarization.")
        features = speech.RecognitionFeatures(**base_features_kwargs)

    return speech.RecognitionConfig(
        auto_decoding_config=speech.AutoDetectDecodingConfig(),
        language_codes=language_codes,
        model=model,
        features=features,
    )

def _words_to_items(words) -> List[Dict]:
    items = []
    for w in list(words or []):
        start = _dur_to_seconds(getattr(w, "start_offset", None))
        end = _dur_to_seconds(getattr(w, "end_offset", None))
        items.append({
            "StartTime": round(start, 3), "EndTime": round(end, 3),
            "Type": "pronunciation", "Content": getattr(w, "word", ""),
            "Speaker": _spk_label_from_any(w),
        })
    return items

def _normalize_to_aws_like_payload(result: speech.StreamingRecognitionResult, project_id: Optional[str]) -> Dict:
    is_final = bool(result.is_final)
    alt = result.alternatives[0] if result.alternatives else None
    transcript_text = getattr(alt, "transcript", "") if alt else ""
    words = list(getattr(alt, "words", []) or [])
    items = _words_to_items(words)

    detected_lang = getattr(alt, "language_code", None) or ("en-US" if not _has_cjk(transcript_text) else "yue-Hant-HK")

    payload = {
        "Transcript": { "Results": [{
            "Alternatives": [{"Transcript": transcript_text, "Items": items}],
            "ResultId": str(uuid.uuid4()), "IsPartial": not is_final, "LanguageCode": detected_lang,
        }]},
        "_engine": "gcp-v2", "_detected_language": detected_lang,
    }

    if is_final and not transcript_text.strip():
        payload["Transcript"]["Results"][0]["IsPartial"] = True
        return payload

    if is_final and transcript_text.strip():
        payload["DisplayText"] = transcript_text
        if not detected_lang.lower().startswith("en"):
            english_text = _translate_to_english(
                transcript_text, project_id, source_lang=_map_asr_lang_to_translate_source(detected_lang)
            )
            if english_text and english_text != transcript_text:
                payload["TranslatedText"] = english_text
        else:
            english_text = transcript_text

        try: ents = nlp.detect_entities(english_text)
        except Exception: ents = []
        payload["ComprehendEntities"] = ents

    return payload

def _normalize_lang_list_param(raw: str) -> List[str]:
    if not raw: return []
    raw = raw.split("?", 1)[0]
    parts = re.split(r"[,\s&]+", raw.strip())
    out = []
    for p in parts:
        p_clean = p.strip()
        if not p_clean: continue
        out.append("yue-Hant-HK" if p_clean == "zh-HK" else p_clean)
    seen = set()
    return [c for c in out if not (c in seen or seen.add(c))]

def _normalize_v2_langs_and_location(codes: List[str], requested_location: str) -> Tuple[List[str], str]:
    normalized_codes, has_cjk = [], False
    for c in codes:
        cl = c.lower()
        if cl in ("zh-tw", "cmn-hant-tw"):
            normalized_codes.append("cmn-Hant-TW"); has_cjk = True
        elif cl.startswith("yue") or cl == "zh-hk":
            normalized_codes.append("yue-Hant-HK"); has_cjk = True
        elif cl.startswith("zh"):
            normalized_codes.append("cmn-Hant-TW"); has_cjk = True
        else:
            normalized_codes.append(c)

    loc = (requested_location or "global").strip().lower() or "global"
    if has_cjk and loc == "global":
        loc = "asia-east1"
    
    final_codes, seen = [], set()
    return [c for c in normalized_codes if not (c in seen or seen.add(c))], loc


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

        logger.info("GCP v2 WS connected. languages=%s | location=%s | project=%s", language_codes, location, project)
        
        client_options = ClientOptions(api_endpoint=f"{location}-speech.googleapis.com") if location != "global" else None
        speech_client = speech.SpeechClient(client_options=client_options)

        model = "latest_long"
        recognition_config = _build_v2_recognition_config(language_codes, model=model)
        
        # The first request must contain the config. Subsequent requests contain audio.
        streaming_config = speech.StreamingRecognitionConfig(config=recognition_config)

        bytes_src = _QueueBytesSource()
        out_q: "asyncio.Queue[Optional[speech.StreamingRecognizeResponse]]" = asyncio.Queue()
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        async def _reader_from_client():
            try:
                while True:
                    data = await ws.receive_bytes()
                    if not data: await bytes_src.put(None); break
                    await bytes_src.put(data)
            except Exception as exc:
                logger.info("Browser disconnected: %s", exc)
                try: await bytes_src.put(None)
                except Exception: pass
            finally: stop_event.set()

        def _request_generator():
            yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
            yield from bytes_src.audio_requests()

        def _gcp_streaming_call():
            try:
                responses = speech_client.streaming_recognize(requests=_request_generator())
                for resp in responses: asyncio.run_coroutine_threadsafe(out_q.put(resp), loop)
            except Exception as exc:
                logger.error("GCP v2 streaming_recognize error: %s", exc)
            finally:
                try: asyncio.run_coroutine_threadsafe(out_q.put(None), loop)
                except Exception: pass

        async def _writer_to_client():
            try:
                while True:
                    resp = await out_q.get()
                    if resp is None: break
                    for result in resp.results or []:
                        payload = _normalize_to_aws_like_payload(result, project)
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_text(json.dumps(payload))
            except Exception as exc: logger.error("Error sending to client: %s", exc)

        tasks = [asyncio.create_task(t) for t in [_reader_from_client(), _writer_to_client()]]
        gcp_future = loop.run_in_executor(None, _gcp_streaming_call)

        try:
            done, pending = await asyncio.wait(tasks + [gcp_future], return_when=asyncio.FIRST_COMPLETED)
            for task in pending: task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            if ws.client_state == WebSocketState.CONNECTED:
                try: await ws.close()
                except Exception: pass
            logger.info("GCP v2 WS session ended.")