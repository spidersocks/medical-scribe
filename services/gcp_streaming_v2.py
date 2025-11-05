import asyncio
import json
import logging
import re
import uuid
import queue
from typing import Dict, List, Optional, Iterable, Tuple

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketState

from google.cloud import speech_v2 as speech
from google.cloud import translate_v3 as translate
from google.api_core.client_options import ClientOptions  # NEW

from services import nlp  # reuse AWS Comprehend Medical for English NER

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
    if code.startswith("zh-tw") or "hant" in code:
        return "zh-TW"
    if code.startswith("zh-cn") or "hans" in code:
        return "zh-CN"
    if code.startswith("zh"):
        return "zh-TW"
    return "auto"

def _translate_to_english(
    text: str,
    project_id: Optional[str] = None,
    location: str = "global",
    source_lang: Optional[str] = None,
) -> str:
    if not text.strip():
        return text
    parent = f"projects/{project_id or '-'}/locations/{location}"
    req = {
        "parent": parent,
        "contents": [text],
        "mime_type": "text/plain",
        "source_language_code": source_lang or "auto",
        "target_language_code": "en",
    }
    try:
        resp = _get_translate_client().translate_text(request=req)
        if resp and resp.translations:
            return resp.translations[0].translated_text or text
    except Exception as exc:
        logger.warning("Translate v3 failed (src=%s). Falling back to original. Error: %s", source_lang or "auto", exc)
    return text

class _QueueBytesSource:
    def __init__(self):
        self._q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._closed = False

    async def put(self, data: Optional[bytes]):
        if self._closed:
            return
        try:
            self._q.put_nowait(data)
        except queue.Full:
            self._q.put(data)

    def close(self):
        self._closed = True

    def audio_requests(self) -> Iterable[speech.StreamingRecognizeRequest]:
        while True:
            item = self._q.get()
            if item is None:
                break
            yield speech.StreamingRecognizeRequest(audio=item)

def _build_v2_streaming_config(language_codes: List[str]) -> speech.StreamingRecognitionConfig:
    """
    Build a v2 StreamingRecognitionConfig.
    Diarization is configured via RecognitionFeatures.diarization_config in v2-capable clients.
    """
    base_features_kwargs = dict(
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
    )

    # Try enabling diarization via features; fallback without diarization if unsupported by installed SDK
    try:
        features = speech.RecognitionFeatures(
            **base_features_kwargs,
            diarization_config=speech.SpeakerDiarizationConfig(
                min_speaker_count=2,
                max_speaker_count=2,
            ),
        )
    except Exception as exc:
        logger.warning("GCP v2: diarization_config unsupported in this client/version; continuing without diarization. %s", exc)
        features = speech.RecognitionFeatures(**base_features_kwargs)

    cfg = speech.RecognitionConfig(
        auto_decoding_config=speech.AutoDetectDecodingConfig(),
        language_codes=language_codes,   # true multi-language list (no primary)
        model="latest_long",             # try latest_long first; change to latest_short if you want less latency
        features=features,
    )
    return speech.StreamingRecognitionConfig(config=cfg)

def _words_to_items(words) -> List[Dict]:
    items = []
    for w in list(words or []):
        start = _dur_to_seconds(getattr(w, "start_offset", None))
        end = _dur_to_seconds(getattr(w, "end_offset", None))
        items.append(
            {
                "StartTime": round(start, 3),
                "EndTime": round(end, 3),
                "Type": "pronunciation",
                "Content": getattr(w, "word", ""),
                "Speaker": _spk_label_from_any(w),
            }
        )
    return items

def _normalize_to_aws_like_payload(
    result: speech.StreamingRecognitionResult,
    project_id_for_translate: Optional[str],
) -> Dict:
    is_final = bool(result.is_final)
    alt = result.alternatives[0] if result.alternatives else None
    transcript_text = getattr(alt, "transcript", "") if alt else ""
    words = list(getattr(alt, "words", []) or [])
    items = _words_to_items(words)

    detected_lang = getattr(alt, "language_code", None)
    if not detected_lang:
        detected_lang = "en-US" if not _has_cjk(transcript_text) else "yue-Hant-HK"

    payload = {
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

    if is_final and not transcript_text.strip():
        payload["Transcript"]["Results"][0]["IsPartial"] = True
        return payload

    if is_final and transcript_text.strip():
        payload["DisplayText"] = transcript_text
        if detected_lang.lower().startswith("en"):
            english_text = transcript_text
        else:
            english_text = _translate_to_english(
                transcript_text,
                project_id_for_translate,
                source_lang=_map_asr_lang_to_translate_source(detected_lang),
            )
            if english_text and english_text != transcript_text:
                payload["TranslatedText"] = english_text

        try:
            ents = nlp.detect_entities(english_text)
        except Exception:
            ents = []
        payload["ComprehendEntities"] = ents

    return payload

def _normalize_lang_list_param(raw: str) -> List[str]:
    """
    Accept 'en-US,yue-Hant-HK,zh-TW?foo=bar' and normalize to ['en-US','yue-Hant-HK','zh-TW'].
    """
    if not raw:
        return []
    raw = raw.split("?", 1)[0]
    parts = re.split(r"[,\s&]+", raw.strip())
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p == "zh-HK":
            out.append("yue-Hant-HK")
        else:
            out.append(p)
    # de-duplicate preserving order
    seen = set()
    dedup = []
    for c in out:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup

def _normalize_v2_langs_and_location(codes: List[str], requested_location: str) -> Tuple[List[str], str]:
    """
    v2 compatibility normalizations:
    - Map zh-TW -> cmn-Hant-TW for better v2 coverage.
    - Keep yue-Hant-HK as-is for Cantonese.
    - If any Chinese (yue/zh/cmn) is requested and location=='global', use asia-east1 (widest zh/yue coverage).
    """
    normalized: List[str] = []
    has_cjk = False
    for c in codes:
        cl = c.lower()
        if cl in ("zh-tw", "cmn-hant-tw"):
            normalized.append("cmn-Hant-TW")
            has_cjk = True
        elif cl == "zh-hk":
            normalized.append("yue-Hant-HK")
            has_cjk = True
        elif cl.startswith("yue"):
            normalized.append("yue-Hant-HK")
            has_cjk = True
        elif cl.startswith("zh"):
            # If user passed generic zh, prefer Traditional path in HK deployment
            normalized.append("cmn-Hant-TW")
            has_cjk = True
        else:
            normalized.append(c)
    # Location normalization
    loc = (requested_location or "global").strip() or "global"
    if has_cjk and loc.lower() == "global":
        loc = "asia-east1"
    return normalized, loc

def register_gcp_streaming_v2_routes(app: FastAPI, *, gcp_project_id: Optional[str] = None, gcp_location: str = "global") -> None:
    @app.websocket("/client-transcribe-gcp-v2")
    async def client_transcribe_gcp_v2(ws: WebSocket):
        await ws.accept()

        # Parse language list or default to all three
        raw = (ws.query_params.get("languages") or "").strip()
        requested_codes = _normalize_lang_list_param(raw) if raw else ["en-US", "yue-Hant-HK", "zh-TW"]

        # Determine location (env default, overridable via query)
        requested_location = (ws.query_params.get("location") or gcp_location or "global").strip()

        # Normalize for v2 support (maps zh-TW -> cmn-Hant-TW; switches global -> asia-east1 when CJK present)
        language_codes, location = _normalize_v2_langs_and_location(requested_codes, requested_location)

        logger.info("GCP v2 WS connected. language_codes=%s | location=%s", language_codes, location)

        # Build a region-matching Speech client for v2
        # v2 requires the API endpoint region to match the recognizer location
        if location.lower() == "global":
            speech_client = speech.SpeechClient()  # default global endpoint
        else:
            api_endpoint = f"{location}-speech.googleapis.com"
            logger.info("GCP v2 using regional endpoint: %s", api_endpoint)
            speech_client = speech.SpeechClient(client_options=ClientOptions(api_endpoint=api_endpoint))

        streaming_config = _build_v2_streaming_config(language_codes)

        # Recognizer shortcut resource ("_") allows inline configs without creating a named recognizer
        project = gcp_project_id or "-"
        recognizer = f"projects/{project}/locations/{location}/recognizers/_"

        bytes_src = _QueueBytesSource()
        out_q: "asyncio.Queue[Optional[speech.StreamingRecognizeResponse]]" = asyncio.Queue()
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        async def _reader_from_client():
            try:
                while True:
                    data = await ws.receive_bytes()
                    if not data:
                        await bytes_src.put(None)
                        break
                    await bytes_src.put(data)
            except Exception as exc:
                logger.info("Browser disconnected or receive error: %s", exc)
                try:
                    await bytes_src.put(None)
                except Exception:
                    pass
            finally:
                stop_event.set()

        def _request_generator():
            # First request includes recognizer and streaming_config
            yield speech.StreamingRecognizeRequest(
                streaming_config=streaming_config,
                recognizer=recognizer,
            )
            # Followed by audio chunks
            for req in bytes_src.audio_requests():
                yield req

        def _gcp_streaming_call():
            try:
                responses = speech_client.streaming_recognize(requests=_request_generator())
                for resp in responses:
                    asyncio.run_coroutine_threadsafe(out_q.put(resp), loop)
            except Exception as exc:
                logger.error("GCP v2 streaming_recognize error: %s", exc)
            finally:
                try:
                    asyncio.run_coroutine_threadsafe(out_q.put(None), loop)
                except Exception:
                    pass

        async def _writer_to_client():
            try:
                while True:
                    resp = await out_q.get()
                    if resp is None:
                        break
                    for result in resp.results or []:
                        payload = _normalize_to_aws_like_payload(result, gcp_project_id)
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_text(json.dumps(payload))
            except Exception as exc:
                logger.error("Error sending to client: %s", exc)

        reader_task = asyncio.create_task(_reader_from_client())
        writer_task = asyncio.create_task(_writer_to_client())
        _ = asyncio.get_running_loop().run_in_executor(None, _gcp_streaming_call)

        try:
            await asyncio.wait(
                [reader_task, writer_task, asyncio.create_task(stop_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            try:
                await ws.close()
            except Exception:
                pass
            try:
                await out_q.put(None)
            except Exception:
                pass
            logger.info("GCP v2 WS session ended.")