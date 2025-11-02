import asyncio
import json
import logging
import re
import uuid
import queue
import itertools
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketState

from google.cloud import speech_v1p1beta1 as speech
from google.cloud import translate_v3 as translate

from services import nlp  # reuse Comprehend Medical via AWS for English NER

logger = logging.getLogger("stethoscribe-gcp")

# ---- Basic helpers ----

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))

def _spk_label_from_tag(speaker_tag: Optional[int]) -> Optional[str]:
    if not speaker_tag or speaker_tag <= 0:
        return None
    return f"spk_{max(0, speaker_tag - 1)}"

def _dur_to_seconds(ts) -> float:
    """
    Convert GCP WordInfo timestamps (google.protobuf Duration or datetime.timedelta) to seconds.
    """
    if ts is None:
        return 0.0
    try:
        return float(ts.total_seconds())
    except Exception:
        pass
    secs = float(getattr(ts, "seconds", 0) or 0)
    nanos = float(getattr(ts, "nanos", 0) or 0)
    return secs + nanos / 1e9


# ---- GCP Translation (v3) ----

_translate_client: Optional[translate.TranslationServiceClient] = None

def _get_translate_client() -> translate.TranslationServiceClient:
    global _translate_client
    if _translate_client is None:
        _translate_client = translate.TranslationServiceClient()
    return _translate_client

def _map_asr_lang_to_translate_source(asr_code: Optional[str]) -> str:
    """
    Map Speech-to-Text detected language -> Cloud Translation source language code.
    Ensure Cantonese uses Traditional Chinese path; bias Traditional for HK.
    """
    if not asr_code:
        return "auto"
    code = asr_code.lower()

    if code.startswith("en"):
        return "en"
    if code.startswith("yue-hant-hk"):  # Cantonese (Traditional, HK)
        return "zh-TW"
    if code.startswith("zh-tw") or "hant" in code:  # Traditional Chinese
        return "zh-TW"
    if code.startswith("zh-cn") or "hans" in code:  # Simplified Chinese
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
        logger.warning("GCP Translation failed (src=%s). Returning original. Error: %s", source_lang or "auto", exc)
    return text


# ---- GCP Speech streaming plumbing ----

class _QueueBytesSource:
    """
    Bridges WS audio bytes into a blocking generator read by Speech client in a worker thread.
    Uses thread-safe queue.Queue to avoid cross-thread asyncio issues.
    """
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

    def audio_requests(self):
        """
        Yield audio_content requests. The StreamingRecognitionConfig is passed separately to the API.
        Runs in a worker thread and blocks on queue.get().
        """
        while True:
            item = self._q.get()
            if item is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=item)


def _build_gcp_streaming_config(primary_lang: str, alt_langs: List[str]) -> speech.StreamingRecognitionConfig:
    """
    IMPORTANT: Do NOT set use_enhanced/model with alt languages in v1; many enhanced models reject alt codes.
    """
    diarization = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=2,
    )

    recog = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=primary_lang,
        alternative_language_codes=alt_langs,  # e.g., ["en-US","yue-Hant-HK","zh-TW"]
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        diarization_config=diarization,
    )

    return speech.StreamingRecognitionConfig(
        config=recog,
        interim_results=True,
        single_utterance=False,
    )


def _words_to_items(words: List[speech.WordInfo]) -> List[Dict]:
    items = []
    for w in words or []:
        start = _dur_to_seconds(getattr(w, "start_time", None))
        end = _dur_to_seconds(getattr(w, "end_time", None))
        items.append(
            {
                "StartTime": round(start, 3),
                "EndTime": round(end, 3),
                "Type": "pronunciation",
                "Content": w.word,
                "Speaker": _spk_label_from_tag(getattr(w, "speaker_tag", None)),
            }
        )
    return items


def _normalize_to_aws_like_payload(
    result: speech.StreamingRecognitionResult,
    project_id_for_translate: Optional[str],
) -> Dict:
    is_final = bool(result.is_final)
    alt = result.alternatives[0] if result.alternatives else None
    transcript_text = alt.transcript if alt else ""
    words = list(getattr(alt, "words", []) or [])
    items = _words_to_items(words)

    detected_lang = getattr(alt, "language_code", None) or None
    if not detected_lang:
        detected_lang = "en-US" if not _has_cjk(transcript_text) else "yue-Hant-HK"

    payload = {
        "Transcript": {
            "Results": [
                {
                    "Alternatives": [
                        {"Transcript": transcript_text, "Items": items}
                    ],
                    "ResultId": str(uuid.uuid4()),
                    "IsPartial": not is_final,
                    "LanguageCode": detected_lang,
                }
            ]
        },
        "_engine": "gcp",
        "_detected_language": detected_lang,
    }

    # Avoid emitting empty finals (prevents blank segments/422s)
    if is_final and not transcript_text.strip():
        payload["Transcript"]["Results"][0]["IsPartial"] = True
        return payload

    if is_final and transcript_text.strip():
        payload["DisplayText"] = transcript_text

        if detected_lang.lower().startswith("en"):
            english_text = transcript_text
        else:
            source_lang = _map_asr_lang_to_translate_source(detected_lang)
            english_text = _translate_to_english(
                transcript_text,
                project_id_for_translate,
                source_lang=source_lang,
            )
            if english_text and english_text != transcript_text:
                payload["TranslatedText"] = english_text

        try:
            ents = nlp.detect_entities(english_text)
        except Exception:
            ents = []
        payload["ComprehendEntities"] = ents

    return payload


def register_gcp_streaming_routes(app: FastAPI, *, gcp_project_id: Optional[str] = None) -> None:
    @app.websocket("/client-transcribe-gcp")
    async def client_transcribe_gcp(ws: WebSocket):
        await ws.accept()

        primary_ui = ws.query_params.get("language_code", "auto") or "auto"

        # Bias "auto" toward Chinese pickup (Cantonese primary improves CJK detection)
        if primary_ui == "auto":
            primary = "yue-Hant-HK"
            alts: List[str] = ["en-US", "zh-TW"]  # use zh-TW (v1 supported), not cmn-Hant-TW
            mode = "auto"
        else:
            mode = primary_ui
            if primary_ui == "en-US":
                primary = "en-US"
                alts = ["yue-Hant-HK", "zh-TW"]
            elif primary_ui == "zh-HK":
                primary = "yue-Hant-HK"
                alts = ["en-US", "zh-TW"]
            elif primary_ui == "zh-TW":
                primary = "zh-TW"
                alts = ["en-US", "yue-Hant-HK"]
            else:
                primary = "yue-Hant-HK"
                alts = ["en-US", "zh-TW"]
                mode = "auto"

        logger.info("GCP WS connected. Mode=%s | Primary=%s | Alts=%s", mode, primary, alts)

        speech_client = speech.SpeechClient()
        stream_cfg = _build_gcp_streaming_config(primary, alts)

        bytes_src = _QueueBytesSource()
        out_q: "asyncio.Queue[Optional[speech.StreamingRecognizeResponse]]" = asyncio.Queue()
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()  # capture main loop for thread-safe handoff

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

        def _gcp_streaming_call():
            """
            Runs in a worker thread. Sends audio requests and pushes responses back
            to the main asyncio loop via out_q using run_coroutine_threadsafe.
            """
            try:
                # Pass config explicitly; generator yields only audio_content requests.
                responses = speech_client.streaming_recognize(
                    config=stream_cfg,
                    requests=bytes_src.audio_requests(),
                )
                for resp in responses:
                    asyncio.run_coroutine_threadsafe(out_q.put(resp), loop)
            except Exception as exc:
                logger.error("GCP streaming_recognize error: %s", exc)
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
            logger.info("GCP WS session ended.")