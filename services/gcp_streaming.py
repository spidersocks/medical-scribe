import asyncio
import json
import logging
import re
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Tuple

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


# ---- GCP Translation (v3) ----

_translate_client: Optional[translate.TranslationServiceClient] = None

def _get_translate_client() -> translate.TranslationServiceClient:
    global _translate_client
    if _translate_client is None:
        _translate_client = translate.TranslationServiceClient()
    return _translate_client

def _translate_to_english(text: str, project_id: Optional[str] = None, location: str = "global") -> str:
    """
    Translate arbitrary text to English using Cloud Translation v3.
    - If you have a project id in env (GOOGLE_CLOUD_PROJECT), you can pass it through main and fill here.
    """
    if not text.strip():
        return text
    parent = f"projects/{project_id or '-'}/locations/{location}"
    try:
        resp = _get_translate_client().translate_text(
            request={
                "parent": parent,
                "contents": [text],
                "mime_type": "text/plain",
                "source_language_code": "auto",
                "target_language_code": "en",
            }
        )
        if resp and resp.translations:
            return resp.translations[0].translated_text or text
    except Exception as exc:
        logger.warning("GCP Translation failed, returning original. Error: %s", exc)
    return text


# ---- GCP Speech streaming plumbing ----

class _QueueBytesSource:
    """
    Bridges websocket audio bytes into the blocking generator the Speech client expects.
    """
    def __init__(self):
        self._q: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._closed = False

    async def put(self, data: Optional[bytes]):
        if self._closed:
            return
        await self._q.put(data)

    def close(self):
        self._closed = True

    def generator(self):
        # First request carries the config; callers should send it before consuming this.
        while True:
            item = asyncio.get_event_loop().run_until_complete(self._q.get())
            if item is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=item)


def _build_gcp_streaming_config(primary_lang: str, alt_langs: List[str]) -> speech.StreamingRecognitionConfig:
    diarization = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=2,
    )

    recog = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=primary_lang,
        alternative_language_codes=alt_langs,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        use_enhanced=True,
        model="phone_call",  # pragmatic choice for 2-speaker conversations
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
        start = (w.start_time.seconds or 0) + (w.start_time.nanos or 0) / 1e9
        end = (w.end_time.seconds or 0) + (w.end_time.nanos or 0) / 1e9
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
    """
    Normalize one GCP result into the AWS Transcribe-like envelope the frontend expects.
    - Includes interim/final flag, Alternatives[0].Transcript, Alternatives[0].Items with Speaker tags,
      and on finals: DisplayText, TranslatedText (if needed), ComprehendEntities.
    """
    is_final = bool(result.is_final)
    alt = result.alternatives[0] if result.alternatives else None
    transcript_text = alt.transcript if alt else ""
    words = list(getattr(alt, "words", []) or [])
    items = _words_to_items(words)

    # GCP may include detected language per alternative
    detected_lang = getattr(alt, "language_code", None) or None
    if not detected_lang:
        # Fallback heuristic
        detected_lang = "en-US" if not _has_cjk(transcript_text) else "yue-Hant-HK"

    # Build base payload with interim flag
    payload = {
        "Transcript": {
            "Results": [
                {
                    "Alternatives": [
                        {
                            "Transcript": transcript_text,
                            "Items": items,
                        }
                    ],
                    "ResultId": str(uuid.uuid4()),
                    "IsPartial": not is_final,
                    "LanguageCode": detected_lang,
                }
            ]
        }
    }

    if is_final and transcript_text.strip():
        # Top-level fields for the UI and persistence
        payload["DisplayText"] = transcript_text

        if detected_lang.lower().startswith("en"):
            english_text = transcript_text
        else:
            # Use GCP Translate v3
            english_text = _translate_to_english(transcript_text, project_id_for_translate)
            if english_text and english_text != transcript_text:
                payload["TranslatedText"] = english_text

        # Run English NER via existing AWS Comprehend (services.nlp)
        try:
            ents = nlp.detect_entities(english_text)
        except Exception:
            ents = []
        payload["ComprehendEntities"] = ents

    return payload


def register_gcp_streaming_routes(app: FastAPI, *, gcp_project_id: Optional[str] = None) -> None:
    """
    Registers /client-transcribe-gcp WebSocket endpoint to stream audio to GCP Speech and
    forward normalized messages to the browser.

    Query param:
      - language_code in { "en-US", "zh-HK", "zh-TW" } used as primary hint.

    Notes:
      - Audio frames must be 16kHz PCM16 mono (the existing AudioWorklet already sends that).
      - Messages are shaped to match the existing frontend contract.
    """
    @app.websocket("/client-transcribe-gcp")
    async def client_transcribe_gcp(ws: WebSocket):
        await ws.accept()

        # Map UI codes -> GCP language codes
        primary_ui = ws.query_params.get("language_code", "en-US")
        if primary_ui not in {"en-US", "zh-HK", "zh-TW"}:
            primary_ui = "en-US"

        primary = "en-US"
        alts: List[str] = []

        if primary_ui == "en-US":
            primary = "en-US"
            alts = ["yue-Hant-HK", "cmn-Hant-TW"]
        elif primary_ui == "zh-HK":
            primary = "yue-Hant-HK"
            alts = ["en-US", "cmn-Hant-TW"]
        elif primary_ui == "zh-TW":
            primary = "cmn-Hant-TW"
            alts = ["en-US", "yue-Hant-HK"]

        logger.info("GCP WS connected. Primary=%s, Alts=%s", primary, alts)

        speech_client = speech.SpeechClient()
        stream_cfg = _build_gcp_streaming_config(primary, alts)

        # Queue bridges WS -> GCP generator
        bytes_src = _QueueBytesSource()
        stop_event = asyncio.Event()

        # Build blocking generator with first config message prepended
        def _request_generator():
            # Initial config request
            yield speech.StreamingRecognizeRequest(streaming_config=stream_cfg)
            # Audio content messages
            for req in bytes_src.generator():
                yield req

        out_q: asyncio.Queue[Optional[speech.StreamingRecognizeResponse]] = asyncio.Queue()

        async def _reader_from_client():
            try:
                while True:
                    data = await ws.receive_bytes()
                    if not data:
                        # Treat empty payload as end-of-stream signal
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
            try:
                responses = speech_client.streaming_recognize(requests=_request_generator())
                for resp in responses:
                    asyncio.run_coroutine_threadsafe(out_q.put(resp), asyncio.get_event_loop())
            except Exception as exc:
                logger.error("GCP streaming_recognize error: %s", exc)
            finally:
                asyncio.run_coroutine_threadsafe(out_q.put(None), asyncio.get_event_loop())

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

        # Start tasks
        reader_task = asyncio.create_task(_reader_from_client())
        writer_task = asyncio.create_task(_writer_to_client())
        gcp_task = asyncio.get_running_loop().run_in_executor(None, _gcp_streaming_call)

        # Wait on termination
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
            # Ensure background finishes
            try:
                await out_q.put(None)
            except Exception:
                pass

            logger.info("GCP WS session ended.")