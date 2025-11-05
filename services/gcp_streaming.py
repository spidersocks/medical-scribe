import asyncio
import json
import logging
import re
import uuid
import queue
from itertools import chain
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocket, WebSocketState, WebSocketDisconnect

from google.cloud import speech_v2 as speech  # v2 API
from google.cloud import translate_v3 as translate
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import InvalidArgument

from services import nlp  # reuse Comprehend Medical via AWS for English NER

logger = logging.getLogger("stethoscribe-gcp")

GCP_REGION = "us-central1" 

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))

def _spk_label_from_tag(speaker_tag: Optional[int]) -> Optional[str]:
    if not speaker_tag or speaker_tag <= 0:
        return None
    return f"spk_{max(0, speaker_tag - 1)}"

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
    if code.startswith("yue-hant-hk"):
        return "zh-TW"
    if code.startswith("zh-tw") or "hant" in code:
        return "zh-TW"
    if code.startswith("zh-cn") or "hans" in code:
        return "zh-CN"
    if code.startswith("zh"):
        return "zh-TW"
    return "auto"

def _translate_to_english(text: str, project_id: Optional[str] = None, location: str = "global", source_lang: Optional[str] = None) -> str:
    if not text.strip():
        return text
    # Note: Translation API can still use a 'global' location, separate from Speech API
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

    def audio_requests(self, recognizer_name: str):
        while True:
            item = self._q.get()
            if item is None:
                break
            # Each audio chunk request must include the recognizer resource name.
            yield speech.StreamingRecognizeRequest(
                recognizer=recognizer_name,
                audio=item
            )

def _build_gcp_streaming_config(languages: List[str]) -> speech.StreamingRecognitionConfig:
    explicit_config = speech.ExplicitDecodingConfig(
        encoding=speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        audio_channel_count=1,
    )
    
    # --- CHANGE: Provide exactly ONE language code ---
    # The API has conflicting requirements for the 'chirp' model:
    # 1. The `language_codes` field cannot be empty.
    # 2. The multi-language detection feature (using >1 language) is not supported.
    # The solution is to provide a list with a single language code. This satisfies
    # the schema validation without enabling the unsupported feature. The 'chirp'
    # model will still perform its own universal language detection.
    # We use the first language from the client's list, or a default.
    single_language_list = [languages[0]] if languages else ["en-US"]

    config = speech.RecognitionConfig(
        explicit_decoding_config=explicit_config,
        language_codes=single_language_list, # THIS IS THE FIX
        model="chirp",
        features=speech.RecognitionFeatures(
            enable_automatic_punctuation=True,
            enable_word_time_offsets=False,
        ),
    )
    
    return speech.StreamingRecognitionConfig(
        config=config,
        streaming_features=speech.StreamingRecognitionFeatures(
            interim_results=True,
        ),
    )

def _words_to_items(words: List[speech.WordInfo]) -> List[Dict]:
    items = []
    for w in words or []:
        start = _dur_to_seconds(getattr(w, "start_offset", None))
        end = _dur_to_seconds(getattr(w, "end_offset", None))
        items.append(
            {
                "StartTime": round(start, 3),
                "EndTime": round(end, 3),
                "Type": "pronunciation",
                "Content": w.word,
                "Speaker": getattr(w, "speaker_label", None),
            }
        )
    return items

def _normalize_to_aws_like_payload(result: speech.StreamingRecognitionResult, project_id_for_translate: Optional[str]) -> Dict:
    is_final = bool(getattr(result, 'is_final', False))
    alt = result.alternatives[0] if result.alternatives else None
    transcript_text = alt.transcript if alt else ""
    words = list(getattr(alt, "words", []) or [])
    items = _words_to_items(words)

    detected_lang = getattr(result, "language_code", None) or None
    confidence = getattr(alt, "confidence", None)
    
    if is_final:
        logger.info(
            "GCP v2 result: is_final=%s, transcript='%s', detected_lang='%s', confidence=%.3f, alternatives_count=%d",
            is_final, transcript_text[:100], detected_lang, confidence or 0.0, len(result.alternatives) if result.alternatives else 0
        )
    
    if result.alternatives and len(result.alternatives) > 1 and is_final:
        for i, alt_item in enumerate(result.alternatives):
            alt_lang = getattr(alt_item, "language_code", None)
            alt_conf = getattr(alt_item, "confidence", None)
            logger.info("  Alt %d: lang='%s', conf=%.3f, text='%s'", i, alt_lang, alt_conf or 0.0, alt_item.transcript[:50])

    if not detected_lang:
        detected_lang = "en-US" if not _has_cjk(transcript_text) else "cmn-Hant-TW"

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
        "_engine": "gcp",
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
            source_lang = _map_asr_lang_to_translate_source(detected_lang)
            english_text = _translate_to_english(transcript_text, project_id_for_translate, source_lang=source_lang)
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

        if not gcp_project_id:
            logger.error("GCP Project ID is not configured on the server. Closing connection.")
            await ws.close(code=1011, reason="Server configuration error.")
            return

        languages = ws.query_params.get("languages", "en-US,es-US").split(",")
        languages = [lang.strip() for lang in languages if lang.strip()]

        logger.info("GCP WS connected. Project=%s, Languages from client=%s", gcp_project_id, languages)

        client_options = ClientOptions(
            api_endpoint=f"{GCP_REGION}-speech.googleapis.com"
        )
        speech_client = speech.SpeechClient(client_options=client_options)
        
        stream_cfg = _build_gcp_streaming_config(languages)

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
            except WebSocketDisconnect as exc:
                if exc.code == 1000:
                    logger.info("Browser closed the connection gracefully (Code: 1000).")
                else:
                    logger.warning("Browser disconnected with code: %s", exc.code)
                try:
                    await bytes_src.put(None)
                except Exception:
                    pass
            except Exception as exc:
                logger.error("An unexpected error occurred while receiving from client: %s", exc, exc_info=True)
                try:
                    await bytes_src.put(None)
                except Exception:
                    pass
            finally:
                stop_event.set()

        def _gcp_streaming_call(project_id: str):
            recognizer_name = f"projects/{project_id}/locations/{GCP_REGION}/recognizers/_"

            config_request = speech.StreamingRecognizeRequest(
                recognizer=recognizer_name,
                streaming_config=stream_cfg
            )

            all_requests = chain([config_request], bytes_src.audio_requests(recognizer_name))
            
            try:
                responses = speech_client.streaming_recognize(requests=all_requests)
                for resp in responses:
                    asyncio.run_coroutine_threadsafe(out_q.put(resp), loop)
            except InvalidArgument as exc:
                logger.error(
                    "GCP streaming_recognize error due to InvalidArgument: %s. "
                    "This may be due to an unsupported feature combination with the 'chirp' model.",
                    exc,
                    exc_info=True
                )
            except Exception as exc:
                logger.error("GCP streaming_recognize error: %s", exc, exc_info=True)
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
                        if not result.alternatives or not result.alternatives[0].transcript:
                            continue
                        payload = _normalize_to_aws_like_payload(result, gcp_project_id)
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_text(json.dumps(payload))
            except Exception as exc:
                logger.error("Error sending to client: %s", exc)

        reader_task = asyncio.create_task(_reader_from_client())
        writer_task = asyncio.create_task(_writer_to_client())
        
        _ = asyncio.get_running_loop().run_in_executor(None, _gcp_streaming_call, gcp_project_id)

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