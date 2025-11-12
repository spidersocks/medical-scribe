"""
Alibaba Paraformer real-time transcription via RAW WebSocket "run-task" message.

This replaces the previous SDK-based approach so we can explicitly send:
  diarization_enabled: True
  speaker_count: <env ALIBABA_SPEAKER_COUNT>
  enable_words: True
  format: pcm
  sample_rate: 16000
  channel_num: 1

and obtain speaker_id in each sentence (and word-level speakers if the
service returns them). We then reshape responses into the AWS-style
payload the existing frontend expects.

Environment variables used:
  DASHSCOPE_API_KEY          (required)
  ALIBABA_SPEAKER_COUNT      (optional, default 2)
  DEBUG_ALIBABA_TRANSCRIBE   (optional, set to '1' for verbose debug)
  ENABLE_ALIBABA_TRANSLATION (optional '1')
  ENABLE_ALIBABA_ENTITIES    (optional '1')

If translation/entities flags are set and services.nlp is available,
we run lightweight enrichment per final segment.

Endpoint (unchanged for the frontend):
  ws://<host>/transcribe/alibaba
"""
import asyncio
import json
import logging
import os
import uuid
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

try:
    from services import nlp  # translate + entities
except ImportError:
    nlp = None

router = APIRouter()
logger = logging.getLogger(__name__)

# -------------------- Flags / Config --------------------
DBG = os.getenv("DEBUG_ALIBABA_TRANSCRIBE") == "1"
DO_TRANSLATE = os.getenv("ENABLE_ALIBABA_TRANSLATION") == "1"
DO_ENTITIES = os.getenv("ENABLE_ALIBABA_ENTITIES") == "1"
SPEAKER_COUNT = int(os.getenv("ALIBABA_SPEAKER_COUNT", "2"))
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# Official raw inference WebSocket (run-task)
ALIBABA_WS_URL = os.getenv(
    "ALIBABA_INFERENCE_WS_URL",
    "wss://dashscope.aliyuncs.com/api-ws/v1/inference/",
)

if DBG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# -------------------- Language Mapping / Heuristics --------------------
CANTONESE_HINT_CHARS = set("嘅咩喺咗冇哋啲呢啦噉仲咪睇醫")

def map_lang_to_ui(lang_raw: Optional[str], text: str) -> str:
    if lang_raw:
        lr = lang_raw.lower()
        if lr.startswith("en"): return "en-US"
        if "yue" in lr or "cant" in lr: return "zh-HK"
        if lr.startswith("zh"): return "zh-TW"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        if any(ch in CANTONESE_HINT_CHARS for ch in text):
            return "zh-HK"
        return "zh-TW"
    return "en-US"

# -------------------- Speaker Extraction --------------------
WORD_SPEAKER_KEYS = ("speaker", "spk", "speaker_id", "spk_id")

def _norm_spk(val: Any) -> Optional[str]:
    if val is None: return None
    s = str(val).strip()
    if not s: return None
    if s.startswith("spk_"): return s
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits: return f"spk_{int(digits)}"
    if s.isdigit(): return f"spk_{int(s)}"
    return s.lower()

def extract_majority_word_speaker(words: Any) -> Optional[str]:
    if not isinstance(words, list) or not words: return None
    speakers: List[str] = []
    for w in words:
        if not isinstance(w, dict): continue
        raw = None
        for k in WORD_SPEAKER_KEYS:
            if k in w:
                raw = w[k]
                break
        sp = _norm_spk(raw)
        if sp:
            speakers.append(sp)
    if not speakers:
        return None
    winner, _ = Counter(speakers).most_common(1)[0]
    return winner

def extract_sentence_speaker(sentence: Dict[str, Any]) -> Tuple[Optional[str], str]:
    for key in ("speaker", "speaker_id", "spk", "spk_id"):
        raw = sentence.get(key)
        sp = _norm_spk(raw)
        if sp:
            return sp, "sentence"
    words = sentence.get("words")
    sp_words = extract_majority_word_speaker(words)
    if sp_words:
        return sp_words, "words"
    return None, "none"

# -------------------- AWS-Shaped Payload --------------------
def build_aws_payload(
    *,
    text: str,
    is_partial: bool,
    result_id: str,
    speaker: Optional[str],
    language_code: str,
    start_ms: Optional[int],
    end_ms: Optional[int],
    translated_text: Optional[str],
    entities: Optional[List[dict]],
    debug: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    items = []
    if speaker:
        items.append({"Type": "pronunciation", "Speaker": speaker})
    result_obj: Dict[str, Any] = {
        "ResultId": result_id,
        "IsPartial": is_partial,
        "Alternatives": [{"Transcript": text, "Items": items}],
        "LanguageCode": language_code,
    }
    if speaker:
        result_obj["Speaker"] = speaker
    if start_ms is not None:
        result_obj["StartTimeMs"] = start_ms
    if end_ms is not None:
        result_obj["EndTimeMs"] = end_ms
    payload: Dict[str, Any] = {
        "Transcript": {"Results": [result_obj]},
        "DisplayText": text,
        "TranslatedText": translated_text,
        "ComprehendEntities": entities or [],
    }
    if debug and DBG:
        payload["_debug"] = debug
    return payload

# -------------------- Alibaba Message Helpers --------------------
def build_run_task_message() -> Dict[str, Any]:
    """
    Raw 'run-task' frame per Alibaba guidance.
    """
    task_id = str(uuid.uuid4())
    return {
        "header": {
            "action": "run-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {
            "task_group": "audio",
            "task": "asr",
            "function": "recognition",
            "model": "paraformer-realtime-v2",
            "parameters": {
                "format": "pcm",
                "sample_rate": 16000,
                "diarization_enabled": True,
                "speaker_count": SPEAKER_COUNT,     # optional guidance
                "enable_words": True,               # request word-level tokens
                "channel_num": 1,                   # mono requirement
                "enable_punctuation_prediction": True,
                "enable_inverse_text_normalization": True,
            },
            "input": {},  # real-time audio will follow as binary frames
        },
    }

def parse_incoming(text_data: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Returns (sentence_dict, is_final).
    The 'payload' may contain either 'sentence' (final) or 'result' (interim) depending on API version.
    We standardize to a sentence dict with keys we use downstream.
    """
    try:
        obj = json.loads(text_data)
    except Exception:
        return None, False

    payload = obj.get("payload") or {}
    # The real-time responses usually embed recognized sentence here:
    # - payload["sentence"] for final
    # - payload["sentence"]["text"] partial updates with 'is_sentence_end': False
    sentence = payload.get("sentence")
    if not isinstance(sentence, dict):
        # Some versions might use 'result' for partial. Try fallback.
        sentence = payload.get("result")

    if not isinstance(sentence, dict):
        return None, False

    # Determine final: speaker diarization doc says 'is_sentence_end' or 'sentence_end'
    is_final = bool(
        sentence.get("is_sentence_end")
        or sentence.get("sentence_end")
        or sentence.get("end_time")  # sentences with end_time usually are final
    )
    return sentence, is_final

# -------------------- WebSocket Endpoint --------------------
@router.websocket("/alibaba")
async def transcribe_alibaba(websocket: WebSocket):
    """
    Raw WebSocket bridge:
      1. Accept browser connection.
      2. Open Alibaba inference WebSocket.
      3. Send run-task message with diarization + words enabled.
      4. Forward PCM 16k mono frames from browser to Alibaba.
      5. Convert Alibaba responses to existing AWS-shaped payloads.
    """
    await websocket.accept()
    logger.info("Client connected -> /transcribe/alibaba")

    if not API_KEY:
        await websocket.close(code=1011, reason="DASHSCOPE_API_KEY missing")
        return

    # Track segment IDs
    final_counter = 0

    async with aiohttp.ClientSession() as session:
        try:
            upstream = await session.ws_connect(
                ALIBABA_WS_URL,
                headers={"Authorization": f"bearer {API_KEY}"},
                max_msg_size=0,
            )
        except Exception as exc:
            logger.exception("Failed to connect to Alibaba ASR: %s", exc)
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close(code=1011, reason="Upstream unavailable")
            return

        # Send run-task
        start_msg = build_run_task_message()
        await upstream.send_str(json.dumps(start_msg))
        logger.info("Sent run-task (diarization_enabled=True, speaker_count=%s, enable_words=True)", SPEAKER_COUNT)

        # For timing fallback if begin_time/end_time unreliable
        session_start_monotonic = time.monotonic()

        async def forward_audio():
            try:
                while True:
                    data = await websocket.receive_bytes()
                    if not data:
                        # No more audio – simply stop sending frames.
                        break
                    await upstream.send_bytes(data)
            except WebSocketDisconnect:
                logger.info("Client disconnected (audio).")
            except Exception as e:
                logger.error("forward_audio error: %s", e)

        async def forward_results():
            nonlocal final_counter
            try:
                async for msg in upstream:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        sentence, is_final = parse_incoming(msg.data)
                        if not isinstance(sentence, dict):
                            continue

                        text = sentence.get("text") or ""
                        if not text:
                            continue

                        # Timing
                        try:
                            start_ms = int(sentence.get("begin_time")) if sentence.get("begin_time") is not None else None
                        except Exception:
                            start_ms = None
                        try:
                            end_ms = int(sentence.get("end_time")) if (is_final and sentence.get("end_time") is not None) else None
                        except Exception:
                            end_ms = None

                        if start_ms is None:
                            # Fallback approximate timing
                            start_ms = int((time.monotonic() - session_start_monotonic) * 1000)

                        sentence_id = sentence.get("sentence_id")
                        lang_raw = sentence.get("language") or sentence.get("lang") or sentence.get("language_code")
                        language_code = map_lang_to_ui(lang_raw, text)

                        # Speaker extraction
                        speaker_label, speaker_source = extract_sentence_speaker(sentence)

                        # Unique ID
                        if is_final:
                            final_counter += 1
                            rid = f"alibaba-seg-{sentence_id if sentence_id is not None else final_counter}-{uuid.uuid4().hex[:8]}"
                        else:
                            rid = f"alibaba-temp-{sentence_id if sentence_id is not None else final_counter}"

                        translated_text: Optional[str] = None
                        entities: Optional[List[dict]] = None
                        if is_final:
                            if DO_TRANSLATE and nlp and not language_code.startswith("en"):
                                translated_text = nlp.to_english(text, language_code)
                            if DO_ENTITIES and nlp:
                                analysis = translated_text or text
                                entities = nlp.detect_entities(analysis)

                        debug_block = None
                        if DBG:
                            word0_keys = None
                            words = sentence.get("words")
                            if isinstance(words, list) and words and isinstance(words[0], dict):
                                word0_keys = list(words[0].keys())
                            debug_block = {
                                "sentence_id": sentence_id,
                                "is_final": is_final,
                                "speaker_extracted": speaker_label,
                                "speaker_source": speaker_source,
                                "language_raw": lang_raw,
                                "language_final": language_code,
                                "begin_time": sentence.get("begin_time"),
                                "end_time": sentence.get("end_time"),
                                "word0_keys": word0_keys,
                                "result_id": rid,
                            }

                        payload = build_aws_payload(
                            text=text,
                            is_partial=not is_final,
                            result_id=rid,
                            speaker=speaker_label,
                            language_code=language_code,
                            start_ms=start_ms,
                            end_ms=end_ms,
                            translated_text=translated_text,
                            entities=entities,
                            debug=debug_block,
                        )

                        if websocket.client_state == WebSocketState.CONNECTED:
                            await websocket.send_text(json.dumps(payload))

                    elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                        break
            except Exception as e:
                logger.error("forward_results error: %s", e)

        # Run both directions concurrently
        try:
            t_audio = asyncio.create_task(forward_audio())
            t_results = asyncio.create_task(forward_results())
            done, pending = await asyncio.wait({t_audio, t_results}, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
                try:
                    await t
                except Exception:
                    pass
        finally:
            try:
                await upstream.close()
            except Exception:
                pass
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.close()
                except Exception:
                    pass
            logger.info("Alibaba transcription session ended (raw WebSocket).")