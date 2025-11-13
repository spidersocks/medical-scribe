from dotenv import load_dotenv
load_dotenv()

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import struct
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote
from zlib import crc32

import aiohttp
import boto3
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from starlette.websockets import WebSocketState
from starlette.middleware.gzip import GZipMiddleware

# Application-specific imports
from prompts import PROMPT_REGISTRY, get_prompt_generator
from services import template_service
from config import settings
from prompts.base import BUILTIN_NOTE_KEYS  # ✅ Added per request

logger = logging.getLogger(__name__)

# API router and services
try:
    from api import api_router
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "api_router could not be imported. Ensure your api package exposes `api_router`."
    ) from exc


# ---- AWS clients lazily created ----
@lru_cache(maxsize=1)
def get_boto3_session() -> boto3.session.Session:
    kwargs: Dict[str, Any] = {"region_name": settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    if settings.aws_session_token:
        kwargs["aws_session_token"] = settings.aws_session_token
    return boto3.session.Session(**kwargs)


@lru_cache(maxsize=1)
def get_comprehend_client():
    return get_boto3_session().client("comprehendmedical")


@lru_cache(maxsize=1)
def get_bedrock_client():
    return get_boto3_session().client("bedrock-runtime")


@lru_cache(maxsize=1)
def get_translate_client():
    return get_boto3_session().client("translate")


# ---- Models for routes ----
class PatientInfo(BaseModel):
    name: Optional[str] = None
    sex: Optional[str] = None
    age: Optional[str] = None
    referring_physician: Optional[str] = None
    additional_context: Optional[str] = None


class FinalNotePayload(BaseModel):
    full_transcript: str
    patient_info: Optional[PatientInfo] = None
    note_type: str = "standard"
    template_id: Optional[str] = None
    encounter_time: Optional[str] = None
    encounter_type: Optional[str] = None


class CommandPayload(BaseModel):
    note_content: Dict[str, Any]
    command: str


# ✅ Added helper for normalization before register_routes
def normalize_note_keys(note, expected_keys):
    """
    Ensure all expected keys are present in the note.
    If missing, set to "None". Remove unexpected keys unless it's error fallback.
    """
    # Accept error fallback if present
    if note == {"error": "insufficient_medical_information"}:
        return note
    fixed = {}
    for k in expected_keys:
        v = note.get(k)
        if v in (None, "", [], {}, "Not documented", "No known drug allergies", "No past medical history discussed"):
            fixed[k] = "None"
        else:
            fixed[k] = v
    # Remove unexpected keys
    extra = set(note.keys()) - set(expected_keys)
    for k in extra:
        pass  # optionally log
    return fixed


def get_expected_keys(payload, template_dict=None):
    if getattr(payload, "template_id", None) and template_dict:
        # Template keys
        sections = template_dict.get("sections", [])
        # if sections were stored as json string, load
        if isinstance(sections, str):
            try:
                sections = json.loads(sections)
            except Exception:
                sections = []
        return [s["name"] for s in sections if isinstance(s, dict) and "name" in s]
    # Built-in
    return BUILTIN_NOTE_KEYS.get(payload.note_type, [])


# ---- PHI Masking Utility ----
def _mask_phi_entities(text: str, entities: List[dict], mask_token: str = "patient") -> str:
    spans = [
        (e["BeginOffset"], e["EndOffset"])
        for e in entities
        if e.get("Category") == "PROTECTED_HEALTH_INFORMATION"
        and e.get("BeginOffset") is not None
        and e.get("EndOffset") is not None
        and e["BeginOffset"] < e["EndOffset"]
    ]
    spans.sort(reverse=True)
    masked = text
    for start, end in spans:
        masked = masked[:start] + mask_token + masked[end:]
    return masked


# The rest of main.py logic remains completely intact below.

# ---- Event stream utilities (AWS Transcribe proxy) ----
def _encode_event_stream(headers: Dict[str, str], payload: bytes) -> bytes:
    headers_payload = b""
    for header_name, header_value in headers.items():
        headers_payload += struct.pack(">B", len(header_name))
        headers_payload += header_name.encode("utf-8")
        headers_payload += struct.pack(">B", 7)
        headers_payload += struct.pack(">H", len(header_value))
        headers_payload += header_value.encode("utf-8")

    headers_len = len(headers_payload)
    total_len = 16 + headers_len + len(payload)

    message = struct.pack(">I", total_len)
    message += struct.pack(">I", headers_len)

    prelude_crc = crc32(message)
    message += struct.pack(">I", prelude_crc)

    message += headers_payload
    message += payload

    message_crc = crc32(message)
    message += struct.pack(">I", message_crc)

    return message


class EventStreamParser:
    def __init__(self) -> None:
        self._buffer = b""

    def _parse_headers(self, header_data: bytes) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        offset = 0
        while offset < len(header_data):
            header_name_len = header_data[offset]
            offset += 1
            header_name = header_data[offset: offset + header_name_len].decode("utf-8")
            offset += header_name_len
            header_value_type = header_data[offset]
            offset += 1
            if header_value_type == 7:
                header_value_len = struct.unpack(">H", header_data[offset: offset + 2])[0]
                offset += 2
                header_value = header_data[offset: offset + header_value_len].decode("utf-8")
                offset += header_value_len
                headers[header_name] = header_value
        return headers

    def parse(self, chunk: bytes):
        self._buffer += chunk
        while len(self._buffer) >= 16:
            try:
                total_len, headers_len = struct.unpack(">II", self._buffer[0:8])
                if len(self._buffer) < total_len:
                    break

                headers_and_payload = self._buffer[12: total_len - 4]
                headers = self._parse_headers(headers_and_payload[:headers_len])
                payload = headers_and_payload[headers_len:]

                event = {"type": headers.get(":event-type"), "headers": headers}
                content_type = headers.get(":content-type")
                if content_type == "application/json":
                    event["payload"] = json.loads(payload.decode("utf-8"))
                else:
                    event["payload"] = payload

                yield event
                self._buffer = self._buffer[total_len:]
            except Exception as exc:
                logger.error("EventStreamParser error: %s. Clearing buffer.", exc)
                self._buffer = b""
                break


# ---- Translation helpers (large-text chunking) ----
TRANSLATE_MAX_BYTES = 9000

def _utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))

def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))

def _split_sentences(text: str) -> List[str]:
    paragraphs = re.split(r"\n{2,}", text.strip())
    out: List[str] = []
    sent_re = re.compile(r"(.+?[\.!?。！？；;])(\s+|$)", re.S)
    for para in paragraphs:
        if _utf8_len(para) <= TRANSLATE_MAX_BYTES:
            out.append(para.strip())
            continue
        idx = 0
        any_match = False
        for m in sent_re.finditer(para):
            any_match = True
            out.append((m.group(1) or "").strip())
            idx = m.end()
        if idx < len(para):
            tail = para[idx:].strip()
            if tail:
                out.append(tail)
        if not any_match and para.strip():
            out.append(para.strip())
    return [s for s in out if s]

def _chunk_text_for_translate(text: str, max_bytes: int = TRANSLATE_MAX_BYTES) -> List[str]:
    sentences = _split_sentences(text)
    chunks: List[str] = []
    buf: List[str] = []
    buf_bytes = 0

    def flush():
        nonlocal buf, buf_bytes
        if buf:
            chunks.append(" ".join(buf))
            buf = []
            buf_bytes = 0

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        s_bytes = _utf8_len(s)
        if s_bytes > max_bytes:
            flush()
            curr: List[str] = []
            curr_bytes = 0
            for ch in s:
                ch_b = _utf8_len(ch)
                if curr_bytes + ch_b > max_bytes:
                    chunks.append("".join(curr))
                    curr = []
                    curr_bytes = 0
                curr.append(ch)
                curr_bytes += ch_b
            if curr:
                chunks.append("".join(curr))
            continue

        if buf_bytes + (1 if buf else 0) + s_bytes <= max_bytes:
            buf.append(s)
            buf_bytes += (1 if buf_bytes else 0) + s_bytes
        else:
            flush()
            buf = [s]
            buf_bytes = s_bytes

    flush()
    return chunks

def _translate_large_text(source_lang: str, target_lang: str, text: str) -> str:
    if not text.strip():
        return text
    if _utf8_len(text) <= TRANSLATE_MAX_BYTES:
        result = get_translate_client().translate_text(
            Text=text, SourceLanguageCode=source_lang, TargetLanguageCode=target_lang
        )
        return result["TranslatedText"]
    pieces = _chunk_text_for_translate(text, TRANSLATE_MAX_BYTES)
    out: List[str] = []
    for piece in pieces:
        res = get_translate_client().translate_text(
            Text=piece, SourceLanguageCode=source_lang, TargetLanguageCode=target_lang
        )
        out.append(res["TranslatedText"])
    return "\n\n".join(out)


# ---- Entity compaction for LLM prompts ----
def _filter_and_compact_entities_for_llm(raw_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    SCORE_MIN = 0.90
    MAX_ENTS = 200
    ALLOWED = {"MEDICAL_CONDITION", "MEDICATION", "TEST_TREATMENT_PROCEDURE", "ANATOMY"}
    PHI_CATEGORY = "PROTECTED_HEALTH_INFORMATION"

    ents: List[Dict[str, Any]] = []
    summary: Dict[str, int] = {}
    phi_counts: Dict[str, int] = {}
    seen = set()

    for e in raw_entities or []:
        cat = str(e.get("Category") or "")
        typ = str(e.get("Type") or "")
        txt = str(e.get("Text") or "").strip()
        score = float(e.get("Score") or 0.0)
        if not txt or score < SCORE_MIN:
            continue
        if cat == PHI_CATEGORY:
            phi_counts[typ] = phi_counts.get(typ, 0) + 1
            continue
        if cat not in ALLOWED:
            continue
        key = (txt.lower(), cat, typ)
        if key in seen:
            continue
        seen.add(key)
        ents.append({"t": txt, "c": cat, "y": typ})
        summary[cat] = summary.get(cat, 0) + 1
        if len(ents) >= MAX_ENTS:
            break
    ents.sort(key=lambda d: (d["c"], d["y"], d["t"]))
    return {"ents": ents, "ents_summary": summary, "phi_counts": phi_counts}

# ---- Bedrock invocation + robust JSON extraction helpers ----
def _find_json_substring(text: str) -> Tuple[str, Optional[str]]:
    """
    Attempt to extract the main JSON substring from model output; returns (substring, error_message)
    """
    if not text:
        return "", "Empty model output"

    start_index = text.find("{")
    if start_index == -1:
        return "", "Could not find JSON object in model output."

    json_substring = text[start_index:].strip()

    # strip fenced codeblock markers if present
    if json_substring.startswith("```"):
        lines = json_substring.split("\n")
        # remove triple-backticks lines
        if lines and lines[-1].strip() == "```":
            json_substring = "\n".join(lines[1:-1])
        else:
            json_substring = "\n".join(lines[1:])

    # trim to matching braces if possible
    brace_level = 0
    last_brace_index = -1
    for idx, ch in enumerate(json_substring):
        if ch == "{":
            brace_level += 1
        elif ch == "}":
            brace_level -= 1
            if brace_level == 0:
                last_brace_index = idx
                break
    if last_brace_index != -1:
        json_substring = json_substring[: last_brace_index + 1]
    elif not json_substring.endswith("}"):
        json_substring += "}"

    return json_substring, None


def _repair_and_parse_json(json_substring: str) -> Dict[str, Any]:
    """
    Try to fix common mistakes and parse JSON. Raises JSONDecodeError or ValueError on failure.
    """
    s = json_substring
    # Quick pattern fixes observed from LLM outputs
    s = re.sub(r'"text:\s*"', '"text": "', s)
    s = re.sub(r'"(\w+):\s*', r'"\1": ', s)
    parsed = json.loads(s)
    return parsed


def _invoke_bedrock_and_parse(system_prompt: str, user_payload: str, max_tokens: int = 4096, temperature: float = 0.1) -> Dict[str, Any]:
    """
    Invokes Bedrock model with composed prompt and returns parsed JSON object.
    Adds helpful logging of the prompt and model output (truncated) for debugging.
    """
    composed_prompt = f"System: {system_prompt}\n\nUser: {user_payload}\n\nAssistant:"
    # Log truncated prompt for debugging (avoid logging full patient PHI in production)
    try:
        safe_prompt = composed_prompt.replace("\n", "\\n")
        logger.debug("COMPOSED PROMPT (truncated 4000 chars): %s", safe_prompt[:4000])
    except Exception:
        logger.debug("COMPOSED PROMPT (could not stringify)")

    body = json.dumps({"prompt": composed_prompt, "max_tokens": max_tokens, "temperature": temperature}, ensure_ascii=False)

    try:
        response = get_bedrock_client().invoke_model(
            body=body,
            modelId=settings.bedrock_model_id,
            accept="application/json",
            contentType="application/json",
        )
    except Exception as exc:
        raise ValueError(f"Bedrock invocation failed: {exc}") from exc

    try:
        response_body = json.loads(response.get("body").read())
        model_output = response_body.get("outputs", [{}])[0].get("text", "")
        stop_reason = response_body.get("outputs", [{}])[0].get("stop_reason")
    except Exception as exc:
        raise ValueError(f"Failed to read Bedrock response: {exc}") from exc

    # Log model output truncated for debugging
    try:
        logger.debug("MODEL OUTPUT (truncated 2000 chars): %s", (model_output or "")[:2000])
    except Exception:
        logger.debug("MODEL OUTPUT (could not stringify)")

    json_substring, find_err = _find_json_substring(model_output)
    if find_err:
        # If we couldn't even find a JSON start, include snippet in error to aid debugging
        raise ValueError(find_err + f" Raw model output (first 1000 chars): {model_output[:1000]}")

    try:
        parsed = _repair_and_parse_json(json_substring)
        logger.info("Successfully parsed JSON from model output with %d top-level keys.", len(parsed))
        return parsed
    except Exception as exc:
        # attempt a repair and include the model_output snippet in the error for debugging
        try:
            repaired = json_substring
            if 'text:' in repaired:
                repaired = re.sub(r'\{"text:\s*"', '{"text": "', repaired)
            repaired = re.sub(r'"([^"]+):\s*(["\[\{])', r'"\1": \2', repaired)
            parsed = json.loads(repaired)
            logger.info("Parsed JSON after secondary repair.")
            return parsed
        except Exception as repair_error:
            # Provide rich debugging context in the raised error
            error_msg = (
                "Failed to parse JSON from final note even after repair. "
                f"Stop Reason: {stop_reason}. Initial parse error: {exc}. Repair error: {repair_error}. "
                f"Full model output (first 2000 chars): '{model_output[:2000]}'"
            )
            raise ValueError(error_msg) from repair_error


def generate_note_from_system_prompt(
    system_prompt: str,
    full_transcript: str,
    comprehend_json: Dict[str, Any],
    patient_info: Optional[Dict[str, Any]] = None,
    encounter_time: Optional[str] = None,
    encounter_type: Optional[str] = None,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    High-level wrapper to prepare user payload and call Bedrock + parsing.
    Adds patient_info and encounter metadata to the user payload so the LLM can
    reference them explicitly (in addition to any patient context embedded in the system prompt).
    """
    user_payload_obj = {
        "full_transcript": full_transcript,
        "ents": comprehend_json.get("ents", []),
        "ents_summary": comprehend_json.get("ents_summary", {}),
        "phi_counts": comprehend_json.get("phi_counts", {}),
    }

    if patient_info:
        user_payload_obj["patient_info"] = patient_info
    if encounter_time:
        user_payload_obj["encounter_time"] = encounter_time
    if encounter_type:
        user_payload_obj["encounter_type"] = encounter_type

    user_payload = json.dumps(user_payload_obj, ensure_ascii=False)
    return _invoke_bedrock_and_parse(system_prompt, user_payload, max_tokens=4096, temperature=temperature)


def _prepare_template_instructions(template_item: dict) -> str:
    """
    Convert the TemplateRead dict into a human-readable but strict instruction block
    that biases the LLM to produce a JSON note that matches the template sections.

    The block contains:
    - A human summary of sections and their descriptions.
    - A required OUTPUT FORMAT section with an explicit JSON skeleton where keys
      exactly match the template section names. The model is told to return ONLY
      the JSON object (no surrounding explanation).
    """
    parts: List[str] = []
    name = template_item.get("name") or "Custom Template"
    parts.append(f"Template: {name}")
    parts.append("Structure (section name: description):")

    sections = template_item.get("sections") or []
    if isinstance(sections, str):
        try:
            sections = json.loads(sections)
        except Exception:
            sections = []

    section_names: List[Dict[str, str]] = []
    for idx, s in enumerate(sections, start=1):
        if not isinstance(s, dict):
            continue
        sec_name = (s.get("name") or s.get("title") or f"Section {idx}").strip()
        sec_desc = (s.get("description") or s.get("desc") or "").strip()
        parts.append(f"{idx}. {sec_name}: {sec_desc}")
        section_names.append({"name": sec_name, "description": sec_desc})

    example_text = template_item.get("example_text") or template_item.get("exampleNoteText") or ""
    if example_text:
        parts.append("\nExample note text (style / tone excerpt):")
        parts.append(example_text.strip()[:2000])

    # Build explicit JSON skeleton using the exact section names.
    skeleton = {}
    for s in section_names:
        skeleton[s["name"]] = []  # default to array-of-objects skeleton for clarity

    parts.append("\nOUTPUT FORMAT (REQUIRED):")
    parts.append(
        "Return a single, valid JSON object and NOTHING ELSE. The object MUST have keys exactly "
        "matching the section names below (names are case-sensitive). For list sections return "
        'an array of objects like [{"text": "..."}]. For narrative sections return a string. '
        "Do NOT include commentary, notes, or explanation — ONLY the JSON."
    )

    try:
        skeleton_example = {k: [{"text": ""}] for k in skeleton.keys()}
        parts.append("Example JSON skeleton (you must follow keys exactly):")
        parts.append(json.dumps(skeleton_example, indent=2, ensure_ascii=False))
    except Exception:
        parts.append("Sections: " + ", ".join(skeleton.keys()))

    parts.append("\nIf a section is truly not applicable, set its value to the string \"None\" (without quotes).")
    parts.append("Do not invent additional top-level keys. Order of keys does not matter, but keys must match.")

    return "\n".join(parts)


# -------- Routes registration (includes WebSocket proxy) --------
def register_routes(app: FastAPI) -> None:
    @app.websocket("/client-transcribe")
    async def client_transcribe(ws: WebSocket) -> None:
        await ws.accept()

        language_code = ws.query_params.get("language_code", "en-US")
        if language_code not in {"en-US", "zh-HK", "zh-TW"}:
            language_code = "en-US"

        logger.info("Browser connected. Selected language=%s", language_code)

        try:
            aws_url = build_presigned_url(selected_language=language_code)
        except RuntimeError as exc:
            logger.error("Unable to build presigned URL: %s", exc)
            await ws.close(code=1011, reason=str(exc))
            return

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(aws_url, max_msg_size=0) as aws_ws:
                    logger.info("Connected to AWS Transcribe.")

                    async def forward_to_aws() -> None:
                        try:
                            while True:
                                data = await ws.receive_bytes()
                                await aws_ws.send_bytes(
                                    _encode_event_stream(
                                        headers={
                                            ":message-type": "event",
                                            ":event-type": "AudioEvent",
                                            ":content-type": "application/octet-stream",
                                        },
                                        payload=data,
                                    )
                                )
                        except WebSocketDisconnect:
                            logger.info("Browser disconnected. Signalling EndOfStream to AWS.")
                        finally:
                            if not aws_ws.closed:
                                try:
                                    await aws_ws.send_bytes(
                                        _encode_event_stream(
                                            headers={
                                                ":message-type": "event",
                                                ":event-type": "AudioEvent",
                                                ":content-type": "application/octet-stream",
                                            },
                                            payload=b"",
                                        )
                                    )
                                    await aws_ws.send_bytes(
                                        _encode_event_stream(
                                            headers={
                                                ":message-type": "event",
                                                ":event-type": "EndOfStream",
                                            },
                                            payload=b"",
                                        )
                                    )
                                except Exception as exc:
                                    logger.warning("Error while sending EndOfStream: %s", exc)

                    async def forward_to_client() -> None:
                        parser = EventStreamParser()
                        async for message in aws_ws:
                            if message.type != aiohttp.WSMsgType.BINARY:
                                continue

                            for event in parser.parse(message.data):
                                event_type = event.get("type")
                                if event_type != "TranscriptEvent":
                                    logger.debug("Received non-transcript event from AWS: %s", event)
                                    continue

                                payload = event["payload"]
                                try:
                                    results = payload.get("Transcript", {}).get("Results", [])
                                    if not results:
                                        continue

                                    result = results[0]
                                    if result.get("IsPartial", True):
                                        if ws.client_state == WebSocketState.CONNECTED:
                                            await ws.send_text(json.dumps(payload))
                                        continue

                                    original_text = result.get("Alternatives", [{}])[0].get("Transcript", "")
                                    if not original_text:
                                        continue

                                    detected_language = result.get("LanguageCode", "en-US")
                                    payload["DisplayText"] = original_text

                                    if detected_language == "en-US":
                                        english_text = original_text
                                    else:
                                        source_lang = detected_language.split("-")[0]
                                        translation = get_translate_client().translate_text(
                                            Text=original_text,
                                            SourceLanguageCode=source_lang,
                                            TargetLanguageCode="en",
                                        )
                                        english_text = translation["TranslatedText"]
                                        payload["TranslatedText"] = english_text

                                    entities = get_comprehend_client().detect_entities_v2(Text=english_text)
                                    payload["ComprehendEntities"] = entities.get("Entities", [])
                                    logger.info(
                                        "Processed final segment (%s). Entities=%d",
                                        detected_language,
                                        len(payload["ComprehendEntities"]),
                                    )
                                except Exception as exc:
                                    logger.error("Real-time processing error: %s", exc)

                                if ws.client_state == WebSocketState.CONNECTED:
                                    await ws.send_text(json.dumps(payload))

                        if aws_ws.close_code != 1000:
                            reason = aws_ws.exception() or f"Code: {aws_ws.close_code}"
                            raise ConnectionAbortedError(
                                f"AWS connection closed unexpectedly. Reason: {reason}"
                            )

                    client_reader = asyncio.create_task(forward_to_aws())
                    aws_reader = asyncio.create_task(forward_to_client())

                    done, pending = await asyncio.wait(
                        [client_reader, aws_reader],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in done:
                        if task.exception():
                            logger.error("Proxy task failed with exception: %s", task.exception())

                    for task in pending:
                        task.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)

        except Exception as exc:
            logger.error("WebSocket proxy failed: %s", exc)
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1011, reason="Internal server error.")
        finally:
            logger.info("Browser session ended.")

    @app.post("/generate-final-note")
    async def generate_final_note(payload: FinalNotePayload) -> Dict[str, Any]:
        full_transcript = (payload.full_transcript or "").strip()
        if not full_transcript:
            raise HTTPException(status_code=400, detail="Received an empty transcript.")

        logger.info(
            "Received transcript (%d chars) for note_type=%s template_id=%s",
            len(full_transcript),
            payload.note_type,
            getattr(payload, "template_id", None),
        )

        patient_info_dict = payload.patient_info.model_dump(exclude_none=True) if payload.patient_info else None

        try:
            # Translate to English if needed (chunked safely)
            if _has_cjk(full_transcript):
                logger.info("Detected CJK characters. Translating to English with chunking.")
                english_transcript = _translate_large_text("zh", "en", full_transcript)
                logger.info("Translation complete. English transcript length=%d.", len(english_transcript))
            else:
                english_transcript = full_transcript

            # Comprehend Medical entity detection (then compact for prompt)
            entities_response = get_comprehend_client().detect_entities_v2(Text=english_transcript)
            entities = entities_response.get("Entities", []) or []
            logger.info("Comprehend Medical returned %d entities.", len(entities))
            ents_compact = _filter_and_compact_entities_for_llm(entities)

            # Guardrail: ensure we have enough content before LLM
            MIN_TRANSCRIPT_LENGTH = 50
            if len(english_transcript.strip()) < MIN_TRANSCRIPT_LENGTH:
                raise HTTPException(
                    status_code=400,
                    detail="Transcript too short—contains insufficient medical detail for note generation."
                )

            medical_ents = [e for e in ents_compact.get("ents", [])
                            if e["c"] in ("MEDICAL_CONDITION", "MEDICATION", "TEST_TREATMENT_PROCEDURE", "ANATOMY")]
            if not medical_ents:
                raise HTTPException(
                    status_code=400,
                    detail="Transcript contains insufficient medical detail for note generation."
                )

            # Mask PHI entities in the transcript before sending to the LLM
            english_transcript_masked = _mask_phi_entities(english_transcript, entities)

            # Prepare encounter metadata EARLY so we can pass it into the prompt generator
            encounter_time = getattr(payload, "encounter_time", None)
            encounter_type = getattr(payload, "encounter_type", None)

            # Determine base system prompt for the requested note type
            try:
                prompt_module = get_prompt_generator(payload.note_type)
                # IMPORTANT: pass encounter_time so it is human-readable via build_patient_context
                base_system_prompt = prompt_module.generate_prompt(patient_info_dict, encounter_time)
            except Exception as e:
                logger.warning("Unknown note type '%s': %s. Falling back to standard prompt.", payload.note_type, e)
                prompt_module = get_prompt_generator("standard")
                base_system_prompt = prompt_module.generate_prompt(patient_info_dict, encounter_time)

            # Optionally augment with template instructions when template_id is provided
            final_system_prompt = base_system_prompt
            temperature = 0.1  # default temperature
            template_dict = None
            if getattr(payload, "template_id", None):
                if not template_service:
                    raise HTTPException(status_code=400, detail="Template support not available on server.")
                try:
                    template_obj = await template_service.get(str(payload.template_id))
                    template_dict = template_obj.model_dump()
                    template_instructions = _prepare_template_instructions(template_dict)

                    # Place template instructions after the base prompt and add a strict header to increase compliance
                    final_system_prompt = (
                        f"{base_system_prompt}\n\n"
                        "TEMPLATE INSTRUCTIONS (FOLLOW THESE EXACTLY):\n"
                        f"{template_instructions}\n\n"
                        "STRICT REQUIREMENT: Return ONLY a single valid JSON object that matches the template keys exactly. Do not include any extra text or explanation."
                    )

                    # Lower temperature for more deterministic, template-compliant output
                    temperature = 0.0

                    logger.info("Using template %s to augment system prompt.", template_dict.get("id"))
                except Exception as exc:
                    logger.exception("Failed to fetch/prepare template: %s", exc)
                    raise HTTPException(status_code=400, detail=f"Could not load template: {exc}") from exc

            # Generate note using composed system prompt and pass patient/encounter metadata
            final_note = generate_note_from_system_prompt(
                final_system_prompt,
                english_transcript_masked,
                ents_compact,
                patient_info=patient_info_dict,
                encounter_time=encounter_time,
                encounter_type=encounter_type,
                temperature=temperature,
            )

            expected_keys = get_expected_keys(payload, template_dict)
            if expected_keys:
                final_note = normalize_note_keys(final_note, expected_keys)

            # Optional normalization for "standard"
            if payload.note_type == "standard":
                final_note = _normalize_assessment_plan(
                    _normalize_empty_sections(_fix_pertinent_negatives(final_note))
                )

            return {"notes": final_note}
        except ValueError as exc:
            logger.error("Note generation error: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected error during final note generation.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/execute-command")
    async def execute_command(payload: CommandPayload) -> Dict[str, str]:
        if not payload.command.strip():
            raise HTTPException(status_code=400, detail="Received an empty command.")
        if not payload.note_content:
            raise HTTPException(status_code=400, detail="Received empty note content.")

        system_prompt = (
            "You are an expert AI assistant for medical professionals. Your task is to take a clinical "
            "note in JSON format and a command, and then execute that command to modify or generate new "
            "text based on the note. You must return ONLY the raw text of the result, without any extra "
            "formatting, explanation, or markdown."
        )

        note_as_string = json.dumps(payload.note_content, indent=2, ensure_ascii=False)
        user_prompt = (
            f"Here is the clinical note:\n```json\n{note_as_string}\n```\n\n"
            f"Here is my command: \"{payload.command}\"\n\n"
            "Please execute this command and return only the resulting text."
        )

        bedrock_body = json.dumps(
            {"prompt": f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:", "max_tokens": 2048, "temperature": 0.2},
            ensure_ascii=False
        )

        try:
            response = get_bedrock_client().invoke_model(
                body=bedrock_body,
                modelId=settings.bedrock_model_id,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            result_text = response_body.get("outputs", [{}])[0].get("text", "").strip()

            if result_text.startswith("```") and result_text.endswith("```"):
                result_text = result_text[3:-3].strip()

            return {"result": result_text}
        except Exception as exc:
            logger.error("Error during command execution: %s", exc)
            raise HTTPException(status_code=500, detail=f"Failed to execute command: {exc}") from exc

    @app.get("/note-types")
    async def get_note_types(user_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return available builtin note types and, when user_id is supplied, the user's custom templates.

        Template entries are returned with:
          - id: "template:<uuid>"
          - name: template name
          - description: "Custom template"
          - source: "template"
          - template_id: the UUID string

        Builtin prompt modules are returned with:
          - id: key from PROMPT_REGISTRY (e.g., "standard")
          - name: module.NOTE_TYPE
          - description: "Generate a <NOTE_TYPE>"
          - source: "builtin"
        """
        response: List[Dict[str, Any]] = []

        # builtin prompt types
        for key, module in PROMPT_REGISTRY.items():
            response.append(
                {
                    "id": key,
                    "name": module.NOTE_TYPE,
                    "description": f"Generate a {module.NOTE_TYPE}",
                    "source": "builtin",
                }
            )

        # optionally include user templates
        if user_id and template_service is not None:
            try:
                templates = await template_service.list_for_user(str(user_id))
                for tmpl in templates:
                    tdict = tmpl.model_dump()
                    tname = tdict.get("name") or f"Template {tdict.get('id')}"
                    response.append(
                        {
                            "id": f"template:{tdict.get('id')}",
                            "name": tname,
                            "description": "Custom template",
                            "source": "template",
                            "template_id": str(tdict.get("id")),
                        }
                    )
            except Exception as exc:
                logger.warning("Could not load templates for user %s: %s", user_id, exc)

        return {"note_types": response}


# ---- helper utilities for presigned transcribe URL ----
def build_presigned_url(selected_language: str = "en-US") -> str:
    if not settings.aws_access_key_id or not settings.aws_secret_access_key:
        raise RuntimeError(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required to build the Transcribe presigned URL."
        )

    if selected_language in ("zh-HK", "zh-TW"):
        language_options = f"{selected_language},en-US"
    else:
        language_options = "en-US,zh-HK"

    method = "GET"
    service = "transcribe"
    host = f"transcribestreaming.{settings.aws_region}.amazonaws.com:8443"
    endpoint = f"wss://{host}/stream-transcription-websocket"

    timestamp = datetime.now(timezone.utc)
    amz_date = timestamp.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = timestamp.strftime("%Y%m%d")
    canonical_uri = "/stream-transcription-websocket"

    query_params = {
        "identify-multiple-languages": "true",
        "language-options": language_options,
        "preferred-language": language_options.split(",", 1)[0],
        "show-speaker-label": "true",
        "media-encoding": "pcm",
        "sample-rate": "16000",
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{settings.aws_access_key_id}/{date_stamp}/{settings.aws_region}/{service}/aws4_request",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": "300",
        "X-Amz-SignedHeaders": "host",
    }

    if settings.aws_session_token:
        query_params["X-Amz-Security-Token"] = settings.aws_session_token

    canonical_querystring = "&".join(
        f"{key}={quote(str(value), safe='~')}" for key, value in sorted(query_params.items())
    )
    canonical_headers = f"host:{host}\n"
    signed_headers = "host"
    payload_hash = hashlib.sha256(b"").hexdigest()

    canonical_request = (
        f"{method}\n"
        f"{canonical_uri}\n"
        f"{canonical_querystring}\n"
        f"{canonical_headers}\n"
        f"{signed_headers}\n"
        f"{payload_hash}"
    )

    credential_scope = f"{date_stamp}/{settings.aws_region}/{service}/aws4_request"
    string_to_sign = (
        "AWS4-HMAC-SHA256\n"
        f"{amz_date}\n"
        f"{credential_scope}\n"
        f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
    )

    signing_key = _get_signature_key(settings.aws_secret_access_key, date_stamp, settings.aws_region, service)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    return f"{endpoint}?{canonical_querystring}&X-Amz-Signature={signature}"


def _sign(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signature_key(key, date_stamp, region_name, service_name):
    k_date = _sign(("AWS4" + key).encode("utf-8"), date_stamp)
    k_region = _sign(k_date, region_name)
    k_service = _sign(k_region, service_name)
    k_signing = _sign(k_service, "aws4_request")
    return k_signing


# ---- small normalization helpers used after LLM output ----
def _normalize_assessment_plan(note: Dict[str, Any]) -> Dict[str, Any]:
    ap_section = note.get("Assessment and Plan")
    if isinstance(ap_section, str):
        note["Assessment and Plan"] = ap_section.replace("\\n", "\n").strip()
        logger.info("Normalized 'Assessment and Plan' section.")
    return note


def _fix_pertinent_negatives(note: Dict[str, Any]) -> Dict[str, Any]:
    pn_section = note.get("Pertinent Negatives")
    if isinstance(pn_section, str):
        logger.warning("Converting string-based 'Pertinent Negatives' into structured list.")
        cleaned = pn_section.lower().replace("patient denies", "").strip()
        negatives = [entry.strip().rstrip(".") for entry in cleaned.split(",") if entry.strip()]
        note["Pertinent Negatives"] = [{"text": negative.capitalize()} for negative in negatives]
    return note


def _normalize_empty_sections(note: Dict[str, Any]) -> Dict[str, Any]:
    sections_to_check = ["Pertinent Negatives", "Past Medical History", "Medications"]
    empty_markers = {"no pertinent negatives", "no past medical history", "not discussed", "none mentioned", "none"}

    for section in sections_to_check:
        items = note.get(section)
        value = ""

        if isinstance(items, str):
            value = items.strip().lower()
        elif isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
            value = items[0].get("text", "").strip().lstrip("-").lower()

        if value in empty_markers:
            if note.get(section) != "None":
                logger.info("Normalizing verbose empty section '%s' to 'None'.", section)
                note[section] = "None"

    return note


# ---- App creation ----
def create_app() -> FastAPI:
    app = FastAPI(title="Stethoscribe Proxy", version="2.0.0")

    app.add_middleware(GZipMiddleware, minimum_size=512)

    allowed = settings.allowed_origins or ["*"]
    allow_credentials = False if "*" in allowed else True
    logger.info("CORS allow_origins: %s | allow_credentials=%s", allowed, allow_credentials)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=86400,
    )

    # Avoid automatic trailing-slash redirect responses which can break CORS preflight flows
    app.router.redirect_slashes = False

    # include API router and register additional bespoke routes
    app.include_router(api_router)
    register_routes(app)
    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)