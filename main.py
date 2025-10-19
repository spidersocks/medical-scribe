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
from typing import Any, Dict, List, Optional
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

from prompts import PROMPT_REGISTRY, get_prompt_generator

try:
    from api import api_router
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "api_router could not be imported. Ensure your api package exposes `api_router`."
    ) from exc


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stethoscribe-main")

load_dotenv()


class Settings(BaseModel):
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    allowed_origins: List[str] = []
    bedrock_model_id: str = "mistral.mistral-large-2402-v1:0"

    @classmethod
    def from_env(cls) -> "Settings":
        raw_origins = os.getenv("ALLOWED_ORIGINS", "")
        allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
        return cls(
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            allowed_origins=allowed_origins,
            bedrock_model_id=os.getenv("BEDROCK_MODEL_ID", "mistral.mistral-large-2402-v1:0"),
        )


settings = Settings.from_env()


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


class CommandPayload(BaseModel):
    note_content: Dict[str, Any]
    command: str


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


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signature_key(secret_key: str, date_stamp: str, region_name: str, service_name: str) -> bytes:
    k_date = _sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = hmac.new(k_date, region_name.encode("utf-8"), hashlib.sha256).digest()
    k_service = hmac.new(k_region, service_name.encode("utf-8"), hashlib.sha256).digest()
    return hmac.new(k_service, b"aws4_request", hashlib.sha256).digest()


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


def register_routes(app: FastAPI) -> None:
    @app.websocket("/client-transcribe")
    async def client_transcribe(ws: WebSocket) -> None:
        await ws.accept()

        language_code = ws.query_params.get("language_code", "en-US")
        if language_code not in {"en-US", "zh-HK", "zh-TW"}:
            language_code = "en-US"

        if language_code in {"zh-HK", "zh-TW"}:
            language_options_log = f"{language_code},en-US"
        else:
            language_options_log = "en-US,zh-HK"

        logger.info(
            "Browser connected. Selected language=%s. Using language-options=%s",
            language_code,
            language_options_log,
        )

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
                                    logger.debug(
                                        "Received non-transcript event from AWS: %s", event
                                    )
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

                                    original_text = (
                                        result.get("Alternatives", [{}])[0].get("Transcript", "")
                                    )
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
        full_transcript = payload.full_transcript.strip()
        if not full_transcript:
            raise HTTPException(status_code=400, detail="Received an empty transcript.")

        logger.info(
            "Received transcript (%d chars) for note_type=%s",
            len(full_transcript),
            payload.note_type,
        )

        patient_info_dict = (
            payload.patient_info.model_dump(exclude_none=True) if payload.patient_info else None
        )

        try:
            if re.search(r"[\u4e00-\u9fff]", full_transcript):
                logger.info("Detected Chinese characters. Translating to English.")
                translation = get_translate_client().translate_text(
                    Text=full_transcript,
                    SourceLanguageCode="zh",
                    TargetLanguageCode="en",
                )
                english_transcript = translation["TranslatedText"]
                logger.info("Translation complete. English transcript length=%d.", len(english_transcript))
            else:
                english_transcript = full_transcript

            entities_response = get_comprehend_client().detect_entities_v2(Text=english_transcript)
            entities = entities_response.get("Entities", [])
            logger.info("Comprehend Medical returned %d entities.", len(entities))

            simplified_entities = [
                {"Text": entity["Text"], "Category": entity["Category"], "Type": entity["Type"]}
                for entity in entities
            ]

            final_note = generate_note_from_scratch(
                comprehend_json=simplified_entities,
                full_transcript=english_transcript,
                patient_info=patient_info_dict,
                note_type=payload.note_type,
            )

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

        note_as_string = json.dumps(payload.note_content, indent=2)
        user_prompt = (
            f"Here is the clinical note:\n```json\n{note_as_string}\n```\n\n"
            f"Here is my command: \"{payload.command}\"\n\n"
            "Please execute this command and return only the resulting text."
        )

        bedrock_body = json.dumps(
            {"prompt": f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:", "max_tokens": 2048, "temperature": 0.2}
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
    async def get_note_types() -> Dict[str, List[Dict[str, str]]]:
        response = []
        for key, module in PROMPT_REGISTRY.items():
            response.append(
                {
                    "id": key,
                    "name": module.NOTE_TYPE,
                    "description": f"Generate a {module.NOTE_TYPE}",
                }
            )
        return {"note_types": response}


def create_app() -> FastAPI:
    app = FastAPI(title="Stethoscribe Proxy", version="2.0.0")

    # CORS: be explicit and robust
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

    app.include_router(api_router)
    register_routes(app)
    return app


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


def _insert_missing_commas(raw_json: str) -> str:
    lines = raw_json.splitlines()
    fixed_lines: List[str] = []

    for idx, line in enumerate(lines):
        stripped_line = line.rstrip()
        trimmed = stripped_line.strip()

        if trimmed.startswith('"') and ":" in trimmed and not trimmed.endswith(","):
            next_trimmed = ""
            for j in range(idx + 1, len(lines)):
                next_trimmed = lines[j].strip()
                if next_trimmed:
                    break
            if next_trimmed and not next_trimmed.startswith(("}", "]")):
                stripped_line += ","
        fixed_lines.append(stripped_line)

    return "\n".join(fixed_lines)


def generate_note_from_scratch(
    comprehend_json: Dict[str, Any],
    full_transcript: str,
    patient_info: Optional[Dict[str, Any]],
    note_type: str,
) -> Dict[str, Any]:
    try:
        prompt_module = get_prompt_generator(note_type)
        logger.info("Generating %s note via modular prompt system.", prompt_module.NOTE_TYPE)
        system_prompt = prompt_module.generate_prompt(patient_info)
    except ValueError as exc:
        logger.error("Invalid note type requested: %s", note_type)
        raise ValueError(f"Invalid note type: {note_type}") from exc

    user_payload = json.dumps({"full_transcript": full_transcript, "comprehend_output": comprehend_json})
    composed_prompt = f"System: {system_prompt}\n\nUser: {user_payload}\n\nAssistant:"
    body = json.dumps({"prompt": composed_prompt, "max_tokens": 4096, "temperature": 0.1})

    try:
        response = get_bedrock_client().invoke_model(
            body=body,
            modelId=settings.bedrock_model_id,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
        model_output = response_body.get("outputs", [{}])[0].get("text", "")
        stop_reason = response_body.get("outputs", [{}])[0].get("stop_reason")

        try:
            start_index = model_output.find("{")
            if start_index == -1:
                raise ValueError("Could not find JSON object in Bedrock response.")

            json_substring = model_output[start_index:].strip()

            if json_substring.startswith("```"):
                lines = json_substring.split("\n")
                json_substring = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            brace_level = 0
            last_brace_index = -1
            for idx, char in enumerate(json_substring):
                if char == "{":
                    brace_level += 1
                elif char == "}":
                    brace_level -= 1
                    if brace_level == 0:
                        last_brace_index = idx
            if last_brace_index != -1:
                json_substring = json_substring[: last_brace_index + 1]
            elif not json_substring.endswith("}"):
                json_substring += "}"

            json_substring = re.sub(r'"text:\s*"', '"text": "', json_substring)
            json_substring = re.sub(r'"(\w+):\s*', r'"\1": ', json_substring)

            parsed = json.loads(json_substring)
            logger.info("Successfully generated note with %d top-level sections.", len(parsed))
            return parsed
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Initial JSON parse failed: %s. Attempting repair.", exc)
            try:
                repaired = _insert_missing_commas(json_substring)
                repaired = re.sub(r'\{"text:\s*"', '{"text": "', repaired)
                repaired = re.sub(r'"([^"]+):\s*(["\[\{])', r'"\1": \2', repaired)
                parsed = json.loads(repaired)
                logger.info("Successfully parsed JSON after repair step.")
                return parsed
            except Exception as repair_error:
                error_msg = (
                    "Failed to parse JSON from final note even after repair. "
                    f"Stop Reason: {stop_reason}. Original error: {exc}. "
                    f"Repair error: {repair_error}. Full Generation: '{model_output}'"
                )
                raise ValueError(error_msg) from repair_error
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Bedrock invocation failed: {exc}") from exc


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)