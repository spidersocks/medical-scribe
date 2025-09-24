# main.py

import os
import hmac
import hashlib
import logging
import base64
import asyncio
import struct
import json
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from urllib.parse import quote
from zlib import crc32
import re
import uvicorn
import aiohttp
import boto3
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from starlette.websockets import WebSocketState

# config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scribe-proxy")
load_dotenv()

# access variables securely stored in ENV variables
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

ALLOWED_ORIGINS_STR = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [o.strip() for o in ALLOWED_ORIGINS_STR.split(",") if o.strip()]

app = FastAPI(title="Stethoscribe Proxy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=86400,  # cache preflight 1 day
)

comprehend_medical = boto3.client("comprehendmedical", region_name=AWS_REGION)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
translate = boto3.client("translate", region_name=AWS_REGION)

class FinalNotePayload(BaseModel):
    full_transcript: str

class CommandPayload(BaseModel):
    note_content: Dict[str, Any]
    command: str

# AWS event stream encoding

def _encode_event_stream(headers: dict, payload: bytes) -> bytes:
    headers_payload = b''
    for header_name, header_value in headers.items():
        headers_payload += struct.pack('>B', len(header_name))
        headers_payload += header_name.encode('utf-8')
        headers_payload += struct.pack('>B', 7)
        headers_payload += struct.pack('>H', len(header_value))
        headers_payload += header_value.encode('utf-8')

    headers_len = len(headers_payload)
    total_len = 16 + headers_len + len(payload)

    message = struct.pack('>I', total_len)
    message += struct.pack('>I', headers_len)
    
    prelude_crc = crc32(message)
    message += struct.pack('>I', prelude_crc)

    message += headers_payload
    message += payload
    
    message_crc = crc32(message)
    message += struct.pack('>I', message_crc)
    
    return message


# AWS event stream parser
class EventStreamParser:
    def __init__(self):
        self._buffer = b''
    def _parse_headers(self, header_data):
        headers = {}
        offset = 0
        while offset < len(header_data):
            header_name_len = header_data[offset]
            offset += 1
            header_name = header_data[offset:offset + header_name_len].decode('utf-8')
            offset += header_name_len
            header_value_type = header_data[offset]
            offset += 1
            if header_value_type == 7:
                header_value_len = struct.unpack('>H', header_data[offset:offset+2])[0]
                offset += 2
                header_value = header_data[offset:offset + header_value_len].decode('utf-8')
                offset += header_value_len
                headers[header_name] = header_value
        return headers
    def parse(self, chunk: bytes):
        self._buffer += chunk
        while len(self._buffer) >= 16:
            try:
                total_len, headers_len = struct.unpack('>II', self._buffer[0:8])
                if len(self._buffer) < total_len: break
                headers_and_payload = self._buffer[12 : total_len - 4]
                headers = self._parse_headers(headers_and_payload[:headers_len])
                payload = headers_and_payload[headers_len:]
                event_type = headers.get(':event-type')
                content_type = headers.get(':content-type')
                event = {"type": event_type, "headers": headers}
                if content_type == 'application/json':
                    event["payload"] = json.loads(payload.decode('utf-8'))
                else:
                    event["payload"] = payload
                yield event
                self._buffer = self._buffer[total_len:]
            except Exception as e:
                logger.error(f"EventStreamParser error: {e}. Clearing buffer.")
                self._buffer = b''
                break

# helper functions
def sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()
def get_signature_key(secret_key: str, date_stamp: str, region_name: str, service_name: str) -> bytes:
    k_date = sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = hmac.new(k_date, region_name.encode("utf-8"), hashlib.sha256).digest()
    k_service = hmac.new(k_region, service_name.encode("utf-8"), hashlib.sha256).digest()
    return hmac.new(k_service, b"aws4_request", hashlib.sha256).digest()

def build_presigned_url(selected_language: str = "en-US") -> str:
    # Enforcing only one Chinese language option when English is primary (AWS doesn't support dialecte detection yet)
    if selected_language in ("zh-HK", "zh-TW"):
        # Chinese primary, English backup
        language_options = f"{selected_language},en-US"
    else:
        # English primary, Cantonese backup
        language_options = "en-US,zh-HK"

    method = "GET"
    service = "transcribe"
    host = f"transcribestreaming.{AWS_REGION}.amazonaws.com:8443"
    endpoint = f"wss://{host}/stream-transcription-websocket"

    t = datetime.now(timezone.utc)
    amz_date = t.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = t.strftime("%Y%m%d")
    canonical_uri = "/stream-transcription-websocket"

    query_params = {
        "identify-multiple-languages": "true",
        "language-options": language_options,
        # preferred-language defaults to English
        "preferred-language": language_options.split(",", 1)[0],
        "show-speaker-label":"true",
        "media-encoding": "pcm",
        "sample-rate": "16000",
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{AWS_ACCESS_KEY}/{date_stamp}/{AWS_REGION}/{service}/aws4_request",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": "300",
        "X-Amz-SignedHeaders": "host",
    }
    if AWS_SESSION_TOKEN:
        query_params["X-Amz-Security-Token"] = AWS_SESSION_TOKEN 

    canonical_querystring = "&".join(
        f"{k}={quote(str(v), safe='~')}" for k, v in sorted(query_params.items())
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
    credential_scope = f"{date_stamp}/{AWS_REGION}/{service}/aws4_request"
    string_to_sign = (
        "AWS4-HMAC-SHA256\n"
        f"{amz_date}\n"
        f"{credential_scope}\n"
        f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
    )
    signing_key = get_signature_key(AWS_SECRET_KEY, date_stamp, AWS_REGION, service)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    return f"{endpoint}?{canonical_querystring}&X-Amz-Signature={signature}"

# websocket code
@app.websocket("/client-transcribe")
async def client_transcribe(ws: WebSocket):
    await ws.accept()

    # read language_code from query (?language_code=en-US|zh-HK|zh-TW); default en-US
    try:
        q = dict(ws.query_params)
    except Exception:
        q = {}
    language_code = q.get("language_code", "en-US")
    if language_code not in ("en-US", "zh-HK", "zh-TW"):
        language_code = "en-US"

    # log the primary+backup composition
    if language_code in ("zh-HK", "zh-TW"):
        language_options_log = f"{language_code},en-US"
    else:
        language_options_log = "en-US,zh-HK"

    logger.info(f"📡 Browser connected. Dropdown selected={language_code}. Using language-options={language_options_log}")

    aws_url = build_presigned_url(selected_language=language_code)
    logger.info("Connecting to AWS Transcribe with computed language-options...")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(aws_url, max_msg_size=0) as aws_ws:
                logger.info("✅ Connected to AWS Transcribe")

                async def forward_to_aws():
                    saw_audio = False
                    try:
                        while True:
                            data = await ws.receive_bytes()
                            if data:
                                saw_audio = True
                                logger.debug(f"Forwarding audio chunk: {len(data)} bytes")
                            else:
                                logger.warning("Received empty audio chunk from client.")
                            await aws_ws.send_bytes(_encode_event_stream(
                                headers={
                                    ":message-type": "event",
                                    ":event-type": "AudioEvent",
                                    ":content-type": "application/octet-stream"
                                },
                                payload=data
                            ))
                    except WebSocketDisconnect:
                        logger.info("Browser disconnected. Sending end-of-stream to AWS.")
                        if not aws_ws.closed:
                            try:
                                await aws_ws.send_bytes(_encode_event_stream(
                                    headers={
                                        ":message-type": "event",
                                        ":event-type": "AudioEvent",
                                        ":content-type": "application/octet-stream"
                                    },
                                    payload=b''  # flush any pending partials
                                ))
                                await aws_ws.send_bytes(_encode_event_stream(
                                    headers={
                                        ":message-type": "event",
                                        ":event-type": "EndOfStream"
                                    },
                                    payload=b''
                                ))
                            except Exception as e:
                                logger.warning(f"Error while sending EndOfStream: {e}")

                async def forward_to_client():
                    parser = EventStreamParser()
                    async for msg in aws_ws:
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            for event in parser.parse(msg.data):
                                etype = event.get("type")
                                if etype != "TranscriptEvent":
                                    logger.error(f"AWS non-transcript event: type={etype} headers={event.get('headers')} payload={event.get('payload')}")
                                    continue

                                payload = event['payload']
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
                                        # translate from detected Chinese dialect
                                        source_lang = detected_language.split('-')[0]
                                        translation_response_en = translate.translate_text(
                                            Text=original_text,
                                            SourceLanguageCode=source_lang,
                                            TargetLanguageCode='en'
                                        )
                                        english_text = translation_response_en['TranslatedText']
                                        payload["TranslatedText"] = english_text

                                    comprehend_result = comprehend_medical.detect_entities_v2(Text=english_text)
                                    payload["ComprehendEntities"] = comprehend_result.get("Entities", [])
                                    logger.info(f"Processed final segment in '{detected_language}', entities={len(payload['ComprehendEntities'])}")

                                except Exception as e:
                                    logger.error(f"Real-time processing error: {e}")

                                if ws.client_state == WebSocketState.CONNECTED:
                                    await ws.send_text(json.dumps(payload))

                    if aws_ws.close_code != 1000:
                        reason = aws_ws.exception() or f"Code: {aws_ws.close_code}"
                        raise ConnectionAbortedError(f"AWS connection closed unexpectedly. Reason: {reason}")

                client_reader_task = asyncio.create_task(forward_to_aws())
                aws_reader_task = asyncio.create_task(forward_to_client())

                done, pending = await asyncio.wait(
                    [client_reader_task, aws_reader_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    if task.exception():
                        logger.error(f"A proxy task failed with an exception: {task.exception()}")

                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

    except Exception as e:
        logger.error(f"The WebSocket proxy failed. Exception: {e}")
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close(code=1011, reason="An internal error occurred.")
    finally:
        logger.info("📴 Browser session ended")

def _normalize_assessment_plan(note: Dict[str, Any]) -> Dict[str, Any]:
    ap_section = note.get("Assessment and Plan")
    if isinstance(ap_section, str):
        note["Assessment and Plan"] = ap_section.replace('\\n', '\n').strip()
        logger.info("Passing through Assessment and Plan with minimal normalization.")
    return note

# clinical note prompting
def generate_note_from_scratch(comprehend_json: dict, full_transcript: str):
    system_prompt = """You are an AI medical scribe with the reasoning ability of a senior clinician. Your task is to transform a conversation transcript into a perfectly structured, factually accurate clinical note in JSON format. Your response must be a single, valid JSON object and nothing else.

**CORE PRINCIPIPLES:**
1.  **DO NOT COPY FROM THE EXAMPLES:** The examples are for structure and style ONLY. Every single fact in your output (diagnoses, medications, history) MUST come from the user's transcript. Hallucinating information from the examples is a critical failure and must be avoided.
2.  **NARRATIVE STYLE:** If age and gender are known, begin the HPI with "The patient is a [age]-year-old [gender]...". If unknown, use "The patient..." and omit age/gender.
3.  **Fact-Check for Consistency:** After generating the note, re-read the entire transcript and your note. The summary in the Assessment MUST be consistent with the HPI. A symptom confirmed as positive in the HPI cannot be denied in the Pertinent Negatives.
4.  **STRICT FORMATTING FOR NEGATIVES:** The "Pertinent Negatives" section is for denied symptoms ONLY. The value for this key MUST BE a JSON array of objects, each with a "text" key. FOR EXAMPLE: `[{"text": "Patient denies fevers"}, {"text": "Patient denies chills"}]`. IT MUST NOT BE A SINGLE STRING.
5.  **Handle History and Medications Correctly:**
    -   **Past Medical History:** List all medical conditions mentioned in the transcript, even if they seem minor (e.g., 'back pain').
    -   **Medications:** This section is for pre-existing home medications. Include dosage and frequency if mentioned (e.g., "Ibuprofen 3-4 times a week"). If none, state "None".
6.  **Be Comprehensive & CONSOLIDATED:** The note must be complete. For the Assessment and Plan, group related items (diagnostics, treatments, counseling) under a single diagnosis or problem number whenever possible, as shown in the examples.

**"GOLD STANDARD" EXAMPLE 1: ADULT PRIMARY CARE**
```json
{
  "Chief Complaint": [{"text": "Fatigue"}],
  "History of Present Illness": "The patient is a 45-year-old female with a two-month history of progressive physical exhaustion. She feels drained upon waking and experiences worsening fatigue in the afternoons. This is associated with new-onset headaches and shortness of breath on exertion.",
  "Pertinent Negatives": [{"text": "Patient denies dizziness, weight changes, changes in appetite, and fevers."}],
  "Past Medical History": [{"text": "Hypertension"}, {"text": "Family history of anemia"}],
  "Medications": [{"text": "Lisinopril 10mg daily"}, {"text": "Amlodipine 5mg daily"}],
  "Assessment and Plan": "1. Fatigue and Shortness of Breath: The constellation of symptoms is concerning for iron deficiency anemia. Plan is to order blood work including a CBC and iron studies.\\n2. Hypertension Management: Patient's home medications seem effective; they should continue their current regimen.\\n3. Patient Instructions: Advised to schedule a follow-up to review lab results in one week.\\n4. Red Flags: Instructed to seek urgent care for any new chest pain, severe shortness of breath, or fainting episodes."
}
```

**"GOLD STANDARD" EXAMPLE 2: PEDIATRIC ACUTE VISIT**
```json
{
  "Chief Complaint": [{"text": "Fever and Rash"}],
  "History of Present Illness": "The patient is a child with a two-day history of low-grade fever and a red, blanching rash that appeared this morning on her chest and back. The mother notes the child has also had a mild runny nose and has been fussier than usual. The child remains active and is tolerating oral intake.",
  "Pertinent Negatives": [{"text": "Patient's mother denies any cough, sore throat, vomiting, or diarrhea."}],
  "Past Medical History": [{"text": "Vaccinations are up to date."}],
  "Medications": [{"text": "None"}],
  "Assessment and Plan": "1. Viral Exanthem: The combination of low-grade fever, blanching rash, and mild URI symptoms in a well-appearing, vaccinated child is most consistent with a viral exanthem, such as roseola. The plan is to check vital signs, ensure good oxygen levels, and look in the throat and ears.\\n2. Patient Instructions: Advised to manage with supportive care including fluids, rest, and Tylenol as needed for fever.\\n3. Red Flags: If the rash turns purple or does not blanch, or if the patient develops a stiff neck, difficulty breathing, persistent high fever, or seems very lethargic, she should be brought in immediately or go to the ER."
}
```

**YOUR TASK:**
Now, using all the principles and the two "Gold Standard" examples above, generate a complete and accurate JSON note from the following transcript. For the Chief Complaint, use a concise clinical term that best represents the patient's main issue (e.g., "Abdominal Pain" instead of "stomach problem").
"""

    user_prompt = json.dumps({ "full_transcript": full_transcript, "comprehend_output": comprehend_json })
    prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
    body = json.dumps({"prompt": prompt, "max_tokens": 4096, "temperature": 0.1})

    try:
        response = bedrock_runtime.invoke_model(body=body, modelId="mistral.mistral-large-2402-v1:0", accept="application/json", contentType="application/json")
        response_body = json.loads(response.get('body').read())
        note_json_string = response_body.get('outputs')[0].get('text')
        stop_reason = response_body.get('outputs')[0].get('stop_reason')

        try:
            start_index = note_json_string.find('{')
            if start_index == -1: raise ValueError("Could not find a starting '{' in the model's response.")
            json_substring = note_json_string[start_index:].strip()
            brace_level = 0
            last_brace_index = -1
            for i, char in enumerate(json_substring):
                if char == '{': brace_level += 1
                elif char == '}':
                    brace_level -= 1
                    if brace_level == 0: last_brace_index = i
            
            if last_brace_index != -1: json_substring = json_substring[:last_brace_index + 1]
            else:
                if not json_substring.endswith('}'): json_substring += '}'

            parsed_json = json.loads(json_substring)
            return parsed_json
            
        except (ValueError, json.JSONDecodeError) as e:
            error_message = f"Failed to parse JSON from final note. Stop Reason: {stop_reason}. Details: {e}. Full Generation: '{note_json_string}'"
            raise ValueError(error_message)

    except Exception as e:
        if isinstance(e, ValueError): raise e
        raise ValueError(f"An unexpected error occurred during Bedrock invocation. Details: {str(e)}")

def _fix_pertinent_negatives(note: Dict[str, Any]) -> Dict[str, Any]:
    pn_section = note.get("Pertinent Negatives")
    if isinstance(pn_section, str):
        logger.warning("Pertinent Negatives was a string. Converting to correct format.")
        cleaned_string = pn_section.lower().replace("patient denies", "").strip()
        negatives = [s.strip().rstrip('.') for s in cleaned_string.split(',') if s.strip()]
        note["Pertinent Negatives"] = [{"text": neg.capitalize()} for neg in negatives]
    return note

def _normalize_empty_sections(note: Dict[str, Any]) -> Dict[str, Any]:
    sections_to_check = ["Pertinent Negatives", "Past Medical History", "Medications"]
    empty_phrases = ["no pertinent negatives", "no past medical history", "not discussed", "none mentioned"]

    for section in sections_to_check:
        items = note.get(section)
        text_to_check = ""
        if isinstance(items, str): text_to_check = items.lower()
        elif isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
            text_to_check = items[0].get("text", "").lower().strip().lstrip('-').strip()

        is_verbose_empty = any(phrase in text_to_check for phrase in empty_phrases)
        is_simple_none = "none" == text_to_check

        if is_verbose_empty or is_simple_none:
            if note.get(section) != "None":
                logger.info(f"Normalizing empty/verbose section '{section}' to 'None'.")
                note[section] = "None"
    return note


@app.post("/generate-final-note")
async def generate_final_note(payload: FinalNotePayload):
    logger.info(f"Received full transcript for final note generation. Size: {len(payload.full_transcript)} chars.")
    if not payload.full_transcript.strip():
        raise HTTPException(status_code=400, detail="Received an empty transcript.")
    
    try:
        # Check if the transcript has Chinese characters
        # IF so, translate
        if re.search(r'[\u4e00-\u9fff]', payload.full_transcript):
            logger.info("Chinese transcript detected. Translating to English for processing...")
            translation_response = translate.translate_text(
                Text=payload.full_transcript,
                SourceLanguageCode='zh',
                TargetLanguageCode='en'
            )
            english_transcript = translation_response['TranslatedText']
            logger.info(f"Translation complete. English transcript size: {len(english_transcript)} chars.")
        else:
            english_transcript = payload.full_transcript

        # rst of the logic runs with assumption of a full English transcript
        logger.info("Running AWS Comprehend Medical on full English transcript...")
        result = comprehend_medical.detect_entities_v2(Text=english_transcript)
        all_entities = result.get('Entities', [])
        logger.info(f"Comprehend found {len(all_entities)} total entities.")
        
        simplified_entities = [
            {"Text": e["Text"], "Category": e["Category"], "Type": e["Type"]}
            for e in all_entities
        ]
        
        # recieving English transcript to generate note
        final_note = generate_note_from_scratch(simplified_entities, english_transcript)

        # post-processing functions
        structured_note = _fix_pertinent_negatives(final_note)
        normalized_note = _normalize_empty_sections(structured_note)
        formatted_note = _normalize_assessment_plan(normalized_note)

        return {"notes": formatted_note}
    except Exception as e:
        logger.error(f"Error during final note generation endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# new execute command endpoint
@app.post("/execute-command")
async def execute_command(payload: CommandPayload):
    if not payload.command.strip():
        raise HTTPException(status_code=400, detail="Received an empty command.")
    if not payload.note_content:
        raise HTTPException(status_code=400, detail="Received empty note content.")

    system_prompt = """You are an expert AI assistant for medical professionals. Your task is to take a clinical note in JSON format and a command, and then execute that command to modify or generate new text based on the note. You must return ONLY the raw text of the result, without any extra formatting, explanation, or markdown.

For example, if the command is "Write a referral letter to a cardiologist", you should output only the text of the letter itself. If the command is "Summarize the assessment and plan", you should output only the summary text.
"""
    
    note_as_string = json.dumps(payload.note_content, indent=2)
    user_prompt = f"Here is the clinical note:\n```json\n{note_as_string}\n```\n\nHere is my command: \"{payload.command}\"\n\nPlease execute this command and return only the resulting text."

    prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
    
    body = json.dumps({"prompt": prompt, "max_tokens": 2048, "temperature": 0.2})
    
    try:
        response = bedrock_runtime.invoke_model(body=body, modelId="mistral.mistral-large-2402-v1:0", accept="application/json", contentType="application/json")
        response_body = json.loads(response.get('body').read())
        result_text = response_body.get('outputs')[0].get('text').strip()
        
        if result_text.startswith("```") and result_text.endswith("```"):
            result_text = result_text[3:-3].strip()

        return {"result": result_text}
    except Exception as e:
        logger.error(f"Error during command execution: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute command: {e}")


# on run...
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)