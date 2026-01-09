from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from http import HTTPStatus

import boto3
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.websockets import WebSocketState
from starlette.middleware.gzip import GZipMiddleware

import dashscope
from dashscope.api_entities.dashscope_response import Role

# Application-specific imports
from prompts import PROMPT_REGISTRY, get_prompt_generator
from services import template_service, transcript_segment_service
from config import settings
from prompts.base import BUILTIN_NOTE_KEYS

# --- LOGGING CONFIGURATION ---
# 1. Basic config for the app
logging.basicConfig(
    level=logging.INFO, # Changed default to INFO to reduce general noise
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# 2. Suppress noisy third-party libraries
for lib in ['botocore', 'boto3', 'urllib3', 's3transfer', 'asyncio', 'websockets']:
    logging.getLogger(lib).setLevel(logging.WARNING)

# 3. Ensure our app logs are visible
logging.getLogger("api").setLevel(logging.DEBUG)
logging.getLogger("services").setLevel(logging.DEBUG)

# Initialize DashScope
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# Target Alibaba Cloud Model
DASHSCOPE_MODEL_ID = "qwen3-next-80b-a3b-instruct"

# API router and services
try:
    from api import api_router
    from api.v1.transcribe import transcribe_alibaba
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "API modules could not be imported. Ensure your api package and transcribe module are available."
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
    template_id: Optional[str] = None  # optional: reference to a saved template
    encounter_time: Optional[str] = None  # ISO timestamp the frontend records at end of transcript
    encounter_type: Optional[str] = None  # e.g., "in-person" or "telehealth"
    consultation_id: Optional[str] = None # Added for final cleanup diarization


class CommandPayload(BaseModel):
    note_content: Dict[str, Any]
    command: str


# ---- PHI Masking Utility ----
def _mask_phi_entities(text: str, entities: List[dict], mask_token: str = "patient") -> str:
    """
    Replace all text spans labeled as PROTECTED_HEALTH_INFORMATION in Comprehend Medical entities
    with a generic mask token. This is done by offset in reverse order to preserve indices.
    """
    # Only consider PHI entities with valid offsets
    spans = [
        (e["BeginOffset"], e["EndOffset"])
        for e in entities
        if e.get("Category") == "PROTECTED_HEALTH_INFORMATION"
        and e.get("BeginOffset") is not None
        and e.get("EndOffset") is not None
        and e["BeginOffset"] < e["EndOffset"]
    ]
    # Sort in reverse order so replacement doesn't affect later offsets
    spans.sort(reverse=True)
    masked = text
    for start, end in spans:
        masked = masked[:start] + mask_token + masked[end:]
    return masked


# ---- Translation helpers (large-text chunking) ----
TRANSLATE_MAX_BYTES = 9000  # margin under 10,000 bytes


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
    """
    Translates large text using Qwen-MT via DashScope, handling chunking.
    """
    if not text.strip():
        return text

    # Set up prompt based on target language
    if target_lang == "en":
        sys_prompt = "You are a specialized medical translator. Translate the following text (which may be Cantonese, Mandarin, or mixed Chinese) into English. Preserve medical terminology accuracy. Output ONLY the English translation."
    elif target_lang == "zh-TW":
        sys_prompt = "You are a specialized translator. Convert the following text into Traditional Chinese (Hong Kong standard). Output ONLY the converted text."
    else:
        sys_prompt = f"Translate the following text to {target_lang}. Output ONLY the translation."

    def _call_qwen(chunk: str) -> str:
        try:
            # qwen-mt-turbo only supports 'user' and 'assistant' roles, not 'system'
            # So we prepend the system prompt to the user message
            combined_message = f"{sys_prompt}\n\n{chunk}"
            response = dashscope.Generation.call(
                model='qwen-mt-turbo',
                messages=[
                    {'role': Role.USER, 'content': combined_message}
                ],
                result_format='message'
            )
            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0].message.content.strip()
            else:
                return chunk
        except Exception as e:
            logger.error("Translation error: %s", e)
            return chunk

    if _utf8_len(text) <= TRANSLATE_MAX_BYTES:
        return _call_qwen(text)

    pieces = _chunk_text_for_translate(text, TRANSLATE_MAX_BYTES)
    out: List[str] = []
    for piece in pieces:
        out.append(_call_qwen(piece))
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


# ---- DashScope invocation + robust JSON extraction helpers ----
def _find_json_substring(text: str) -> Tuple[str, Optional[str]]:
    """
    A truly robust JSON finder. It locates the first occurrence of a complete
    JSON object or array, ignoring any leading text or markdown fences.
    """
    if not text:
        return "", "Empty model output"

    # Find the start of the first JSON object '{' or array '['
    first_brace = text.find('{')
    first_bracket = text.find('[')

    # Determine which comes first, if any
    start_index = -1
    start_char = ''
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start_index = first_brace
        start_char = '{'
    elif first_bracket != -1:
        start_index = first_bracket
        start_char = '['
    else:
        return "", "Could not find JSON object or array in model output."

    end_char = '}' if start_char == '{' else ']'
    level = 0
    in_string = False

    for i in range(start_index, len(text)):
        char = text[i]

        if char == '"' and (i == 0 or text[i-1] != '\\'):
            in_string = not in_string
        
        if not in_string:
            if char == start_char:
                level += 1
            elif char == end_char:
                level -= 1
        
        if level == 0:
            return text[start_index : i + 1], None

    return "", "Could not find matching closing brace/bracket for JSON."


def _repair_and_parse_json(json_substring: str) -> Dict[str, Any]:
    """
    Try to fix common mistakes and parse JSON.
    """
    s = json_substring
    # Quick pattern fixes observed from LLM outputs
    s = re.sub(r'"text:\s*"', '"text": "', s)
    s = re.sub(r'"(\w+):\s*', r'"\1": ', s)
    parsed = json.loads(s)
    return parsed


def _invoke_dashscope_and_parse(system_prompt: str, user_payload: str, temperature: float = 0.1) -> Dict[str, Any]:
    """
    Invokes Alibaba Cloud DashScope model and returns parsed JSON object.
    Replaces the previous Bedrock invocation.
    """
    messages = [
        {'role': Role.SYSTEM, 'content': system_prompt},
        {'role': Role.USER, 'content': user_payload}
    ]

    try:
        response = dashscope.Generation.call(
            model=DASHSCOPE_MODEL_ID,
            messages=messages,
            result_format='message',
            temperature=temperature,
        )

        if response.status_code == HTTPStatus.OK:
            model_output = response.output.choices[0].message.content
        else:
            raise ValueError(f"DashScope API Error: {response.code} - {response.message}")

    except Exception as exc:
        raise ValueError(f"DashScope invocation failed: {exc}") from exc

    logger.debug("MODEL OUTPUT (full): %s", model_output)

    json_substring, find_err = _find_json_substring(model_output)
    if find_err:
        raise ValueError(find_err + f" Raw model output (first 1000 chars): {model_output[:1000]}")

    try:
        parsed = _repair_and_parse_json(json_substring)
        logger.info("Successfully parsed JSON substring from model output with %d keys.", len(parsed))
        return parsed
    except Exception as exc:
        error_msg = (
            "Failed to parse JSON from final note even after repair. "
            f"Initial parse error: {exc}. "
            f"Full model output (first 2000 chars): '{model_output[:2000]}'"
        )
        raise ValueError(error_msg) from exc


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
    High-level wrapper to prepare user payload and call DashScope + parsing.
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
    return _invoke_dashscope_and_parse(system_prompt, user_payload, temperature=temperature)


def _prepare_template_instructions(template_item: dict) -> str:
    """
    Convert the TemplateRead dict into a human-readable but strict instruction block
    that biases the LLM to produce a JSON note that matches the template sections.
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

    parts.append('\nIf a section is truly not applicable, set its value to the string "None" (without quotes).')
    parts.append("Do not invent additional top-level keys. Order of keys does not matter, but keys must match.")

    return "\n".join(parts)


# ---- Key canonicalization and normalization helpers ----
def _canonicalize_note_keys(
    note: Dict[str, Any],
    expected_keys: List[str],
    note_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Map model output keys to the canonical expected keys (case-insensitive),
    with a few common aliases per note type.
    """
    if not isinstance(note, dict) or not expected_keys:
        return note

    # Base mapping: case-insensitive exact matches
    canonical_map = {ek.lower(): ek for ek in expected_keys}

    # Note-type-specific aliases
    if note_type == "soap":
        canonical_map.update({
            "subjective": "Subjective",
            "subj": "Subjective",
            "s": "Subjective",
            "objective": "Objective",
            "obj": "Objective",
            "o": "Objective",
            "assessment": "Assessment",
            "a": "Assessment",
            "plan": "Plan",
            "p": "Plan",
        })
    elif note_type == "standard":
        canonical_map.update({
            "chief complaint": "Chief Complaint",
            "history of present illness": "History of Present Illness",
            "pertinent negatives": "Pertinent Negatives",
            "past medical history": "Past Medical History",
            "medications": "Medications",
            "assessment and plan": "Assessment and Plan",
        })
    elif note_type == "consultation":
        canonical_map.update({
            "consultation request": "Consultation Request",
            "history of present illness": "History of Present Illness",
            "past medical history": "Past Medical History",
            "past surgical history": "Past Surgical History",
            "family history": "Family History",
            "social history": "Social History",
            "current medications": "Current Medications",
            "medications": "Current Medications",
            "allergies": "Allergies",
            "pertinent physical examination": "Pertinent Physical Examination",
            "physical examination": "Pertinent Physical Examination",
            "diagnostic studies reviewed": "Diagnostic Studies Reviewed",
            "diagnostic studies": "Diagnostic Studies Reviewed",
            "assessment": "Assessment",
            "recommendations": "Recommendations",
        })
    elif note_type == "hp":
        canonical_map.update({
            "chief complaint": "Chief Complaint",
            "history of present illness": "History of Present Illness",
            "past medical history": "Past Medical History",
            "past surgical history": "Past Surgical History",
            "family history": "Family History",
            "social history": "Social History",
            "medications": "Medications",
            "allergies": "Allergies",
            "review of systems": "Review of Systems",
            "physical examination": "Physical Examination",
            "assessment and plan": "Assessment and Plan",
        })

    out: Dict[str, Any] = {}
    # First pass: move any keys that match (case-insensitive/alias)
    for k, v in list(note.items()):
        kl = k.lower() if isinstance(k, str) else k
        target = canonical_map.get(kl) if isinstance(kl, str) else None
        if target:
            # Prefer non-empty existing if duplicate
            if target in out and out[target] not in (None, "", [], {}):
                continue
            out[target] = v
        else:
            # Keep unknowns for now; they may be dropped later by normalize
            out[k] = v

    return out


# ---- Placeholder coercion and key normalization ----
_PLACEHOLDER_VALUES = {
    "undefined",
    "null",
    "none",
    "n/a",
    "na",
    "not provided",
    "not discussed",
    "not mentioned",
    "empty",
    "",
}


def _coerce_placeholder_to_none(val: Any) -> Any:
    """
    Convert various placeholder / empty sentinel strings to the literal 'None'.
    Leaves structured objects/lists unless they themselves contain placeholder-only content.
    """
    if isinstance(val, str):
        v = val.strip().lower()
        if v in _PLACEHOLDER_VALUES:
            return "None"
        # Catch short variants like "none." or "none," etc.
        if v.startswith("none") and len(v) <= 6:
            return "None"
    return val


def _sanitize_note_values(note: Dict[str, Any]) -> Dict[str, Any]:
    """
    Walk all top-level sections and:
    - Coerce placeholder strings to 'None'
    - If a list contains a single object whose 'text' is placeholder, set the section to 'None'
    - If a list is empty, set to 'None'
    - Enforce {"text": "..."} structure for items in a list.
    """
    if not isinstance(note, dict):
        return note

    for k, v in list(note.items()):
        # Direct string case
        if isinstance(v, str):
            note[k] = _coerce_placeholder_to_none(v)
            continue

        # List case
        if isinstance(v, list):
            if not v:
                note[k] = "None"
                continue
            
            # Check if the list contains only a single placeholder entry
            if len(v) == 1:
                item = v[0]
                text_val = ""
                if isinstance(item, dict):
                    text_val = str(item.get("text", "")).strip().lower()
                elif isinstance(item, str):
                    text_val = item.strip().lower()
                
                if text_val in _PLACEHOLDER_VALUES or (text_val.startswith("none") and len(text_val) <= 6):
                    note[k] = "None"
                    continue

            # Sanitize each item in the list to enforce {"text": "..."}
            sanitized_list = []
            for item in v:
                text_to_add = None
                if isinstance(item, dict):
                    text_to_add = item.get("text")
                elif isinstance(item, str):
                    text_to_add = item
                
                if text_to_add is not None:
                    coerced = _coerce_placeholder_to_none(text_to_add)
                    # Don't add items that are just placeholders, unless it's the only one (handled above)
                    if coerced != "None" or len(v) == 1:
                         sanitized_list.append({"text": str(coerced)})
            
            if not sanitized_list:
                note[k] = "None"
            else:
                note[k] = sanitized_list
            continue

        # Dict sub-sections (e.g., Physical Examination object) – shallow sanitize fields
        if isinstance(v, dict):
            inner = {}
            for ik, iv in v.items():
                if isinstance(iv, str):
                    inner[ik] = _coerce_placeholder_to_none(iv)
                else:
                    inner[ik] = iv
            note[k] = inner

    return note


def normalize_note_keys(note: Dict[str, Any], expected_keys: List[str]) -> Dict[str, Any]:
    """
    Ensure all expected keys are present in the note.
    Coerce placeholder / empty values to 'None' and drop extras.
    """
    if note == {"error": "insufficient_medical_information"}:
        return note
    if not isinstance(note, dict):
        return {k: "None" for k in expected_keys}

    fixed: Dict[str, Any] = {}
    for k in expected_keys:
        v = note.get(k)

        # Coerce single-item list style like [{"text": "undefined"}]
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict):
            candidate = v[0].get("text")
            if isinstance(candidate, str) and candidate.strip().lower() in _PLACEHOLDER_VALUES:
                v = "None"

        if isinstance(v, str) and v.strip().lower() in _PLACEHOLDER_VALUES:
            fixed[k] = "None"
        elif v in (None, "", [], {}):
            fixed[k] = "None"
        else:
            fixed[k] = v

    # Unexpected keys are dropped (can log if desired)
    return fixed


def get_expected_keys(payload: FinalNotePayload, template_dict: Optional[dict] = None) -> List[str]:
    if getattr(payload, "template_id", None) and template_dict:
        # Template keys
        sections = template_dict.get("sections", [])
        if isinstance(sections, str):
            try:
                sections = json.loads(sections)
            except Exception:
                sections = []
        return [s["name"] for s in sections if isinstance(s, dict) and "name" in s]
    # Built-in
    return BUILTIN_NOTE_KEYS.get(payload.note_type, [])


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
    empty_markers = {
        "no pertinent negatives",
        "no past medical history",
        "not discussed",
        "none mentioned",
        "none",
        "n/a",
        "na",
        "not provided",
        "undefined",
        "null",
        "empty"
    }

    for section in sections_to_check:
        items = note.get(section)
        value = ""

        if isinstance(items, str):
            value = items.strip().lower()
        elif isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
            value = items[0].get("text", "").strip().lstrip("-").lower()

        if value in empty_markers or value == "":
            if note.get(section) != "None":
                logger.info("Normalizing empty section '%s' to 'None'. Raw value='%s'", section, value)
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


# -------- Routes registration (includes WebSocket proxy) --------
def register_routes(app: FastAPI) -> None:
    @app.websocket("/client-transcribe")
    async def client_transcribe(ws: WebSocket) -> None:
        """
        Real-time medical transcription using Alibaba Paraformer.
        
        This endpoint delegates to the Alibaba Paraformer implementation which uses
        content-based language detection (CJK characters) to accurately identify
        Chinese text instead of relying on unreliable ASR language labels.
        
        Breaking Changes from AWS Transcribe:
        - Query parameter 'language_code' is no longer used
        - Language detection is now automatic based on text content
        - Supports English, Mandarin (zh), and Cantonese (yue) via language_hints
        """
        # Delegate to the Alibaba implementation
        await transcribe_alibaba(ws)

    @app.post("/generate-final-note")
    async def generate_final_note(payload: FinalNotePayload, background_tasks: BackgroundTasks) -> Dict[str, Any]:
        # Clean speaker labels (e.g., "[sentence_1]: ...") from the transcript
        full_transcript = re.sub(r"\[[^\]]+\]:\s*", "", (payload.full_transcript or "")).strip()

        if not full_transcript:
            raise HTTPException(status_code=400, detail="Received an empty transcript.")

        logger.info(
            "Received transcript (%d chars) for note_type=%s template_id=%s",
            len(full_transcript),
            payload.note_type,
            getattr(payload, "template_id", None),
        )

        # --- FINAL CLEANUP DIARIZATION ---
        # Trigger a final pass to label any trailing segments (segments % 5 != 0)
        # that haven't been labeled yet. This ensures the DB is consistent.
        if payload.consultation_id:
            logger.info("Triggering final cleanup diarization for %s", payload.consultation_id)
            background_tasks.add_task(transcript_segment_service.diarize_consultation, payload.consultation_id)
        else:
            logger.warning("No consultation_id provided in Generate Note payload; skipping final diarization.")

        patient_info_dict = payload.patient_info.model_dump(exclude_none=True) if payload.patient_info else {}

        try:
            # --- PHI SAFETY ---
            # Create a sanitized version of patient info for the LLM prompt.
            # Never pass raw patient identifiers to the model.
            patient_info_safe = {
                "age": patient_info_dict.get("age"),
                "sex": patient_info_dict.get("sex"),
                "referring_physician": "Dr. [REDACTED]" if patient_info_dict.get("referring_physician") else None,
                "additional_context": patient_info_dict.get("additional_context") # Assume this is non-PHI context
            }
            # Remove None values for a cleaner prompt
            patient_info_safe = {k: v for k, v in patient_info_safe.items() if v is not None}
            
            # The encounter time is also PHI and should not be sent to the LLM.
            # We can, however, send the encounter *type* (e.g., telehealth).
            encounter_type = getattr(payload, "encounter_type", None)

            # Translate to English if needed (chunked safely) using Qwen-MT
            if _has_cjk(full_transcript):
                logger.info("Detected CJK characters. Translating to English with Qwen-MT (chunked).")
                english_transcript = _translate_large_text("zh", "en", full_transcript)
                logger.info("Translation complete. English transcript length=%d.", len(english_transcript))
            else:
                english_transcript = full_transcript

            # Comprehend Medical entity detection (then compact for prompt)
            entities_response = get_comprehend_client().detect_entities_v2(Text=english_transcript)
            entities = entities_response.get("Entities", []) or []
            logger.info("Comprehend Medical returned %d entities.", len(entities))
            ents_compact = _filter_and_compact_entities_for_llm(entities)

            # --- Medical detail validation ---
            MIN_TRANSCRIPT_LENGTH = 50
            if len(english_transcript.strip()) < MIN_TRANSCRIPT_LENGTH:
                raise HTTPException(
                    status_code=400,
                    detail="Transcript too short—contains insufficient medical detail for note generation."
                )

            # Mask PHI entities in the transcript before sending to the LLM
            english_transcript_masked = _mask_phi_entities(english_transcript, entities)

            # Determine base system prompt for the requested note type
            try:
                prompt_module = get_prompt_generator(payload.note_type)
                # Use ONLY the PHI-safe data for prompt generation
                base_system_prompt = prompt_module.generate_prompt(patient_info_safe)
            except Exception as e:
                logger.warning("Unknown note type '%s': %s. Falling back to standard prompt.", payload.note_type, e)
                prompt_module = get_prompt_generator("standard")
                base_system_prompt = prompt_module.generate_prompt(patient_info_safe)

            # Optionally augment with template instructions when template_id is provided
            final_system_prompt = base_system_prompt
            temperature = 0.1  # default temperature
            template_dict: Optional[dict] = None
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

            # Generate note using composed system prompt and pass ONLY PHI-safe metadata
            final_note = generate_note_from_system_prompt(
                final_system_prompt,
                english_transcript_masked,
                ents_compact,
                patient_info=patient_info_safe,
                encounter_time=None,  # NEVER send encounter_time to the LLM
                encounter_type=encounter_type,
                temperature=temperature,
            )

            # DEBUG: log raw keys returned by model (to confirm casing/aliases)
            try:
                logger.debug("Raw model note keys: %s", list(final_note.keys()) if isinstance(final_note, dict) else type(final_note))
            except Exception:
                pass

            # Determine expected keys for enforcement
            expected_keys = get_expected_keys(payload, template_dict)

            # 1) Canonicalize keys to expected case/aliases BEFORE sanitization
            if expected_keys:
                final_note = _canonicalize_note_keys(final_note, expected_keys, payload.note_type)

            # 2) Sanitize placeholders like 'undefined' and empty shells
            final_note = _sanitize_note_values(final_note)

            # 3) Normalize: fill missing/empty sections with "None" and enforce keys
            if expected_keys:
                final_note = normalize_note_keys(final_note, expected_keys)

            # Optional normalization for "standard"
            if payload.note_type == "standard":
                final_note = _normalize_assessment_plan(
                    _normalize_empty_sections(_fix_pertinent_negatives(final_note))
                )

            return {"notes": final_note}
        except HTTPException as exc:
            # Preserve intended status codes (e.g., 400 guardrail failures)
            raise exc
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

        # DashScope Call
        messages = [
            {'role': Role.SYSTEM, 'content': system_prompt},
            {'role': Role.USER, 'content': user_prompt}
        ]

        try:
            response = dashscope.Generation.call(
                model=DASHSCOPE_MODEL_ID,
                messages=messages,
                result_format='message',
                temperature=0.2
            )
            
            if response.status_code == HTTPStatus.OK:
                result_text = response.output.choices[0].message.content.strip()
            else:
                raise HTTPException(status_code=500, detail=f"DashScope error: {response.message}")

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
          - id: "template:"
          - name: template name
          - description: "Custom template"
          - source: "template"
          - template_id: the UUID string

        Builtin prompt modules are returned with:
          - id: key from PROMPT_REGISTRY (e.g., "standard")
          - name: module.NOTE_TYPE
          - description: "Generate a "
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


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)