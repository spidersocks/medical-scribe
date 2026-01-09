from __future__ import annotations

import logging
import os
import json
import re
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple

import dashscope
from dashscope.api_entities.dashscope_response import Role

from data.dynamodb import get_session

logger = logging.getLogger(__name__)

# Ensure API key is available
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


def get_comprehend_client():
    return get_session().client("comprehendmedical")


def _find_json_substring(text: str) -> Tuple[str, Optional[str]]:
    """
    Locates the first occurrence of a complete JSON object or array.
    """
    if not text:
        return "", "Empty model output"

    first_brace = text.find('{')
    first_bracket = text.find('[')

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
    # Quick pattern fixes
    s = re.sub(r'"text:\s*"', '"text": "', s)
    s = re.sub(r'"(\w+):\s*', r'"\1": ', s)
    return json.loads(s)


def _qwen_translate(text: str, system_prompt: str) -> str:
    """
    Helper to call Qwen-MT via DashScope for translation.
    """
    if not text.strip():
        return text
    
    try:
        # qwen-mt-turbo only supports 'user' and 'assistant' roles, not 'system'
        combined_message = f"{system_prompt}\n\n{text}"
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
            logger.error(
                "Translation failed with status %s: %s",
                response.status_code,
                getattr(response, 'message', 'No error message')
            )
            return text
    except Exception as e:
        logger.error("Translation exception: %s", e, exc_info=True)
        return text


def to_english(text: str, detected_language: Optional[str] = None) -> str:
    if not text:
        return ""
    
    # Check if text contains CJK characters
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    
    # If explicitly English AND no CJK characters, we can skip.
    if detected_language and detected_language.lower().startswith("en") and not has_cjk:
        return text

    sys_prompt = (
        "You are a specialized medical translator. Translate the following text "
        "(which may be Cantonese, Mandarin, or mixed Chinese) into English. "
        "Preserve medical terminology accuracy. Output ONLY the English translation."
    )
    return _qwen_translate(text, sys_prompt)


def to_traditional_chinese(text: str) -> str:
    """
    Translates Simplified Chinese text (from Paraformer) to Traditional Chinese.
    """
    if not text:
        return ""
    sys_prompt = (
        "You are a specialized translator. Convert the following text into "
        "Traditional Chinese (Hong Kong standard). Output ONLY the converted text."
    )
    return _qwen_translate(text, sys_prompt)


def detect_entities(text_en: str) -> List[Dict[str, Any]]:
    if not text_en:
        return []
    try:
        resp = get_comprehend_client().detect_entities_v2(Text=text_en)
        return resp.get("Entities", [])
    except Exception:
        return []


def predict_speaker_roles(segments: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Uses Qwen to infer speaker roles (Doctor vs Patient) from a list of transcript segments.
    segments: list of dicts with {"id": str, "text": str}
    Returns: Dict[segment_id, role_string]
    """
    if not segments:
        return {}
    
    # Prepare JSON context for the LLM
    context_data = [
        {"id": s["id"], "text": s["text"]} 
        for s in segments
    ]
    context_str = json.dumps(context_data, ensure_ascii=False, indent=2)

    system_prompt = (
        "You are an expert medical transcription assistant. "
        "Your task is to label each transcript segment as spoken by 'Doctor' or 'Patient' based on the medical context. "
        "The conversation may be in English, Chinese, or mixed. "
        "Return a single valid JSON object where keys are the segment IDs and values are the roles ('Doctor' or 'Patient'). "
        "Do not include any explanation."
    )
    
    user_prompt = f"Here are the transcript segments:\n{context_str}\n\nReturn the JSON mapping:"
    
    try:
        response = dashscope.Generation.call(
            model='qwen-turbo', # Fast and cheap, suitable for classification tasks
            messages=[
                {'role': Role.SYSTEM, 'content': system_prompt},
                {'role': Role.USER, 'content': user_prompt}
            ],
            result_format='message',
            temperature=0.0 # Deterministic
        )
        
        if response.status_code == HTTPStatus.OK:
            content = response.output.choices[0].message.content
            json_str, err = _find_json_substring(content)
            if not err:
                return _repair_and_parse_json(json_str)
            else:
                logger.warning("Failed to find JSON in diarization response: %s", content)
        else:
            logger.error("Diarization API failed: %s", getattr(response, 'message', 'Unknown error'))
            
    except Exception as e:
        logger.error("Diarization exception: %s", e)
    
    return {}