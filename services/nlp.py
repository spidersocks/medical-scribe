from __future__ import annotations

import logging
import os
from http import HTTPStatus
from typing import Any, Dict, List, Optional

import dashscope
from dashscope.api_entities.dashscope_response import Role

from data.dynamodb import get_session

logger = logging.getLogger(__name__)

# Ensure API key is available
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


def get_comprehend_client():
    return get_session().client("comprehendmedical")


def _qwen_translate(text: str, system_prompt: str) -> str:
    """
    Helper to call Qwen-MT via DashScope for translation.
    """
    if not text.strip():
        return text
    
    try:
        # qwen-mt-turbo only supports 'user' and 'assistant' roles, not 'system'
        # So we prepend the system prompt to the user message
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
            # On failure return original text
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
    
    # Check if text contains CJK characters (Chinese/Japanese/Korean)
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    
    # If explicitly English AND no CJK characters, we can skip.
    if detected_language and detected_language.lower().startswith("en") and not has_cjk:
        return text

    # Prompt designed to handle mixed Cantonese/Mandarin and medical context
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