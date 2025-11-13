from __future__ import annotations

from typing import Any, Dict, List, Optional

from data.dynamodb import get_session


def get_comprehend_client():
    return get_session().client("comprehendmedical")


def get_translate_client():
    return get_session().client("translate")


def to_english(text: str, detected_language: Optional[str]) -> str:
    if not text:
        return ""
    if not detected_language:
        return text
    # If it's already English-ish, skip translation
    if detected_language.lower().startswith("en"):
        return text
    try:
        source = detected_language.split("-")[0]
        res = get_translate_client().translate_text(
            Text=text,
            SourceLanguageCode=source or "auto",
            TargetLanguageCode="en",
        )
        return res.get("TranslatedText", text)
    except Exception:
        return text

def to_traditional_chinese(text: str) -> str:
    """
    Translates Simplified Chinese text to Traditional Chinese.
    """
    if not text:
        return ""
    try:
        res = get_translate_client().translate_text(
            Text=text,
            SourceLanguageCode="zh",
            TargetLanguageCode="zh-TW",
        )
        return res.get("TranslatedText", text)
    except Exception:
        # Fallback to original text on any error
        return text


def detect_entities(text_en: str) -> List[Dict[str, Any]]:
    if not text_en:
        return []
    try:
        resp = get_comprehend_client().detect_entities_v2(Text=text_en)
        return resp.get("Entities", [])
    except Exception:
        return []