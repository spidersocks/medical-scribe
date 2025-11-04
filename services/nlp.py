from __future__ import annotations
from typing import Any, Dict, List, Optional
from data.dynamodb import get_session

def get_comprehend_client():
    return get_session().client("comprehendmedical")

def get_translate_client():
    return get_session().client("translate")

def _normalize_source_lang_for_aws(detected_language: Optional[str]) -> str:
    if not detected_language:
        return "auto"
    code = detected_language.lower()
    if code.startswith("en"):
        return "en"
    if code.startswith("yue"):           # Cantonese -> use Chinese path
        return "zh"
    if code.startswith("zh-tw") or "hant" in code:
        return "zh-TW"
    if code.startswith("zh-cn") or "hans" in code:
        return "zh"
    if code.startswith("zh"):
        return "zh"
    return "auto"

def to_english(text: str, detected_language: Optional[str]) -> str:
    if not text:
        return ""
    if detected_language and detected_language.lower().startswith("en"):
        return text
    try:
        source = _normalize_source_lang_for_aws(detected_language)
        res = get_translate_client().translate_text(
            Text=text,
            SourceLanguageCode=source,
            TargetLanguageCode="en",
        )
        return res.get("TranslatedText", text)
    except Exception:
        return text

def detect_entities(text_en: str) -> List[Dict[str, Any]]:
    if not text_en:
        return []
    try:
        resp = get_comprehend_client().detect_entities_v2(Text=text_en)
        return resp.get("Entities", [])
    except Exception:
        return []