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
    if detected_language and detected_language.lower().startswith("en"):
      # already english
      return text
    # best-effort: if we know source lang code (e.g., zh), translate to English
    try:
        source = (detected_language or "auto").split("-")[0]
        res = get_translate_client().translate_text(
            Text=text,
            SourceLanguageCode=source if source != "auto" else "auto",
            TargetLanguageCode="en",
        )
        return res.get("TranslatedText", text)
    except Exception:
        return text


def detect_entities_en(text_en: str) -> List[Dict[str, Any]]:
    if not text_en:
        return []
    try:
        res = get_comprehend_client().detect_entities_v2(Text=text_en)
        return res.get("Entities", [])
    except Exception:
        return []