from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from data.dynamodb import get_session
from google.cloud import translate_v3 as translate

# Boto3 client for Comprehend Medical (unchanged)
def get_comprehend_client():
    return get_session().client("comprehendmedical")


# Google Translate v3 Client
@lru_cache(maxsize=1)
def get_google_translate_v3_client() -> translate.TranslationServiceClient:
    """
    Initializes the Google Cloud Translation v3 client.
    Assumes GOOGLE_APPLICATION_CREDENTIALS environment variable is set.
    """
    return translate.TranslationServiceClient()


@lru_cache(maxsize=1)
def get_google_project_id() -> str:
    """
    Gets the Google Cloud Project ID from environment variables.
    The v3 client requires the project ID for requests.
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        # As a fallback, try to infer from default credentials
        try:
            import google.auth
            _, project_id = google.auth.default()
            if project_id:
                return project_id
        except (ImportError, google.auth.exceptions.DefaultCredentialsError):
            pass  # Will raise RuntimeError below
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT environment variable must be set to use the Google Translate API."
        )
    return project_id


def to_english(text: str, detected_language: Optional[str]) -> str:
    """
    Translates a given text to English using Google Translate API.
    If the text is already in English or translation fails, it returns the original text.
    """
    if not text or not detected_language:
        return text

    # If it's already English-ish, skip translation
    if detected_language.lower().startswith("en"):
        return text

    try:
        client = get_google_translate_v3_client()
        project_id = get_google_project_id()
        parent = f"projects/{project_id}"

        # Google Translate API uses the base language code (e.g., 'zh' for 'zh-HK')
        source_lang = detected_language.split("-")[0]

        response = client.translate_text(
            parent=parent,
            contents=[text],
            mime_type="text/plain",
            source_language_code=source_lang,
            target_language_code="en",
        )

        return response.translations[0].translated_text if response.translations else text
    except Exception:
        # In case of any translation error, log it and fall back to the original text
        # logger.error(f"Google Translate failed for lang '{detected_language}': {e}")
        return text


def detect_entities(text_en: str) -> List[Dict[str, Any]]:
    """Detects medical entities using Amazon Comprehend Medical (unchanged)."""
    if not text_en:
        return []
    try:
        resp = get_comprehend_client().detect_entities_v2(Text=text_en)
        return resp.get("Entities", [])
    except Exception:
        return []