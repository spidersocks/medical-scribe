from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, conint, constr

SpeakerField = constr(max_length=50)
LanguageField = constr(max_length=20)


class TranscriptSegmentBase(BaseModel):
    consultation_id: UUID
    sequence_number: conint(ge=0)
    speaker_label: Optional[SpeakerField] = None
    speaker_role: Optional[SpeakerField] = None
    original_text: constr(min_length=1)
    translated_text: Optional[str] = None
    detected_language: Optional[LanguageField] = None
    start_time_ms: Optional[conint(ge=0)] = None
    end_time_ms: Optional[conint(ge=0)] = None
    entities: Optional[Dict] = None

    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
        extra="forbid",
    )


class TranscriptSegmentCreate(TranscriptSegmentBase):
    pass


class TranscriptSegmentUpdate(BaseModel):
    translated_text: Optional[str] = None
    detected_language: Optional[LanguageField] = None
    entities: Optional[Dict] = None

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class TranscriptSegmentRead(TranscriptSegmentBase):
    segment_id: UUID
    created_at: datetime