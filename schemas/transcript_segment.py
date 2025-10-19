from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, conint, constr

SpeakerField = constr(max_length=50)
LanguageField = constr(max_length=20)
ConsultationIdField = constr(min_length=1, max_length=64)  # accept any non-empty id

# For maximum compatibility: allow either a raw list of entities or a dict wrapper (e.g., {"Entities":[...]})
EntitiesField = Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]


class TranscriptSegmentBase(BaseModel):
    # Keep required for read models; will override in Create
    consultation_id: ConsultationIdField  # accept string ids (UUID or not)
    sequence_number: conint(ge=0)
    speaker_label: Optional[SpeakerField] = None
    speaker_role: Optional[SpeakerField] = None
    original_text: constr(min_length=1)
    translated_text: Optional[str] = None
    detected_language: Optional[LanguageField] = None
    start_time_ms: Optional[conint(ge=0)] = None
    end_time_ms: Optional[conint(ge=0)] = None
    entities: EntitiesField = None  # <- allow list/dict/None

    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
        extra="forbid",
    )


# Make consultation_id optional in Create so path param can be injected in the route.
class TranscriptSegmentCreate(TranscriptSegmentBase):
    consultation_id: Optional[ConsultationIdField] = None


class TranscriptSegmentUpdate(BaseModel):
    translated_text: Optional[str] = None
    detected_language: Optional[LanguageField] = None
    entities: EntitiesField = None

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class TranscriptSegmentRead(TranscriptSegmentBase):
    segment_id: UUID
    created_at: datetime