from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, conint, constr

SpeakerField = constr(max_length=50)
LanguageField = constr(max_length=20)
ConsultationIdField = constr(min_length=1, max_length=64)  # accept any non-empty id

# Compact entity span used for storage efficiency:
# b = BeginOffset, e = EndOffset, c = Category, y = Type
class EntityCompact(BaseModel):
    b: conint(ge=0)
    e: conint(ge=0)
    c: constr(max_length=64)
    y: constr(max_length=64)

# For maximum compatibility: allow either a raw list of entities or a dict wrapper (e.g., {"Entities":[...]})
EntitiesField = Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]


class TranscriptSegmentBase(BaseModel):
    # Keep required for read models; override in Create to be optional
    consultation_id: ConsultationIdField  # accept string ids (UUID or not)
    sequence_number: conint(ge=0)
    speaker_label: Optional[SpeakerField] = None
    speaker_role: Optional[SpeakerField] = None
    original_text: constr(min_length=1)
    translated_text: Optional[str] = None
    detected_language: Optional[LanguageField] = None
    start_time_ms: Optional[conint(ge=0)] = None
    end_time_ms: Optional[conint(ge=0)] = None

    # Enrichment fields
    # entities: legacy/raw entities shape (list of dicts or a dict with "Entities")
    entities: EntitiesField = None
    # entities_compact: compact form persisted in Dynamo
    entities_compact: Optional[List[EntityCompact]] = None
    # entities_ref: which text the offsets are aligned to ("original" or "translated")
    entities_ref: Optional[Literal["original", "translated"]] = None

    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
        extra="forbid",
    )


# Make consultation_id optional in Create so the path param can be injected in the route.
class TranscriptSegmentCreate(TranscriptSegmentBase):
    consultation_id: Optional[ConsultationIdField] = None


class TranscriptSegmentUpdate(BaseModel):
    translated_text: Optional[str] = None
    detected_language: Optional[LanguageField] = None
    # Allow updating either raw entities or compact form
    entities: EntitiesField = None
    entities_compact: Optional[List[EntityCompact]] = None
    entities_ref: Optional[Literal["original", "translated"]] = None
    speaker_role: Optional[SpeakerField] = None
    speaker_label: Optional[SpeakerField] = None

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class TranscriptSegmentRead(TranscriptSegmentBase):
    segment_id: UUID
    created_at: datetime