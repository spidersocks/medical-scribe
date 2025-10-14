from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, constr

from .clinical_note import ClinicalNoteRead
from .patient import PatientSummary
from .transcript_segment import TranscriptSegmentRead

NameStr = constr(max_length=255)
LangCode = constr(max_length=20)
NoteType = constr(max_length=50)
SessionState = constr(max_length=50)


class ConsultationBase(BaseModel):
    patient_id: UUID
    user_id: UUID
    name: Optional[NameStr] = None
    language_code: Optional[LangCode] = None
    note_type: Optional[NoteType] = None
    session_state: Optional[SessionState] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class ConsultationCreate(ConsultationBase):
    pass


class ConsultationUpdate(BaseModel):
    name: Optional[NameStr] = None
    language_code: Optional[LangCode] = None
    note_type: Optional[NoteType] = None
    session_state: Optional[SessionState] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    model_config = ConfigDict(extra="forbid")


class ConsultationSummary(BaseModel):
    consultation_id: UUID
    name: Optional[NameStr] = None
    note_type: Optional[NoteType] = None
    session_state: Optional[SessionState] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class ConsultationRead(ConsultationBase):
    consultation_id: UUID
    created_at: datetime
    updated_at: datetime
    patient: Optional[PatientSummary] = None
    transcript_segments: Optional[List[TranscriptSegmentRead]] = None
    clinical_note: Optional[ClinicalNoteRead] = None