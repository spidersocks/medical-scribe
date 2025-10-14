from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, constr

NoteType = constr(max_length=50)


class ClinicalNoteBase(BaseModel):
    consultation_id: UUID
    note_type: NoteType
    content: Dict

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class ClinicalNoteCreate(ClinicalNoteBase):
    pass


class ClinicalNoteUpdate(BaseModel):
    note_type: Optional[NoteType] = None
    content: Optional[Dict] = None
    last_edited_at: Optional[datetime] = None

    model_config = ConfigDict(extra="forbid")


class ClinicalNoteRead(ClinicalNoteBase):
    note_id: UUID
    generated_at: datetime
    last_edited_at: Optional[datetime] = None