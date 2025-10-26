from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, constr

NameStr = constr(max_length=255)


class TemplateSection(BaseModel):
    id: str
    name: str
    description: str


class TemplateCreate(BaseModel):
    name: NameStr
    sections: List[TemplateSection]
    example_text: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class TemplateUpdate(BaseModel):
    name: Optional[NameStr] = None
    sections: Optional[List[TemplateSection]] = None
    example_text: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class TemplateRead(BaseModel):
    id: UUID
    owner_user_id: str
    name: NameStr
    sections: List[TemplateSection]
    example_text: Optional[str] = None
    created_at: datetime
    updated_at: datetime