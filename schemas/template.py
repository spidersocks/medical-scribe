from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, constr, Field

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
    # Accept camelCase ownerUserId from Dynamo/service while still exposing owner_user_id in Python model
    owner_user_id: str = Field(..., alias="ownerUserId")
    name: NameStr
    sections: List[TemplateSection]
    example_text: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="forbid")