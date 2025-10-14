from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, constr

SEX_ENUM_VALUES = ("male", "female", "other", "unknown")

FullName = constr(min_length=1, max_length=255)
ShortStr255 = constr(max_length=255)
PhoneStr = constr(max_length=50)


class PatientBase(BaseModel):
    full_name: FullName
    date_of_birth: Optional[date] = None
    sex: Optional[Literal["male", "female", "other", "unknown"]] = Field(
        default=None,
        description="One of male/female/other/unknown",
    )
    hkid_number: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[PhoneStr] = None
    referring_physician: Optional[ShortStr255] = None
    additional_context: Optional[str] = None
    is_starred: bool = False

    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
        extra="forbid",
    )


class PatientCreate(PatientBase):
    user_id: UUID


class PatientUpdate(BaseModel):
    full_name: Optional[FullName] = None
    date_of_birth: Optional[date] = None
    sex: Optional[Literal["male", "female", "other", "unknown"]] = Field(
        default=None,
        description="One of male/female/other/unknown",
    )
    hkid_number: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[PhoneStr] = None
    referring_physician: Optional[ShortStr255] = None
    additional_context: Optional[str] = None
    is_starred: Optional[bool] = None

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class PatientSummary(BaseModel):
    patient_id: UUID
    full_name: str
    is_starred: bool

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class PatientRead(PatientBase):
    patient_id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime