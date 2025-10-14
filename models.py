# models.py
from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ENUM, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


sex_enum = ENUM(
    "male",
    "female",
    "other",
    "unknown",
    name="sex_enum",
    create_type=False,
)


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    patients: Mapped[List["Patient"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    consultations: Mapped[List["Consultation"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Patient(Base):
    __tablename__ = "patients"
    __table_args__ = (
        UniqueConstraint("user_id", "email", name="patients_email_unique_per_user"),
    )

    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    date_of_birth: Mapped[Optional[date]] = mapped_column(Date(), nullable=True)
    sex: Mapped[Optional[str]] = mapped_column(sex_enum, nullable=True)
    hkid_number: Mapped[Optional[str]] = mapped_column(Text(), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    referring_physician: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    additional_context: Mapped[Optional[str]] = mapped_column(Text(), nullable=True)
    is_starred: Mapped[bool] = mapped_column(
        Boolean(),
        nullable=False,
        server_default="false",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    user: Mapped["User"] = relationship(back_populates="patients")
    consultations: Mapped[List["Consultation"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Consultation(Base):
    __tablename__ = "consultations"

    consultation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patients.patient_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    language_code: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    note_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    session_state: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    patient: Mapped["Patient"] = relationship(back_populates="consultations")
    user: Mapped["User"] = relationship(back_populates="consultations")
    transcript_segments: Mapped[List["TranscriptSegment"]] = relationship(
        back_populates="consultation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="TranscriptSegment.sequence_number",
    )
    clinical_note: Mapped[Optional["ClinicalNote"]] = relationship(
        back_populates="consultation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"
    __table_args__ = (
        UniqueConstraint(
            "consultation_id",
            "sequence_number",
            name="uq_segments_consultation_sequence",
        ),
    )

    segment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    consultation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("consultations.consultation_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence_number: Mapped[int] = mapped_column(Integer(), nullable=False)
    speaker_label: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    speaker_role: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    original_text: Mapped[str] = mapped_column(Text(), nullable=False)
    translated_text: Mapped[Optional[str]] = mapped_column(Text(), nullable=True)
    detected_language: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    start_time_ms: Mapped[Optional[int]] = mapped_column(Integer(), nullable=True)
    end_time_ms: Mapped[Optional[int]] = mapped_column(Integer(), nullable=True)
    entities: Mapped[Optional[dict]] = mapped_column(JSONB(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    consultation: Mapped["Consultation"] = relationship(back_populates="transcript_segments")


class ClinicalNote(Base):
    __tablename__ = "clinical_notes"

    note_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    consultation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("consultations.consultation_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    note_type: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[dict] = mapped_column(JSONB(), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    last_edited_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    consultation: Mapped["Consultation"] = relationship(back_populates="clinical_note")