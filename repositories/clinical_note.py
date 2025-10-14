from __future__ import annotations

from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import ClinicalNote
from schemas.clinical_note import ClinicalNoteCreate, ClinicalNoteUpdate


async def create_clinical_note(
    session: AsyncSession,
    payload: ClinicalNoteCreate,
) -> ClinicalNote:
    note = ClinicalNote(**payload.dict())
    session.add(note)
    await session.flush()
    await session.refresh(note)
    return note


async def get_clinical_note(
    session: AsyncSession,
    note_id: UUID,
) -> Optional[ClinicalNote]:
    result = await session.execute(
        select(ClinicalNote).where(ClinicalNote.note_id == note_id)
    )
    return result.scalar_one_or_none()


async def get_clinical_note_by_consultation(
    session: AsyncSession,
    consultation_id: UUID,
) -> Optional[ClinicalNote]:
    result = await session.execute(
        select(ClinicalNote).where(ClinicalNote.consultation_id == consultation_id)
    )
    return result.scalar_one_or_none()


async def update_clinical_note(
    session: AsyncSession,
    note: ClinicalNote,
    payload: ClinicalNoteUpdate,
) -> ClinicalNote:
    for field, value in payload.dict(exclude_unset=True).items():
        setattr(note, field, value)
    await session.flush()
    await session.refresh(note)
    return note


async def delete_clinical_note(session: AsyncSession, note: ClinicalNote) -> None:
    await session.delete(note)
    await session.flush()