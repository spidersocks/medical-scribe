from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from repositories import (
    create_clinical_note,
    delete_clinical_note,
    get_clinical_note,
    get_clinical_note_by_consultation,
    get_consultation,
    update_clinical_note,
)
from schemas.clinical_note import ClinicalNoteCreate, ClinicalNoteRead, ClinicalNoteUpdate
from .base import ensure_condition, unwrap_or_raise
from .exceptions import ValidationError


class ClinicalNoteService:
    @staticmethod
    async def create(session: AsyncSession, payload: ClinicalNoteCreate) -> ClinicalNoteRead:
        consultation = unwrap_or_raise(
            await get_consultation(session, payload.consultation_id),
            resource_name="Consultation",
        )
        existing = await get_clinical_note_by_consultation(session, payload.consultation_id)
        ensure_condition(existing is None, "Consultation already has a clinical note.")
        note = await create_clinical_note(session, payload)
        return ClinicalNoteRead.from_orm(note)

    @staticmethod
    async def get(session: AsyncSession, note_id: UUID) -> ClinicalNoteRead:
        note = unwrap_or_raise(
            await get_clinical_note(session, note_id),
            resource_name="ClinicalNote",
        )
        return ClinicalNoteRead.from_orm(note)

    @staticmethod
    async def get_by_consultation(session: AsyncSession, consultation_id: UUID) -> ClinicalNoteRead:
        note = unwrap_or_raise(
            await get_clinical_note_by_consultation(session, consultation_id),
            resource_name="ClinicalNote",
        )
        return ClinicalNoteRead.from_orm(note)

    @staticmethod
    async def upsert(
        session: AsyncSession,
        consultation_id: UUID,
        payload: ClinicalNoteUpdate,
    ) -> ClinicalNoteRead:
        note = await get_clinical_note_by_consultation(session, consultation_id)
        if note is None:
            ensure_condition(payload.content is not None, "Cannot create note without content.")
            create_payload = ClinicalNoteCreate(
                consultation_id=consultation_id,
                note_type=payload.note_type or "standard",
                content=payload.content,
            )
            note = await create_clinical_note(session, create_payload)
        else:
            note = await update_clinical_note(session, note, payload)
        return ClinicalNoteRead.from_orm(note)

    @staticmethod
    async def update(session: AsyncSession, note_id: UUID, payload: ClinicalNoteUpdate) -> ClinicalNoteRead:
        note = unwrap_or_raise(
            await get_clinical_note(session, note_id),
            resource_name="ClinicalNote",
        )
        note = await update_clinical_note(session, note, payload)
        return ClinicalNoteRead.from_orm(note)

    @staticmethod
    async def delete(session: AsyncSession, note_id: UUID) -> None:
        note = unwrap_or_raise(
            await get_clinical_note(session, note_id),
            resource_name="ClinicalNote",
        )
        await delete_clinical_note(session, note)