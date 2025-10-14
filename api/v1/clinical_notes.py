from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import guard_service, get_session
from schemas.clinical_note import (
    ClinicalNoteCreate,
    ClinicalNoteRead,
    ClinicalNoteUpdate,
)
from services import ClinicalNoteService

router = APIRouter()


@router.post(
    "/consultations/{consultation_id}/clinical-note",
    response_model=ClinicalNoteRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_clinical_note(
    consultation_id: UUID,
    payload: ClinicalNoteCreate,
    session: AsyncSession = Depends(get_session),
) -> ClinicalNoteRead:
    payload = payload.copy(update={"consultation_id": consultation_id})
    return await guard_service(ClinicalNoteService.create(session, payload))


@router.get(
    "/consultations/{consultation_id}/clinical-note",
    response_model=ClinicalNoteRead,
)
async def get_clinical_note_for_consultation(
    consultation_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> ClinicalNoteRead:
    return await guard_service(
        ClinicalNoteService.get_by_consultation(session, consultation_id)
    )


@router.put(
    "/consultations/{consultation_id}/clinical-note",
    response_model=ClinicalNoteRead,
)
async def upsert_clinical_note_for_consultation(
    consultation_id: UUID,
    payload: ClinicalNoteUpdate,
    session: AsyncSession = Depends(get_session),
) -> ClinicalNoteRead:
    return await guard_service(
        ClinicalNoteService.upsert(session, consultation_id, payload)
    )


@router.get("/clinical-notes/{note_id}", response_model=ClinicalNoteRead)
async def get_clinical_note(
    note_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> ClinicalNoteRead:
    return await guard_service(ClinicalNoteService.get(session, note_id))


@router.patch("/clinical-notes/{note_id}", response_model=ClinicalNoteRead)
async def update_clinical_note(
    note_id: UUID,
    payload: ClinicalNoteUpdate,
    session: AsyncSession = Depends(get_session),
) -> ClinicalNoteRead:
    return await guard_service(ClinicalNoteService.update(session, note_id, payload))


@router.delete("/clinical-notes/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_clinical_note(
    note_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> Response:
    await guard_service(ClinicalNoteService.delete(session, note_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)