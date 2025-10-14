from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Response, status

from api.deps import guard_service
from schemas.clinical_note import (
    ClinicalNoteCreate,
    ClinicalNoteRead,
    ClinicalNoteUpdate,
)
from services import clinical_note_service

router = APIRouter()


@router.post(
    "/consultations/{consultation_id}/clinical-note",
    response_model=ClinicalNoteRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_clinical_note(
    consultation_id: UUID,
    payload: ClinicalNoteCreate,
) -> ClinicalNoteRead:
    payload_with_consultation = payload.copy(update={"consultation_id": str(consultation_id)})
    return await guard_service(clinical_note_service.create(payload_with_consultation))


@router.get(
    "/consultations/{consultation_id}/clinical-note",
    response_model=ClinicalNoteRead,
)
async def get_clinical_note_for_consultation(
    consultation_id: UUID,
) -> ClinicalNoteRead:
    return await guard_service(
        clinical_note_service.get_by_consultation(str(consultation_id))
    )


@router.put(
    "/consultations/{consultation_id}/clinical-note",
    response_model=ClinicalNoteRead,
)
async def upsert_clinical_note_for_consultation(
    consultation_id: UUID,
    payload: ClinicalNoteUpdate,
) -> ClinicalNoteRead:
    return await guard_service(
        clinical_note_service.upsert(str(consultation_id), payload)
    )


@router.get("/clinical-notes/{note_id}", response_model=ClinicalNoteRead)
async def get_clinical_note(note_id: UUID) -> ClinicalNoteRead:
    return await guard_service(clinical_note_service.get(str(note_id)))


@router.patch("/clinical-notes/{note_id}", response_model=ClinicalNoteRead)
async def update_clinical_note(
    note_id: UUID,
    payload: ClinicalNoteUpdate,
) -> ClinicalNoteRead:
    return await guard_service(
        clinical_note_service.update(str(note_id), payload)
    )


@router.delete("/clinical-notes/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_clinical_note(note_id: UUID) -> Response:
    await guard_service(clinical_note_service.delete(str(note_id)))
    return Response(status_code=status.HTTP_204_NO_CONTENT)