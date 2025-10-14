from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Response, status

from api.deps import guard_service
from schemas.consultation import (
    ConsultationCreate,
    ConsultationRead,
    ConsultationSummary,
    ConsultationUpdate,
)
from services import (
    clinical_note_service,
    consultation_service,
    patient_service,
    transcript_segment_service,
)

router = APIRouter()


@router.get("/", response_model=List[ConsultationSummary])
async def list_consultations(
    user_id: UUID,
    patient_id: Optional[UUID] = None,
    limit: int = 100,
    offset: int = 0,
    include_patient: bool = False,
) -> List[ConsultationSummary]:
    consultations = await guard_service(
        consultation_service.list_for_user(
            str(user_id),
            patient_id=str(patient_id) if patient_id else None,
            limit=limit,
            offset=offset,
            summary=True,
        )
    )

    if include_patient:
        cache: dict[str, ConsultationSummary] = {}
        enriched: List[ConsultationSummary] = []
        for consultation in consultations:
            pid = getattr(consultation, "patient_id", None)
            if pid is None:
                enriched.append(consultation)
                continue
            pid_str = str(pid)
            if pid_str not in cache:
                cache[pid_str] = await guard_service(patient_service.get(pid_str))
            data = consultation.model_dump()
            data["patient"] = cache[pid_str]
            enriched.append(ConsultationSummary.model_validate(data))
        consultations = enriched

    return consultations


@router.post("/", response_model=ConsultationRead, status_code=status.HTTP_201_CREATED)
async def create_consultation(payload: ConsultationCreate) -> ConsultationRead:
    return await guard_service(consultation_service.create(payload))


@router.get("/{consultation_id}", response_model=ConsultationRead)
async def get_consultation(
    consultation_id: UUID,
    include_patient: bool = False,
    include_segments: bool = False,
    include_clinical_note: bool = False,
) -> ConsultationRead:
    consultation = await guard_service(consultation_service.get(str(consultation_id)))
    data = consultation.model_dump()

    if include_patient:
        patient = await guard_service(patient_service.get(str(data.get("patient_id"))))
        data["patient"] = patient

    if include_segments:
        segments = await guard_service(
            transcript_segment_service.list_for_consultation(str(consultation_id))
        )
        data["segments"] = segments

    if include_clinical_note:
        note = await clinical_note_service.try_get_by_consultation(str(consultation_id))
        if note:
            data["clinical_note"] = note

    return ConsultationRead.model_validate(data)


@router.patch("/{consultation_id}", response_model=ConsultationRead)
async def update_consultation(
    consultation_id: UUID,
    payload: ConsultationUpdate,
) -> ConsultationRead:
    return await guard_service(consultation_service.update(str(consultation_id), payload))


@router.delete("/{consultation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_consultation(consultation_id: UUID) -> Response:
    await guard_service(consultation_service.delete(str(consultation_id)))
    return Response(status_code=status.HTTP_204_NO_CONTENT)