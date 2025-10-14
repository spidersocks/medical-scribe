from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Response, status

from api.deps import guard_service
from schemas.patient import (
    PatientCreate,
    PatientRead,
    PatientSummary,
    PatientUpdate,
)
from services import consultation_service, patient_service

router = APIRouter()


@router.get("/", response_model=List[PatientSummary])
async def list_patients(
    user_id: UUID,
    starred_only: bool = False,
    limit: int = 100,
    offset: int = 0,
) -> List[PatientSummary]:
    return await guard_service(
        patient_service.list_for_user(
            str(user_id),
            limit=limit,
            offset=offset,
            starred_only=starred_only,
            summary=True,
        )
    )


@router.post("/", response_model=PatientRead, status_code=status.HTTP_201_CREATED)
async def create_patient(payload: PatientCreate) -> PatientRead:
    return await guard_service(patient_service.create(payload))


@router.get("/{patient_id}", response_model=PatientRead)
async def get_patient(
    patient_id: UUID,
    include_consultations: bool = False,
) -> PatientRead:
    patient = await guard_service(patient_service.get(str(patient_id)))

    if include_consultations:
        consultations = await guard_service(
            consultation_service.list_for_patient(
                str(patient_id),
                summary=False,
            )
        )
        patient_data = patient.model_dump()
        patient_data["consultations"] = consultations
        patient = PatientRead.model_validate(patient_data)

    return patient


@router.patch("/{patient_id}", response_model=PatientRead)
async def update_patient(
    patient_id: UUID,
    payload: PatientUpdate,
) -> PatientRead:
    return await guard_service(patient_service.update(str(patient_id), payload))


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(patient_id: UUID) -> Response:
    await guard_service(patient_service.delete(str(patient_id)))
    return Response(status_code=status.HTTP_204_NO_CONTENT)