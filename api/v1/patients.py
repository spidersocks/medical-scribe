from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import guard_service, get_session
from schemas.patient import (
    PatientCreate,
    PatientRead,
    PatientSummary,
    PatientUpdate,
)
from services import PatientService

router = APIRouter()


@router.get("/", response_model=List[PatientSummary])
async def list_patients(
    user_id: UUID,
    starred_only: bool = False,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
) -> List[PatientSummary]:
    return await guard_service(
        PatientService.list_for_user(
            session,
            user_id=user_id,
            limit=limit,
            offset=offset,
            starred_only=starred_only,
            summary=True,
        )
    )


@router.post("/", response_model=PatientRead, status_code=status.HTTP_201_CREATED)
async def create_patient(
    payload: PatientCreate,
    session: AsyncSession = Depends(get_session),
) -> PatientRead:
    return await guard_service(PatientService.create(session, payload))


@router.get("/{patient_id}", response_model=PatientRead)
async def get_patient(
    patient_id: UUID,
    include_consultations: bool = False,
    session: AsyncSession = Depends(get_session),
) -> PatientRead:
    return await guard_service(
        PatientService.get(
            session,
            patient_id,
            include_consultations=include_consultations,
        )
    )


@router.patch("/{patient_id}", response_model=PatientRead)
async def update_patient(
    patient_id: UUID,
    payload: PatientUpdate,
    session: AsyncSession = Depends(get_session),
) -> PatientRead:
    return await guard_service(PatientService.update(session, patient_id, payload))


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(
    patient_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> Response:
    await guard_service(PatientService.delete(session, patient_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)