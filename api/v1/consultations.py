from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import guard_service, get_session
from schemas.consultation import (
    ConsultationCreate,
    ConsultationRead,
    ConsultationSummary,
    ConsultationUpdate,
)
from services import ConsultationService

router = APIRouter()


@router.get("/", response_model=List[ConsultationSummary])
async def list_consultations(
    user_id: UUID,
    patient_id: Optional[UUID] = None,
    limit: int = 100,
    offset: int = 0,
    include_patient: bool = False,
    session: AsyncSession = Depends(get_session),
) -> List[ConsultationSummary]:
    return await guard_service(
        ConsultationService.list_for_user(
            session,
            user_id=user_id,
            patient_id=patient_id,
            limit=limit,
            offset=offset,
            summary=True,
            include_patient=include_patient,
        )
    )


@router.post("/", response_model=ConsultationRead, status_code=status.HTTP_201_CREATED)
async def create_consultation(
    payload: ConsultationCreate,
    session: AsyncSession = Depends(get_session),
) -> ConsultationRead:
    return await guard_service(ConsultationService.create(session, payload))


@router.get("/{consultation_id}", response_model=ConsultationRead)
async def get_consultation(
    consultation_id: UUID,
    include_patient: bool = False,
    include_segments: bool = False,
    include_clinical_note: bool = False,
    session: AsyncSession = Depends(get_session),
) -> ConsultationRead:
    return await guard_service(
        ConsultationService.get(
            session,
            consultation_id,
            include_patient=include_patient,
            include_segments=include_segments,
            include_clinical_note=include_clinical_note,
        )
    )


@router.patch("/{consultation_id}", response_model=ConsultationRead)
async def update_consultation(
    consultation_id: UUID,
    payload: ConsultationUpdate,
    session: AsyncSession = Depends(get_session),
) -> ConsultationRead:
    return await guard_service(
        ConsultationService.update(session, consultation_id, payload)
    )


@router.delete("/{consultation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_consultation(
    consultation_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> Response:
    await guard_service(ConsultationService.delete(session, consultation_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)