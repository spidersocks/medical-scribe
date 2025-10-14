from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from repositories import (
    create_consultation,
    delete_consultation,
    get_consultation,
    get_patient,
    list_consultations_for_user,
    update_consultation,
)
from schemas.consultation import (
    ConsultationCreate,
    ConsultationRead,
    ConsultationSummary,
    ConsultationUpdate,
)
from .base import ensure_condition, unwrap_or_raise


class ConsultationService:
    @staticmethod
    async def create(session: AsyncSession, payload: ConsultationCreate) -> ConsultationRead:
        patient = unwrap_or_raise(
            await get_patient(session, payload.patient_id),
            resource_name="Patient",
        )
        ensure_condition(
            patient.user_id == payload.user_id,
            "Patient does not belong to this user.",
        )
        consultation = await create_consultation(session, payload)
        return ConsultationRead.from_orm(consultation)

    @staticmethod
    async def get(
        session: AsyncSession,
        consultation_id: UUID,
        *,
        include_patient: bool = False,
        include_segments: bool = False,
        include_clinical_note: bool = False,
    ) -> ConsultationRead:
        consultation = unwrap_or_raise(
            await get_consultation(
                session,
                consultation_id,
                include_patient=include_patient,
                include_segments=include_segments,
                include_clinical_note=include_clinical_note,
            ),
            resource_name="Consultation",
        )
        return ConsultationRead.from_orm(consultation)

    @staticmethod
    async def list_for_user(
        session: AsyncSession,
        user_id: UUID,
        *,
        patient_id: Optional[UUID] = None,
        limit: int = 100,
        offset: int = 0,
        summary: bool = False,
        include_patient: bool = False,
    ) -> List[ConsultationSummary | ConsultationRead]:
        consultations = await list_consultations_for_user(
            session,
            user_id=user_id,
            patient_id=patient_id,
            limit=limit,
            offset=offset,
            include_patient=include_patient,
            include_segments=not summary,
            include_clinical_note=not summary,
        )
        if summary:
            return [ConsultationSummary.from_orm(c) for c in consultations]
        return [ConsultationRead.from_orm(c) for c in consultations]

    @staticmethod
    async def update(
        session: AsyncSession,
        consultation_id: UUID,
        payload: ConsultationUpdate,
    ) -> ConsultationRead:
        consultation = unwrap_or_raise(
            await get_consultation(session, consultation_id),
            resource_name="Consultation",
        )
        consultation = await update_consultation(session, consultation, payload)
        return ConsultationRead.from_orm(consultation)

    @staticmethod
    async def delete(session: AsyncSession, consultation_id: UUID) -> None:
        consultation = unwrap_or_raise(
            await get_consultation(session, consultation_id),
            resource_name="Consultation",
        )
        await delete_consultation(session, consultation)