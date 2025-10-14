from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from repositories import (
    create_patient,
    delete_patient,
    get_patient,
    get_patient_by_email_for_user,
    list_patients_for_user,
    update_patient,
)
from schemas.patient import (
    PatientCreate,
    PatientRead,
    PatientSummary,
    PatientUpdate,
)
from .base import ensure_condition, unwrap_or_raise


class PatientService:
    @staticmethod
    async def create(session: AsyncSession, payload: PatientCreate) -> PatientRead:
        if payload.email:
            existing = await get_patient_by_email_for_user(
                session,
                user_id=payload.user_id,
                email=payload.email,
            )
            ensure_condition(existing is None, "Patient email already exists for this clinician.")
        patient = await create_patient(session, payload)
        return PatientRead.from_orm(patient)

    @staticmethod
    async def get(
        session: AsyncSession,
        patient_id: UUID,
        *,
        include_consultations: bool = False,
    ) -> PatientRead:
        patient = unwrap_or_raise(
            await get_patient(
                session,
                patient_id,
                include_consultations=include_consultations,
            ),
            resource_name="Patient",
        )
        return PatientRead.from_orm(patient)

    @staticmethod
    async def list_for_user(
        session: AsyncSession,
        user_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
        starred_only: bool = False,
        summary: bool = False,
    ) -> List[PatientSummary | PatientRead]:
        patients = await list_patients_for_user(
            session,
            user_id=user_id,
            limit=limit,
            offset=offset,
            starred_only=starred_only,
        )
        if summary:
            return [PatientSummary.from_orm(p) for p in patients]
        return [PatientRead.from_orm(p) for p in patients]

    @staticmethod
    async def update(
        session: AsyncSession,
        patient_id: UUID,
        payload: PatientUpdate,
    ) -> PatientRead:
        patient = unwrap_or_raise(await get_patient(session, patient_id), resource_name="Patient")

        if payload.email and payload.email != patient.email:
            existing = await get_patient_by_email_for_user(
                session,
                user_id=patient.user_id,
                email=payload.email,
            )
            ensure_condition(existing is None, "Patient email already exists for this clinician.")

        patient = await update_patient(session, patient, payload)
        return PatientRead.from_orm(patient)

    @staticmethod
    async def delete(session: AsyncSession, patient_id: UUID) -> None:
        patient = unwrap_or_raise(await get_patient(session, patient_id), resource_name="Patient")
        await delete_patient(session, patient)