from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models import Patient
from schemas.patient import PatientCreate, PatientUpdate


async def create_patient(session: AsyncSession, payload: PatientCreate) -> Patient:
    patient = Patient(**payload.dict())
    session.add(patient)
    await session.flush()
    await session.refresh(patient)
    return patient


async def get_patient(
    session: AsyncSession,
    patient_id: UUID,
    include_consultations: bool = False,
) -> Optional[Patient]:
    stmt = select(Patient).where(Patient.patient_id == patient_id)
    if include_consultations:
        stmt = stmt.options(selectinload(Patient.consultations))
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_patient_by_email_for_user(
    session: AsyncSession,
    user_id: UUID,
    email: str,
) -> Optional[Patient]:
    stmt = select(Patient).where(
        Patient.user_id == user_id,
        Patient.email == email,
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_patients_for_user(
    session: AsyncSession,
    user_id: UUID,
    limit: int = 100,
    offset: int = 0,
    starred_only: bool = False,
) -> List[Patient]:
    stmt = (
        select(Patient)
        .where(Patient.user_id == user_id)
        .order_by(Patient.full_name.asc())
        .limit(limit)
        .offset(offset)
    )
    if starred_only:
        stmt = stmt.where(Patient.is_starred.is_(True))
    result = await session.execute(stmt)
    return result.scalars().all()


async def update_patient(session: AsyncSession, patient: Patient, payload: PatientUpdate) -> Patient:
    for field, value in payload.dict(exclude_unset=True).items():
        setattr(patient, field, value)
    await session.flush()
    await session.refresh(patient)
    return patient


async def delete_patient(session: AsyncSession, patient: Patient) -> None:
    await session.delete(patient)
    await session.flush()