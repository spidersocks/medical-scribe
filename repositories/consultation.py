from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models import Consultation
from schemas.consultation import ConsultationCreate, ConsultationUpdate


async def create_consultation(
    session: AsyncSession,
    payload: ConsultationCreate,
) -> Consultation:
    consultation = Consultation(**payload.model_dump())
    session.add(consultation)
    await session.flush()

    stmt = (
        select(Consultation)
        .options(
            selectinload(Consultation.patient),
            selectinload(Consultation.transcript_segments),
            selectinload(Consultation.clinical_note),
        )
        .where(Consultation.consultation_id == consultation.consultation_id)
    )
    result = await session.execute(stmt)
    return result.scalar_one()


async def get_consultation(
    session: AsyncSession,
    consultation_id: UUID,
    include_patient: bool = False,
    include_segments: bool = False,
    include_clinical_note: bool = False,
) -> Optional[Consultation]:
    stmt = select(Consultation).where(Consultation.consultation_id == consultation_id)

    if include_patient:
        stmt = stmt.options(selectinload(Consultation.patient))
    if include_segments:
        stmt = stmt.options(selectinload(Consultation.transcript_segments))
    if include_clinical_note:
        stmt = stmt.options(selectinload(Consultation.clinical_note))

    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_consultations_for_user(
    session: AsyncSession,
    user_id: UUID,
    patient_id: Optional[UUID] = None,
    limit: int = 100,
    offset: int = 0,
    include_patient: bool = False,
    include_segments: bool = False,
    include_clinical_note: bool = False,
) -> List[Consultation]:
    stmt = (
        select(Consultation)
        .where(Consultation.user_id == user_id)
        .order_by(Consultation.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if patient_id:
        stmt = stmt.where(Consultation.patient_id == patient_id)
    if include_patient:
        stmt = stmt.options(selectinload(Consultation.patient))
    if include_segments:
        stmt = stmt.options(selectinload(Consultation.transcript_segments))
    if include_clinical_note:
        stmt = stmt.options(selectinload(Consultation.clinical_note))

    result = await session.execute(stmt)
    return result.scalars().all()


async def update_consultation(
    session: AsyncSession,
    consultation: Consultation,
    payload: ConsultationUpdate,
) -> Consultation:
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(consultation, field, value)

    await session.flush()

    stmt = (
        select(Consultation)
        .options(
            selectinload(Consultation.patient),
            selectinload(Consultation.transcript_segments),
            selectinload(Consultation.clinical_note),
        )
        .where(Consultation.consultation_id == consultation.consultation_id)
    )
    result = await session.execute(stmt)
    return result.scalar_one()


async def delete_consultation(session: AsyncSession, consultation: Consultation) -> None:
    await session.delete(consultation)
    await session.flush()