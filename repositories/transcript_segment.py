from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from models import TranscriptSegment
from schemas.transcript_segment import (
    TranscriptSegmentCreate,
    TranscriptSegmentUpdate,
)


async def create_transcript_segment(
    session: AsyncSession,
    payload: TranscriptSegmentCreate,
) -> TranscriptSegment:
    segment = TranscriptSegment(**payload.dict())
    session.add(segment)
    await session.flush()
    await session.refresh(segment)
    return segment


async def get_transcript_segment(
    session: AsyncSession,
    segment_id: UUID,
) -> Optional[TranscriptSegment]:
    result = await session.execute(
        select(TranscriptSegment).where(TranscriptSegment.segment_id == segment_id)
    )
    return result.scalar_one_or_none()


async def list_transcript_segments_for_consultation(
    session: AsyncSession,
    consultation_id: UUID,
) -> List[TranscriptSegment]:
    stmt = (
        select(TranscriptSegment)
        .where(TranscriptSegment.consultation_id == consultation_id)
        .order_by(TranscriptSegment.sequence_number.asc())
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def update_transcript_segment(
    session: AsyncSession,
    segment: TranscriptSegment,
    payload: TranscriptSegmentUpdate,
) -> TranscriptSegment:
    for field, value in payload.dict(exclude_unset=True).items():
        setattr(segment, field, value)
    await session.flush()
    await session.refresh(segment)
    return segment


async def delete_transcript_segment(session: AsyncSession, segment: TranscriptSegment) -> None:
    await session.delete(segment)
    await session.flush()


async def delete_transcript_segments_for_consultation(
    session: AsyncSession,
    consultation_id: UUID,
) -> int:
    stmt = (
        delete(TranscriptSegment)
        .where(TranscriptSegment.consultation_id == consultation_id)
        .execution_options(synchronize_session="fetch")
    )
    result = await session.execute(stmt)
    await session.flush()
    return result.rowcount or 0