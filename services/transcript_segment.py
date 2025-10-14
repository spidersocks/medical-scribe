from __future__ import annotations

from typing import List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from repositories import (
    create_transcript_segment,
    delete_transcript_segment,
    delete_transcript_segments_for_consultation,
    get_consultation,
    get_transcript_segment,
    list_transcript_segments_for_consultation,
    update_transcript_segment,
)
from schemas.transcript_segment import (
    TranscriptSegmentCreate,
    TranscriptSegmentRead,
    TranscriptSegmentUpdate,
)
from .base import unwrap_or_raise


class TranscriptSegmentService:
    @staticmethod
    async def create(session: AsyncSession, payload: TranscriptSegmentCreate) -> TranscriptSegmentRead:
        unwrap_or_raise(
            await get_consultation(session, payload.consultation_id),
            resource_name="Consultation",
        )
        segment = await create_transcript_segment(session, payload)
        return TranscriptSegmentRead.from_orm(segment)

    @staticmethod
    async def get(session: AsyncSession, segment_id: UUID) -> TranscriptSegmentRead:
        segment = unwrap_or_raise(
            await get_transcript_segment(session, segment_id),
            resource_name="TranscriptSegment",
        )
        return TranscriptSegmentRead.from_orm(segment)

    @staticmethod
    async def list_for_consultation(
        session: AsyncSession,
        consultation_id: UUID,
    ) -> List[TranscriptSegmentRead]:
        unwrap_or_raise(
            await get_consultation(session, consultation_id),
            resource_name="Consultation",
        )
        segments = await list_transcript_segments_for_consultation(session, consultation_id)
        return [TranscriptSegmentRead.from_orm(s) for s in segments]

    @staticmethod
    async def update(
        session: AsyncSession,
        segment_id: UUID,
        payload: TranscriptSegmentUpdate,
    ) -> TranscriptSegmentRead:
        segment = unwrap_or_raise(
            await get_transcript_segment(session, segment_id),
            resource_name="TranscriptSegment",
        )
        segment = await update_transcript_segment(session, segment, payload)
        return TranscriptSegmentRead.from_orm(segment)

    @staticmethod
    async def delete(session: AsyncSession, segment_id: UUID) -> None:
        segment = unwrap_or_raise(
            await get_transcript_segment(session, segment_id),
            resource_name="TranscriptSegment",
        )
        await delete_transcript_segment(session, segment)

    @staticmethod
    async def delete_for_consultation(session: AsyncSession, consultation_id: UUID) -> int:
        unwrap_or_raise(
            await get_consultation(session, consultation_id),
            resource_name="Consultation",
        )
        return await delete_transcript_segments_for_consultation(session, consultation_id)