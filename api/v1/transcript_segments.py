from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import guard_service, get_session
from schemas.transcript_segment import (
    TranscriptSegmentCreate,
    TranscriptSegmentRead,
    TranscriptSegmentUpdate,
)
from services import TranscriptSegmentService

router = APIRouter()


@router.get(
    "/consultations/{consultation_id}/segments",
    response_model=List[TranscriptSegmentRead],
)
async def list_segments_for_consultation(
    consultation_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> List[TranscriptSegmentRead]:
    return await guard_service(
        TranscriptSegmentService.list_for_consultation(session, consultation_id)
    )


@router.post(
    "/consultations/{consultation_id}/segments",
    response_model=TranscriptSegmentRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_segment_for_consultation(
    consultation_id: UUID,
    payload: TranscriptSegmentCreate,
    session: AsyncSession = Depends(get_session),
) -> TranscriptSegmentRead:
    payload = payload.copy(update={"consultation_id": consultation_id})
    return await guard_service(TranscriptSegmentService.create(session, payload))


@router.get("/segments/{segment_id}", response_model=TranscriptSegmentRead)
async def get_segment(
    segment_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> TranscriptSegmentRead:
    return await guard_service(TranscriptSegmentService.get(session, segment_id))


@router.patch("/segments/{segment_id}", response_model=TranscriptSegmentRead)
async def update_segment(
    segment_id: UUID,
    payload: TranscriptSegmentUpdate,
    session: AsyncSession = Depends(get_session),
) -> TranscriptSegmentRead:
    return await guard_service(
        TranscriptSegmentService.update(session, segment_id, payload)
    )


@router.delete("/segments/{segment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_segment(
    segment_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> Response:
    await guard_service(TranscriptSegmentService.delete(session, segment_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)