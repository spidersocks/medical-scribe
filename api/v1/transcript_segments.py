from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Response, status

from api.deps import guard_service
from schemas.transcript_segment import (
    TranscriptSegmentCreate,
    TranscriptSegmentRead,
    TranscriptSegmentUpdate,
)
from services import transcript_segment_service

router = APIRouter()


@router.get(
    "/consultations/{consultation_id}/segments",
    response_model=List[TranscriptSegmentRead],
)
async def list_segments_for_consultation(consultation_id: UUID) -> List[TranscriptSegmentRead]:
    return await guard_service(
        transcript_segment_service.list_for_consultation(str(consultation_id))
    )


@router.post(
    "/consultations/{consultation_id}/segments",
    response_model=TranscriptSegmentRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_segment_for_consultation(
    consultation_id: UUID,
    payload: TranscriptSegmentCreate,
) -> TranscriptSegmentRead:
    payload_with_consultation = payload.copy(update={"consultation_id": str(consultation_id)})
    return await guard_service(
        transcript_segment_service.create(payload_with_consultation)
    )


@router.get("/segments/{segment_id}", response_model=TranscriptSegmentRead)
async def get_segment(segment_id: UUID) -> TranscriptSegmentRead:
    return await guard_service(transcript_segment_service.get(str(segment_id)))


@router.patch("/segments/{segment_id}", response_model=TranscriptSegmentRead)
async def update_segment(
    segment_id: UUID,
    payload: TranscriptSegmentUpdate,
) -> TranscriptSegmentRead:
    return await guard_service(
        transcript_segment_service.update(str(segment_id), payload)
    )


@router.delete("/segments/{segment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_segment(segment_id: UUID) -> Response:
    await guard_service(transcript_segment_service.delete(str(segment_id)))
    return Response(status_code=status.HTTP_204_NO_CONTENT)