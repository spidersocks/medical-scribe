from __future__ import annotations

from typing import List
# from uuid import UUID  # no longer needed for path param

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
async def list_segments_for_consultation(
    consultation_id: str,  # accept any string id to support non-UUID consultations
    include_entities: bool = False,  # optional: future enrichment knob
) -> List[TranscriptSegmentRead]:
    # We simply pass the string id through; the service already compares as string.
    segments = await guard_service(
        transcript_segment_service.list_for_consultation(consultation_id)
    )
    # If you later add on-demand enrichment, do it here when include_entities is True.
    return segments


@router.post(
    "/consultations/{consultation_id}/segments",
    response_model=TranscriptSegmentRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_segment_for_consultation(
    consultation_id: str,
    payload: TranscriptSegmentCreate,
) -> TranscriptSegmentRead:
    # DEBUG LOG
    print("[transcript_segments] POST create called", {
        "consultation_id": consultation_id,
        "payload": payload.model_dump()
    })
    payload_with_consultation = payload.copy(update={"consultation_id": consultation_id})
    return await guard_service(
        transcript_segment_service.create(payload_with_consultation)
    )

@router.get("/segments/{segment_id}", response_model=TranscriptSegmentRead)
async def get_segment(segment_id: str) -> TranscriptSegmentRead:
    return await guard_service(transcript_segment_service.get(segment_id))


@router.patch("/segments/{segment_id}", response_model=TranscriptSegmentRead)
async def update_segment(
    segment_id: str,
    payload: TranscriptSegmentUpdate,
) -> TranscriptSegmentRead:
    return await guard_service(
        transcript_segment_service.update(segment_id, payload)
    )


@router.delete("/segments/{segment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_segment(segment_id: str) -> Response:
    await guard_service(transcript_segment_service.delete(segment_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)