from __future__ import annotations

import asyncio
from typing import List, Dict, Any

from fastapi import APIRouter, Response, status, BackgroundTasks

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
    consultation_id: str,
    include_entities: bool = False,
) -> List[TranscriptSegmentRead]:
    return await guard_service(
        transcript_segment_service.list_for_consultation(
            consultation_id, include_entities=include_entities
        )
    )


@router.post(
    "/consultations/{consultation_id}/segments",
    response_model=TranscriptSegmentRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_segment_for_consultation(
    consultation_id: str,
    payload: TranscriptSegmentCreate,
    background_tasks: BackgroundTasks,
) -> TranscriptSegmentRead:
    print("[transcript_segments] POST create called", {
        "consultation_id": consultation_id,
        "payload": payload.model_dump()
    })
    payload_with_consultation = payload.copy(update={"consultation_id": consultation_id})
    
    new_segment = await guard_service(
        transcript_segment_service.create(payload_with_consultation)
    )

    # --- AUTO-DIARIZATION TRIGGER ---
    # Every 5 segments, trigger a background diarization job.
    # This ensures "live" updates without frontend polling changes.
    if new_segment.sequence_number > 0 and new_segment.sequence_number % 5 == 0:
        print(f"[Auto-Trigger] Queueing diarization for {consultation_id} at seq {new_segment.sequence_number}")
        background_tasks.add_task(transcript_segment_service.diarize_consultation, consultation_id)

    return new_segment


@router.post(
    "/consultations/{consultation_id}/enrich",
    status_code=status.HTTP_202_ACCEPTED,
)
async def enrich_segments_for_consultation(
    consultation_id: str,
    force: bool = False,
) -> dict:
    return await guard_service(
        transcript_segment_service.enrich_consultation(consultation_id, force=force)
    )


@router.post(
    "/consultations/{consultation_id}/diarize",
    status_code=status.HTTP_200_OK,
)
async def diarize_consultation_segments(
    consultation_id: str,
) -> Dict[str, Any]:
    """
    Trigger semantic diarization (speaker role assignment) for recent segments
    using Qwen LLM. Useful for periodic updates during live sessions.
    """
    return await guard_service(
        transcript_segment_service.diarize_consultation(consultation_id)
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