from fastapi import APIRouter

from . import (
    clinical_notes,
    consultations,
    patients,
    transcript_segments,
    users,
)

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(patients.router, prefix="/patients", tags=["patients"])
api_router.include_router(consultations.router, prefix="/consultations", tags=["consultations"])
api_router.include_router(transcript_segments.router, tags=["transcript-segments"])
api_router.include_router(clinical_notes.router, tags=["clinical-notes"])