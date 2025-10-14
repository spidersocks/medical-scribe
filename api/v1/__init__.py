from fastapi import APIRouter

from . import (
    clinical_notes,
    consultations,
    patients,
    transcript_segments,
    users,
)

api_router = APIRouter()

api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(patients.router, prefix="/patients", tags=["patients"])
api_router.include_router(consultations.router, prefix="/consultations", tags=["consultations"])
api_router.include_router(transcript_segments.router, prefix="/transcript-segments", tags=["transcript-segments"])
api_router.include_router(clinical_notes.router, prefix="/clinical-notes", tags=["clinical-notes"])