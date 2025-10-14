from .user import (
    create_user,
    delete_user,
    get_user,
    get_user_by_email,
    list_users,
    update_user,
)
from .patient import (
    create_patient,
    delete_patient,
    get_patient,
    get_patient_by_email_for_user,
    list_patients_for_user,
    update_patient,
)
from .consultation import (
    create_consultation,
    delete_consultation,
    get_consultation,
    list_consultations_for_user,
    update_consultation,
)
from .transcript_segment import (
    create_transcript_segment,
    delete_transcript_segment,
    delete_transcript_segments_for_consultation,
    get_transcript_segment,
    list_transcript_segments_for_consultation,
    update_transcript_segment,
)
from .clinical_note import (
    create_clinical_note,
    delete_clinical_note,
    get_clinical_note_by_consultation,
    get_clinical_note,
    update_clinical_note,
)

__all__ = [
    # users
    "create_user",
    "delete_user",
    "get_user",
    "get_user_by_email",
    "list_users",
    "update_user",
    # patients
    "create_patient",
    "delete_patient",
    "get_patient",
    "get_patient_by_email_for_user",
    "list_patients_for_user",
    "update_patient",
    # consultations
    "create_consultation",
    "delete_consultation",
    "get_consultation",
    "list_consultations_for_user",
    "update_consultation",
    # transcript segments
    "create_transcript_segment",
    "delete_transcript_segment",
    "delete_transcript_segments_for_consultation",
    "get_transcript_segment",
    "list_transcript_segments_for_consultation",
    "update_transcript_segment",
    # clinical notes
    "create_clinical_note",
    "delete_clinical_note",
    "get_clinical_note_by_consultation",
    "get_clinical_note",
    "update_clinical_note",
]