# prompts/base.py
from datetime import datetime, timezone

def _format_encounter_time(encounter_time: str) -> str:
    """
    Convert an ISO 8601 timestamp (e.g., 2025-11-13T12:04:58.841Z) into a human-readable string.
    Falls back to the original string on parse errors.
    """
    try:
        dt = datetime.fromisoformat(encounter_time.replace("Z", "+00:00"))
        dt_utc = dt.astimezone(timezone.utc)
        # Example: "November 13, 2025 at 12:04 UTC"
        return f"{dt_utc.strftime('%B %d, %Y at %H:%M')} UTC"
    except Exception:
        return encounter_time

def build_patient_context(patient_info: dict, encounter_time: str = None) -> str:
    """Build the patient information context block used across all note types."""
    if not patient_info and not encounter_time:
        return ""
    
    context_parts = []
    if patient_info:
        if patient_info.get("age") and patient_info.get("sex"):
            context_parts.append(f"Age/Sex: {patient_info['age']}-year-old {patient_info['sex']}")
        elif patient_info.get("age"):
            context_parts.append(f"Age: {patient_info['age']} years old")
        elif patient_info.get("sex"):
            context_parts.append(f"Sex: {patient_info['sex']}")
        if patient_info.get("referring_physician"):
            context_parts.append(f"Referring Physician: {patient_info['referring_physician']}")
        if patient_info.get("additional_context"):
            context_parts.append(f"Additional Context: {patient_info['additional_context']}")
    if encounter_time:
        context_parts.append(f"Encounter Time: {_format_encounter_time(encounter_time)}")
    
    if context_parts:
        return "\n\n**PATIENT INFORMATION:**\n" + "\n".join(context_parts)
    return ""

CORE_PRINCIPLES = """
**CORE PRINCIPLES:**
1. **DO NOT COPY FROM THE EXAMPLES:** The examples are for structure and style ONLY. Every single fact in your output must come from the transcript.
2. **USE PROVIDED PATIENT INFORMATION:** If patient information is provided, use it appropriately in your note.
3. **Fact-Check for Consistency:** Re-read the transcript and your note to ensure consistency.
4. **CRITICAL: VALID JSON SYNTAX:** Ensure all JSON is perfectly formatted with proper quotes around all keys and string values.
5. If there is truly insufficient medical information in the transcript, do NOT fabricate content. Instead, respond with a valid JSON object: {"error": "insufficient_medical_information"}. Do not produce any additional dialogue, clinical details, or roles that are not present in the transcript.
6. **SCRIBE-ONLY ROLE (NO CLINICAL ADVICE):** You are NOT a clinician. Do not diagnose, prescribe, recommend, or add clinical advice. Your sole job is to faithfully summarize what was explicitly said in the transcript.
7. **IF A SECTION WAS NOT DISCUSSED, SET IT TO "None":** For any section where the transcript provides no explicit information, set the value to the string "None" (without quotes). Do not infer, speculate, or fill with plausible details.
8. **ONLY RECORD EXPLICIT RECOMMENDATIONS/PLANS:** Include assessments, recommendations, or plans only if they were explicitly stated in the transcript. Otherwise, use "None".
"""

# --- Add built-in note key lists here ---
BUILTIN_NOTE_KEYS = {
    "standard": [
        "Chief Complaint",
        "History of Present Illness",
        "Pertinent Negatives",
        "Past Medical History",
        "Medications",
        "Assessment and Plan",
    ],
    "soap": [
        "Subjective",
        "Objective",
        "Assessment",
        "Plan",
    ],
    "hp": [
        "Chief Complaint",
        "History of Present Illness",
        "Past Medical History",
        "Past Surgical History",
        "Family History",
        "Social History",
        "Medications",
        "Allergies",
        "Review of Systems",
        "Physical Examination",
        "Assessment and Plan",
    ],
    "consultation": [
        "Consultation Request",
        "History of Present Illness",
        "Past Medical History",
        "Past Surgical History",
        "Family History",
        "Social History",
        "Current Medications",
        "Allergies",
        "Pertinent Physical Examination",
        "Diagnostic Studies Reviewed",
        "Assessment",
        "Recommendations",
    ],
}