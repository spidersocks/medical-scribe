# prompts/base.py

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
        context_parts.append(f"Encounter Time: {encounter_time}")
    
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
        "Past Surgical History",   # Sometimes optional, depending on prompt
        "Family History",
        "Social History",
        "Current Medications",     # or "Medications"
        "Allergies",
        "Pertinent Physical Examination",
        "Diagnostic Studies Reviewed",
        "Assessment",
        "Recommendations",
    ],
}