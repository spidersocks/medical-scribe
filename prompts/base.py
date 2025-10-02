# prompts/base.py

def build_patient_context(patient_info: dict) -> str:
    """Build the patient information context block used across all note types."""
    if not patient_info:
        return ""
    
    context_parts = []
    if patient_info.get("name"):
        context_parts.append(f"Patient Name: {patient_info['name']}")
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
    
    if context_parts:
        return "\n\n**PATIENT INFORMATION:**\n" + "\n".join(context_parts)
    return ""

CORE_PRINCIPLES = """
**CORE PRINCIPLES:**
1. **DO NOT COPY FROM THE EXAMPLES:** The examples are for structure and style ONLY. Every single fact in your output must come from the transcript.
2. **USE PROVIDED PATIENT INFORMATION:** If patient information is provided, use it appropriately in your note.
3. **Fact-Check for Consistency:** Re-read the transcript and your note to ensure consistency.
4. **CRITICAL: VALID JSON SYNTAX:** Ensure all JSON is perfectly formatted with proper quotes around all keys and string values.
"""