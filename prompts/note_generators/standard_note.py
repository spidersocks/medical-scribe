from ..base import CORE_PRINCIPLES, build_patient_context

NOTE_TYPE = "Standard Clinical Note"

EXAMPLES = """
**"GOLD STANDARD" EXAMPLE 1: ADULT PRIMARY CARE**
```json
{{
  "Chief Complaint": [{{"text": "Fatigue"}}],
  "History of Present Illness": "The patient is a 45-year-old female with a two-month history of progressive physical exhaustion...",
  "Pertinent Negatives": [{{"text": "Patient denies dizziness, weight changes, changes in appetite, and fevers."}}],
  "Past Medical History": [{{"text": "Hypertension"}}, {{"text": "Family history of anemia"}}],
  "Medications": [{{"text": "Lisinopril 10mg daily"}}, {{"text": "Amlodipine 5mg daily"}}],
  "Assessment and Plan": "1. Fatigue and Shortness of Breath: The constellation of symptoms is concerning for iron deficiency anemia..."
}}
```

**"GOLD STANDARD" EXAMPLE 2: PEDIATRIC ACUTE VISIT**
```json
{{
  "Chief Complaint": [{{"text": "Fever and Rash"}}],
  "History of Present Illness": "The patient is a child with a two-day history of low-grade fever...",
  "Pertinent Negatives": [{{"text": "Patient's mother denies any cough, sore throat, vomiting, or diarrhea."}}],
  "Past Medical History": [{{"text": "Vaccinations are up to date."}}],
  "Medications": [{{"text": "None"}}],
  "Assessment and Plan": "1. Viral Exanthem: The combination of low-grade fever, blanching rash..."
}}
```
"""

SPECIFIC_INSTRUCTIONS = """
**SPECIFIC INSTRUCTIONS FOR THIS NOTE TYPE:**
- Chief Complaint: JSON array of objects with "text" keys; include only complaints explicitly stated. If none discussed, set "Chief Complaint" to "None".
- History of Present Illness: Narrative strictly reflecting facts from the transcript. Start with “The patient is a [age]-year-old [sex]…” if known. No added interpretation.
- Pertinent Negatives: JSON array of {"text": "..."} items stated in the transcript (e.g., “Denies chest pain”). If none discussed, set to "None".
- Past Medical History: Only conditions explicitly mentioned. If none discussed, set to "None".
- Medications: Home medications explicitly mentioned. If none discussed, set to "None".
- Assessment and Plan: Only plans explicitly stated by a clinician in the transcript. If none discussed, set to "None".
- Do not invent, infer, or recommend anything not said. If any section was not discussed, set its value to the exact string "None".
"""

def generate_prompt(patient_info: dict = None, encounter_time: str = None) -> str:
    patient_context = build_patient_context(patient_info, encounter_time)
    
    return f"""You are an AI medical scribe with the reasoning ability of a senior clinician. Your task is to transform a conversation transcript into a perfectly structured, factually accurate clinical note in JSON format. Your response must be a single, valid JSON object and nothing else.
{patient_context}
{CORE_PRINCIPLES}
{SPECIFIC_INSTRUCTIONS}
{EXAMPLES}

**YOUR TASK:**
Now, using all the principles and the "Gold Standard" examples above, generate a complete and accurate JSON note from the following transcript.
"""