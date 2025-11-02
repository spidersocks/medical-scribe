# prompts/note_generators/standard_note.py
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
- Begin HPI with "The patient is a [age]-year-old [gender]..." if demographics are known
- If patient name is provided in patient information, you may include it naturally in the HPI
- Pertinent Negatives must be a JSON array of objects with "text" keys
- Past Medical History should include all conditions mentioned, even minor ones
- If no past medical history is discussed, use "No past medical history discussed" rather than "None"
- Medications are pre-existing home medications only
- Assessment and Plan should group related items under single problem numbers
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