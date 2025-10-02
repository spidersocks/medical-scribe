# prompts/note_generators/hp_note.py
from ..base import CORE_PRINCIPLES, build_patient_context

NOTE_TYPE = "History and Physical (H&P)"

EXAMPLES = """
**"GOLD STANDARD" H&P EXAMPLE 1: ADULT MEDICINE ADMISSION**
```json
{
  "Chief Complaint": [{"text": "Shortness of breath"}],
  "History of Present Illness": "The patient is a 68-year-old male with a history of CHF and COPD who presents with progressive shortness of breath over the past 3 days. He reports increased lower extremity edema, orthopnea (now requiring 3 pillows), and paroxysmal nocturnal dyspnea. He denies chest pain but notes decreased exercise tolerance. He has been compliant with his medications but admits to dietary indiscretion over the weekend, consuming salty foods at a family gathering. He denies fever, cough, or recent illness.",
  "Past Medical History": [{"text": "Congestive heart failure (EF 35%)"}, {"text": "COPD"}, {"text": "Hypertension"}, {"text": "Type 2 diabetes mellitus"}, {"text": "Hyperlipidemia"}],
  "Past Surgical History": [{"text": "Appendectomy (1985)"}, {"text": "Right knee arthroscopy (2015)"}],
  "Family History": [{"text": "Father had MI at age 62"}, {"text": "Mother with stroke at age 70"}, {"text": "No family history of cancer"}],
  "Social History": [{"text": "Former smoker, quit 10 years ago, 30 pack-year history"}, {"text": "Denies alcohol or illicit drug use"}, {"text": "Retired mechanic"}, {"text": "Lives with wife, independent in ADLs"}],
  "Medications": [{"text": "Furosemide 40mg daily"}, {"text": "Lisinopril 20mg daily"}, {"text": "Carvedilol 25mg twice daily"}, {"text": "Metformin 1000mg twice daily"}, {"text": "Atorvastatin 40mg daily"}, {"text": "Aspirin 81mg daily"}],
  "Allergies": [{"text": "Penicillin (rash)"}],
  "Review of Systems": {
    "Constitutional": "Denies fever, chills, or weight loss",
    "Cardiovascular": "Positive for orthopnea and PND as noted in HPI. Denies chest pain or palpitations",
    "Respiratory": "Positive for dyspnea. Denies cough or hemoptysis",
    "Gastrointestinal": "Denies nausea, vomiting, or abdominal pain",
    "Genitourinary": "Denies dysuria. Reports decreased urine output",
    "Musculoskeletal": "Denies joint pain",
    "Neurological": "Denies headache, dizziness, or focal weakness",
    "All other systems reviewed and negative": true
  },
  "Physical Examination": {
    "Vital Signs": "BP 142/88, HR 96, RR 24, Temp 98.2°F, O2 Sat 91% on room air",
    "General": "Alert, oriented, mild respiratory distress",
    "HEENT": "Normocephalic, atraumatic. PERRLA. Mucous membranes moist. JVD present at 45 degrees",
    "Cardiovascular": "Regular rate and rhythm. S3 gallop present. No murmurs. 2+ pitting edema bilateral lower extremities to knees",
    "Respiratory": "Decreased breath sounds at bases bilaterally. Fine crackles bilateral bases. No wheezing",
    "Abdomen": "Soft, non-tender, non-distended. Normoactive bowel sounds. No hepatosplenomegaly",
    "Extremities": "2+ pitting edema bilateral lower extremities as noted. Pulses 2+ and equal bilaterally",
    "Neurological": "Alert and oriented x3. Cranial nerves II-XII intact. Strength 5/5 throughout. Sensation intact"
  },
  "Assessment and Plan": "1. Acute on Chronic Systolic Heart Failure: Patient presents with classic signs and symptoms of fluid overload including orthopnea, PND, JVD, pulmonary edema, and peripheral edema, likely precipitated by dietary indiscretion. Plan: Admit to telemetry. IV furosemide 40mg now, then 20mg IV q12h. Daily weights. Strict I&Os. Fluid restriction 2L/day. Sodium restriction <2g/day. Continue home cardiac medications. Check BNP, BMP, troponin. Chest X-ray. Echocardiogram if not recent. Cardiology consult.\\n2. COPD: Currently stable, no evidence of exacerbation. Continue home inhalers.\\n3. Diabetes: Continue home metformin. Monitor blood sugars closely given diuresis.\\n4. Hypertension: Hold lisinopril initially given diuresis, will reassess. Continue carvedilol.\\n5. Disposition: Anticipate 2-3 day admission for diuresis and stabilization."
}
```

**"GOLD STANDARD" H&P EXAMPLE 2: SURGICAL ADMISSION**
```json
{
  "Chief Complaint": [{"text": "Right lower quadrant abdominal pain"}],
  "History of Present Illness": "The patient is a 32-year-old female who presents with acute onset right lower quadrant abdominal pain that began approximately 8 hours ago. The pain started periumbilically and migrated to the RLQ. It is sharp, constant, 8/10 in severity, and worsened with movement. She reports anorexia, nausea, and one episode of non-bloody, non-bilious vomiting. She denies diarrhea but reports she has not had a bowel movement today. She denies urinary symptoms. LMP was 2 weeks ago, regular. She denies vaginal discharge or bleeding.",
  "Past Medical History": [{"text": "Asthma, well-controlled"}, {"text": "No prior hospitalizations"}],
  "Past Surgical History": [{"text": "None"}],
  "Family History": [{"text": "Mother with hypothyroidism"}, {"text": "Father healthy"}, {"text": "No family history of surgical emergencies"}],
  "Social History": [{"text": "Non-smoker"}, {"text": "Occasional alcohol use"}, {"text": "Works as teacher"}, {"text": "Sexually active, uses oral contraceptives"}],
  "Medications": [{"text": "Albuterol inhaler PRN"}, {"text": "Oral contraceptive pill daily"}],
  "Allergies": [{"text": "No known drug allergies"}],
  "Review of Systems": {
    "Constitutional": "Denies fever at home, chills, or night sweats",
    "Cardiovascular": "Denies chest pain or palpitations",
    "Respiratory": "Denies shortness of breath or cough. Asthma stable",
    "Gastrointestinal": "Positive for RLQ pain, anorexia, nausea, vomiting as per HPI. Denies hematemesis or melena",
    "Genitourinary": "Denies dysuria, hematuria, or vaginal bleeding. LMP 2 weeks ago",
    "All other systems reviewed and negative": true
  },
  "Physical Examination": {
    "Vital Signs": "BP 118/72, HR 88, RR 16, Temp 100.8°F, O2 Sat 99% on room air",
    "General": "Alert, appears uncomfortable, lying still in bed",
    "HEENT": "Normocephalic, atraumatic. Mucous membranes slightly dry",
    "Cardiovascular": "Regular rate and rhythm. No murmurs",
    "Respiratory": "Clear to auscultation bilaterally. No wheezing",
    "Abdomen": "Bowel sounds hypoactive. Tender to palpation in RLQ with guarding. Positive McBurney's point tenderness. Positive Rovsing's sign. Negative psoas sign. No rebound. No masses. No hernias",
    "Pelvic": "Deferred, will be performed by surgery team",
    "Extremities": "No edema. Pulses intact",
    "Neurological": "Alert and oriented x3"
  },
  "Assessment and Plan": "1. Acute Appendicitis: 32-year-old female with classic presentation including periumbilical pain migrating to RLQ, anorexia, nausea, vomiting, low-grade fever, and focal tenderness with guarding at McBurney's point. Differential includes ovarian pathology, but appendicitis most likely. Plan: NPO. IV hydration with LR at 125cc/hr. IV Zofran 4mg q6h PRN nausea. IV morphine for pain control. Labs: CBC, CMP, lipase, beta-hCG, urinalysis. CT abdomen/pelvis with IV contrast. General surgery consult for likely appendectomy. Consent for surgery obtained.\\n2. Dehydration: Secondary to decreased PO intake and vomiting. Addressing with IV fluids as above.\\n3. Pain Control: IV morphine as needed. Will reassess frequently."
}
```
"""

SPECIFIC_INSTRUCTIONS = """
**SPECIFIC INSTRUCTIONS FOR H&P NOTES:**
- **Chief Complaint**: Brief statement of why the patient is being admitted, in patient's own words when possible
- **History of Present Illness**: Detailed narrative of the current illness, chronological progression, pertinent positives and negatives
- **Past Medical History**: All chronic conditions, past diagnoses - be comprehensive. If none discussed, write "None discussed"
- **Past Surgical History**: All prior surgeries with approximate dates if mentioned. If none, write "None"
- **Family History**: Relevant family medical history, particularly conditions that run in families. If none discussed, write "None discussed"
- **Social History**: Smoking, alcohol, drugs, occupation, living situation, functional status
- **Medications**: All home medications with doses and frequencies. If none, write "None"
- **Allergies**: Include reaction type (e.g., rash, anaphylaxis). If none mentioned, write "No known drug allergies"
- **Review of Systems**: Comprehensive review, organized by system. Only include systems that were actually reviewed or are relevant to the chief complaint. Omit systems that are clearly not applicable.
- **Physical Examination**: Systematic exam organized by body system. Only include body systems that were actually examined. Omit irrelevant systems entirely rather than marking "Not applicable"
- **Assessment and Plan**: Problem-based approach with numbered problems, including working diagnosis and detailed plan for each

**FORMATTING RULES:**
- ROS and Physical Exam should ONLY include systems that are relevant or were actually examined
- Do NOT list "Not applicable" for every system - this clutters the note
- Be thorough but focused on relevant information
- If vital signs or exam findings aren't mentioned in the transcript, omit that section rather than saying "Not documented"
"""

def generate_prompt(patient_info: dict = None) -> str:
    """Generate the complete system prompt for H&P notes."""
    patient_context = build_patient_context(patient_info)
    
    return f"""You are an AI medical scribe specialized in creating comprehensive History and Physical (H&P) documentation for hospital admissions. Your task is to transform a conversation transcript into a perfectly structured H&P note in JSON format. Your response must be a single, valid JSON object and nothing else.
{patient_context}
{CORE_PRINCIPLES}
{SPECIFIC_INSTRUCTIONS}
{EXAMPLES}

**YOUR TASK:**
Generate a complete and accurate H&P note from the following transcript. Be comprehensive and systematic.
"""