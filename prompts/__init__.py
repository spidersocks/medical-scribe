# prompts/__init__.py
from .note_generators import standard_note, soap_note, hp_note, consultation_note

PROMPT_REGISTRY = {
    "standard": standard_note,
    "soap": soap_note,
    "hp": hp_note,
    "consultation": consultation_note,
}

def get_prompt_generator(note_type: str):
    """Get the appropriate prompt generator for the specified note type."""
    if note_type not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown note type: {note_type}. Available types: {list(PROMPT_REGISTRY.keys())}")
    return PROMPT_REGISTRY[note_type]