from pydantic import BaseModel, Field, field_validator


VALID_APPROACHES = {
    "reflective_listening",
    "socratic_questioning",
    "cognitive_reframing",
    "behavioral_activation",
    "acceptance_and_validation",
    "meaning_making",
    "practical_problem_solving",
}

VALID_TONES = {
    "empathetic_warm",
    "curious_gentle",
    "grounding_steady",
    "playful_light",
    "direct_honest",
}


class ResponseStrategy(BaseModel):
    """Selected therapeutic approach for the current response.

    system_prompt_addition is injected into the primary LLM's system prompt.
    reasoning is kept for debugging and detox review — not shown to the user.
    """

    approach: str = Field(..., description="Therapeutic approach to use")
    tone: str = Field(..., description="Conversational tone")
    needs_focus: list[str] = Field(default_factory=list, description="Need categories being addressed")
    system_prompt_addition: str = Field(..., description="Injected into primary response system prompt")
    reasoning: str = Field(..., description="Why this approach was selected — for detox review")

    @field_validator("approach")
    @classmethod
    def validate_approach(cls, v: str) -> str:
        if v not in VALID_APPROACHES:
            raise ValueError(f"approach must be one of {VALID_APPROACHES}, got '{v}'")
        return v

    @field_validator("tone")
    @classmethod
    def validate_tone(cls, v: str) -> str:
        if v not in VALID_TONES:
            raise ValueError(f"tone must be one of {VALID_TONES}, got '{v}'")
        return v
