"""Analysis models for AI-driven evaluation and assessment."""

from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator


@dataclass
class MemoryEvaluation:
    """Result of AI memory evaluation."""

    relevance: float
    chrono_relevance: float
    reasoning: str
    should_boost: bool
    boost_factor: float


class NeedsAnalysis(BaseModel):
    """Phasic assessment of the user's active psychological needs.

    Uses Maslow's hierarchy as dimensional scoring — not a strict hierarchy
    but a set of need categories scored by how activated they are right now.

    need_urgency   — how pressing the need is in this moment
    need_persistence — proxy for how chronic/tonic this pattern is.
                       Currently LLM-assessed per turn; should eventually be
                       computed from recurrence across sessions.
    """

    physiological: float = Field(0.0, ge=0.0, le=1.0)
    safety: float = Field(0.0, ge=0.0, le=1.0)
    belonging: float = Field(0.0, ge=0.0, le=1.0)
    esteem: float = Field(0.0, ge=0.0, le=1.0)
    autonomy: float = Field(0.0, ge=0.0, le=1.0)
    meaning: float = Field(0.0, ge=0.0, le=1.0)
    growth: float = Field(0.0, ge=0.0, le=1.0)

    primary_needs: list[str] = Field(default_factory=list)
    unmet_needs: list[str] = Field(default_factory=list)

    need_urgency: float = Field(..., ge=0.0, le=1.0)
    need_persistence: float = Field(..., ge=0.0, le=1.0)

    context_summary: str

    memory_owner: str = ""

    @field_validator("primary_needs", "unmet_needs")
    @classmethod
    def validate_need_names(cls, v: list[str]) -> list[str]:
        valid = {"physiological", "safety", "belonging", "esteem", "autonomy", "meaning", "growth"}
        for name in v:
            if name not in valid:
                raise ValueError(f"Unknown need '{name}'. Must be one of {valid}")
        return v
