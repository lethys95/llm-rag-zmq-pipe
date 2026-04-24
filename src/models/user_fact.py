from pydantic import BaseModel, Field


class UserFact(BaseModel):
    """An atomic fact about the user extracted from a message.

    Stored long-term in Qdrant as a separate vector point per fact.
    Only directly stated or strongly implied facts — no loose inference.

    VAD scores mirror the emotional register of the fact itself:
      valence   — emotional tone, -1.0 (very negative) to 1.0 (very positive)
      arousal   — intensity/activation, 0.0 (calm) to 1.0 (intense)
      dominance — sense of control, 0.0 (powerless) to 1.0 (in control)
    """

    claim: str = Field(..., description="Phrased as 'user [verb] [subject]'")
    valence: float | None = Field(None, ge=-1.0, le=1.0, description="Turn-level valence when this fact was stated")
    arousal: float | None = Field(None, ge=0.0, le=1.0, description="Turn-level arousal when this fact was stated")
    dominance: float | None = Field(None, ge=0.0, le=1.0, description="Turn-level dominance when this fact was stated")
    chrono_relevance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How stable/long-lasting is this fact. "
            "0.9 = stable preference or life event. "
            "0.5 = moderate. "
            "0.1 = ephemeral."
        ),
    )
    subject: str = Field(..., description="Category, e.g. 'food preferences', 'relationships'")
    memory_owner: str = Field(..., description="Speaker this fact belongs to")
