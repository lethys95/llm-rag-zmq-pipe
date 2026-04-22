from pydantic import BaseModel, Field, field_validator


class UserFact(BaseModel):
    """An atomic fact about the user extracted from a message.

    Stored long-term in Qdrant as a separate vector point per fact.
    Only directly stated or strongly implied facts — no loose inference.
    """

    claim: str = Field(..., description="Phrased as 'user [verb] [subject]'")
    sentiment: str = Field(..., description="positive / negative / neutral")
    confidence: float = Field(..., ge=0.0, le=1.0)
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

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v: str) -> str:
        allowed = {"positive", "negative", "neutral"}
        if v not in allowed:
            raise ValueError(f"sentiment must be one of {allowed}, got '{v}'")
        return v
