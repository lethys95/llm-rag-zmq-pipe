from pydantic import BaseModel, Field


class EmotionalState(BaseModel):
    """Dimensional emotional state of the user at the time of a message.

    Session-scoped — informs response strategy and crisis detection.
    Not stored long-term in Qdrant.

    All emotion scores default to 0.0. Only raised emotions are present.

    VAD model:
        valence   — how positive or negative  (-1.0 to 1.0)
        arousal   — intensity of activation    (0.0 = calm, 1.0 = intense)
        dominance — sense of control           (0.0 = powerless, 1.0 = in control)

    confidence — how reliably this can be scored from text alone.
        High when emotional language is explicit. Low for short/ambiguous messages.
    """

    # --- Emotion scores (0.0–1.0) ---
    joy: float = Field(0.0, ge=0.0, le=1.0)
    sadness: float = Field(0.0, ge=0.0, le=1.0)
    grief: float = Field(0.0, ge=0.0, le=1.0)
    anger: float = Field(0.0, ge=0.0, le=1.0)
    frustration: float = Field(0.0, ge=0.0, le=1.0)
    fear: float = Field(0.0, ge=0.0, le=1.0)
    anxiety: float = Field(0.0, ge=0.0, le=1.0)
    disgust: float = Field(0.0, ge=0.0, le=1.0)
    guilt: float = Field(0.0, ge=0.0, le=1.0)
    shame: float = Field(0.0, ge=0.0, le=1.0)
    loneliness: float = Field(0.0, ge=0.0, le=1.0)
    overwhelm: float = Field(0.0, ge=0.0, le=1.0)
    contentment: float = Field(0.0, ge=0.0, le=1.0)
    confusion: float = Field(0.0, ge=0.0, le=1.0)

    # --- VAD dimensions ---
    valence: float = Field(..., ge=-1.0, le=1.0)
    arousal: float = Field(..., ge=0.0, le=1.0)
    dominance: float = Field(..., ge=0.0, le=1.0)

    # --- Meta ---
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str | None = None
