"""Data models for sentiment analysis and ZMQ communication."""

from pydantic import BaseModel, Field, field_validator


class DialogueInput(BaseModel):
    """Input model for incoming ZMQ dialogue requests.
    
    Represents a dialogue message from STT with speaker identification.
    """
    
    content: str = Field(..., description="The dialogue content/text from STT")
    speaker: str = Field(..., description="Identifier for who is speaking (e.g., 'user', 'assistant', 'john')")
    system_prompt_override: str | None = Field(
        None, 
        description="Optional override for the system prompt persona (e.g., 'You are a technical expert')"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "content": "My mother passed away yesterday",
                    "speaker": "user"
                },
                {
                    "content": "Explain quantum computing",
                    "speaker": "user",
                    "system_prompt_override": "You are a physics professor. Use the following context to answer the user's question."
                }
            ]
        }
    }


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result model.
    
    This model represents the structured output from sentiment analysis,
    designed to be JSON-serializable for storage and further processing.
    """
    
    sentiment: str = Field(..., description="Sentiment classification (positive, negative, neutral)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    memory_owner: str = Field(..., description="Identifier for who this memory belongs to (speaker from input)")
    emotional_tone: str | None = Field(None, description="Emotional tone (e.g., happy, angry, sad, confused)")
    relevance: float | None = Field(None, ge=0.0, le=1.0, description="General impact/importance (0.0 to 1.0)")
    chrono_relevance: float | None = Field(None, ge=0.0, le=1.0, description="How quickly this decreases in relevance over time (0.0 to 1.0)")
    context_summary: str | None = Field(None, description="Brief description of the specific situation")
    key_topics: list[str] | None = Field(None, description="Main topics identified in the message")
    
    @field_validator('sentiment')
    @classmethod
    def validate_sentiment(cls, v: str) -> str:
        """Validate sentiment is one of the allowed values."""
        valid_sentiments = ["positive", "negative", "neutral"]
        if v not in valid_sentiments:
            raise ValueError(f"Sentiment must be one of {valid_sentiments}, got: {v}")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sentiment": "negative",
                    "confidence": 0.95,
                    "memory_owner": "user",
                    "emotional_tone": "grieving",
                    "relevance": 1.0,
                    "chrono_relevance": 0.3,
                    "context_summary": "user's mother died",
                    "key_topics": ["family", "death", "grief"]
                }
            ]
        }
    }
