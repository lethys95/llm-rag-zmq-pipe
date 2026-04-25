"""Data models for ZMQ communication."""

from typing import Literal

from pydantic import BaseModel, Field


class DialogueInput(BaseModel):
    """Input model for incoming ZMQ dialogue requests.

    Represents a dialogue message from STT with speaker identification.
    """

    content: str = Field(..., description="The dialogue content/text from STT")
    speaker: str = Field(
        ...,
        description="Identifier for who is speaking (e.g., 'user', 'assistant', 'john')",
    )
    mode: Literal["spoken", "text"] = Field(
        "spoken",
        description="Delivery medium — 'spoken' for voice/TTS, 'text' for written output",
    )
    system_prompt_override: str | None = Field(
        None,
        description="Optional override for the system prompt persona (e.g., 'You are a technical expert')",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"content": "My mother passed away yesterday", "speaker": "user"},
                {
                    "content": "Explain quantum computing",
                    "speaker": "user",
                    "system_prompt_override": "You are a physics professor. Use the following context to answer the user's question.",
                },
            ]
        }
    }


