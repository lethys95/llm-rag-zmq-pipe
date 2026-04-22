"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    """Represents a function/tool call from an LLM."""

    function_name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass
class FunctionParameters:
    """Parameters definition for a function tool (JSON Schema)."""

    type: str
    properties: dict[str, Any]
    required: list[str]


@dataclass
class FunctionDefinition:
    """Function definition for tool calling."""

    name: str
    description: str
    parameters: FunctionParameters


@dataclass
class ToolDefinition:
    """Tool definition following OpenAI format."""

    type: str
    function: FunctionDefinition


@dataclass
class LLMResponse:
    """Structured response from an LLM that may include tool calls."""

    content: str
    tool_calls: list[ToolCall]


class BaseLLM(ABC):
    """Abstract base class for LLM providers.

    This class defines the interface that all LLM providers must implement,
    ensuring loose coupling and enabling easy addition of new providers.
    """

    @abstractmethod
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The input prompt to generate a response for
            json_mode: If True, instruct the API to constrain output to valid JSON.
                       Not all providers support this; falls back gracefully if not.

        Returns:
            The generated response as a string

        Raises:
            Exception: If generation fails
        """

    @abstractmethod
    def generate_with_tools(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        tool_choice: dict[str, Any] | str | None = None,
    ) -> LLMResponse:
        """Generate a response with function calling support.

        Default implementation raises NotImplementedError. Subclasses should
        override this if they support function calling.

        Args:
            prompt: The input prompt
            tools: List of tool/function definitions
            tool_choice: Optional tool choice constraint

        Returns:
            LLMResponse with content and tool_calls
        """

    @abstractmethod
    def close(self) -> None:
        """Clean up resources and close connections.

        This method should be called when the LLM provider is no longer needed.
        """
