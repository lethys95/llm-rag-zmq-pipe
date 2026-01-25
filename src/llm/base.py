"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Represents a function/tool call from an LLM."""
    
    function_name: str
    arguments: dict
    call_id: str | None = None


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
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The input prompt to generate a response for
            
        Returns:
            The generated response as a string
            
        Raises:
            Exception: If generation fails
        """
        pass
    
    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        tool_choice: dict | str | None = None
    ) -> LLMResponse:
        """Generate a response with function calling support.
        
        Default implementation raises NotImplementedError. Subclasses should
        override this if they support function calling.
        
        Args:
            prompt: The input prompt
            tools: List of tool/function definitions following OpenAI format
            tool_choice: Optional tool choice constraint (e.g., {"type": "function", "function": {"name": "select_nodes"}})
            
        Returns:
            LLMResponse with content and tool_calls
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support function calling. "
            "Use generate() instead or implement generate_with_tools()."
        )
    
    @abstractmethod
    def close(self) -> None:
        """Clean up resources and close connections.
        
        This method should be called when the LLM provider is no longer needed.
        """
        pass