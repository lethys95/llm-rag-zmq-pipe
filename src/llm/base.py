"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod


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
    
    @abstractmethod
    def close(self) -> None:
        """Clean up resources and close connections.
        
        This method should be called when the LLM provider is no longer needed.
        """
        pass
