"""OpenRouter API LLM provider."""

import logging
import os

import requests

from .base import BaseLLM

logger = logging.getLogger(__name__)


class OpenRouterLLM(BaseLLM):
    """OpenRouter API LLM provider.
    
    This provider sends requests to the OpenRouter API for LLM generation.
    """
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(
        self,
        model: str = "z.ai/glm-4.7",
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: str | None = None,
        provider: str | None = None,
        provider_sort: str = "throughput",
        provider_allow_fallbacks: bool = True,
    ):
        """Initialize the OpenRouter LLM provider.
        
        Args:
            model: The model identifier to use (e.g., "z.ai/glm-4.7")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_key: OpenRouter API key (if not provided, reads from environment)
            provider: Specific provider to use (e.g., "Cerebras"). If None, uses automatic routing.
            provider_sort: How to sort providers ("throughput", "latency", "cost", etc.)
            provider_allow_fallbacks: Whether to allow fallback providers if primary unavailable
            
        Raises:
            ValueError: If API key is not provided and not in environment
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        self.provider_sort = provider_sort
        self.provider_allow_fallbacks = provider_allow_fallbacks
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Please set OPENROUTER_API_KEY environment variable "
                "or provide it as a parameter."
            )
        
        provider_info = f" with provider: {provider}" if provider else ""
        logger.info(f"Initialized OpenRouter provider with model: {model}{provider_info}")
    
    def generate(self, prompt: str) -> str:
        """Generate a response from OpenRouter API.
        
        Args:
            prompt: The input prompt to generate a response for
            
        Returns:
            The generated response as a string
            
        Raises:
            Exception: If API request fails
        """
        logger.debug(f"Generating response for prompt: {prompt[:100]}...")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # Add provider configuration if specified
        if self.provider or self.provider_sort:
            provider_config = {}
            
            if self.provider:
                provider_config["only"] = [self.provider]
            
            if self.provider_sort:
                provider_config["sort"] = self.provider_sort
            
            if self.provider_allow_fallbacks is not None:
                provider_config["allow_fallbacks"] = self.provider_allow_fallbacks
            
            payload["provider"] = provider_config
        
        try:
            response = self._make_request(headers, payload)
            generated_text = self._extract_response(response)
            
            logger.debug(f"Generated response: {generated_text[:100]}...")
            return generated_text
            
        except Exception as e:
            logger.error(f"OpenRouter API request failed: {e}")
            raise
    
    def _make_request(self, headers: dict, payload: dict) -> dict:
        """Make HTTP request to OpenRouter API.
        
        Args:
            headers: Request headers
            payload: Request payload
            
        Returns:
            API response as dictionary
            
        Raises:
            requests.HTTPError: If request fails
        """
        response = requests.post(
            self.API_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        
        # Log the error details if request fails
        if not response.ok:
            logger.error(f"OpenRouter API error: {response.status_code}")
            logger.error(f"Response body: {response.text}")
            logger.error(f"Request payload: {payload}")
        
        response.raise_for_status()
        return response.json()
    
    def _extract_response(self, response: dict) -> str:
        """Extract generated text from API response.
        
        Args:
            response: API response dictionary
            
        Returns:
            Generated text
            
        Raises:
            KeyError: If response format is unexpected
        """
        try:
            # Log the full response for debugging
            logger.debug(f"Full API response: {response}")
            
            content = response["choices"][0]["message"]["content"]
            finish_reason = response["choices"][0].get("finish_reason")
            
            logger.debug(f"Extracted content length: {len(content)}")
            logger.debug(f"Finish reason: {finish_reason}")
            
            if finish_reason == "length":
                logger.warning("Response was truncated due to length limit!")
            
            return content
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected API response format: {response}")
            raise ValueError(f"Unexpected API response format: {e}")
    
    def close(self) -> None:
        """Clean up resources.
        
        No cleanup required for HTTP-based provider.
        """
        logger.info("Closing OpenRouter provider")
