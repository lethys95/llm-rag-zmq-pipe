"""OpenRouter API LLM provider."""

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

import requests

from src.config.settings import settings, LLMConfig
from src.llm.base import BaseLLM, ToolCall, ToolDefinition, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class GenerationMessage:
    role: str
    content: str


@dataclass
class ProviderConfig:
    allow_fallbacks: bool
    order: list[str] | None = None
    sort: str | None = None


@dataclass
class GenerationPayload:
    model: str
    messages: list[GenerationMessage]
    temperature: float
    max_tokens: int
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | str | None = None
    provider: ProviderConfig | None = None
    response_format: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.tools is not None:
            result["tools"] = self.tools
        if self.tool_choice is not None:
            result["tool_choice"] = self.tool_choice
        if self.provider is not None:
            result["provider"] = asdict(self.provider)
        if self.response_format is not None:
            result["response_format"] = self.response_format
        return result


class OpenRouterLLM(BaseLLM):
    """OpenRouter API LLM provider.

    Model configuration is passed at construction time via LLMConfig,
    allowing different instances to use different models (e.g. worker vs primary).
    Generation parameters (temperature, max_tokens) are read from the settings singleton.
    """

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        """Generate a response from OpenRouter API.

        Args:
            prompt: The input prompt to generate a response for

        Returns:
            The generated response as a string

        Raises:
            Exception: If API request fails
        """
        logger.debug("Generating response for prompt: %s...", prompt[:100])

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        payload = GenerationPayload(
            model=self._config.openrouter_model,
            messages=[GenerationMessage(role="user", content=prompt)],
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            response_format={"type": "json_object"} if json_mode else None,
        )

        if self._config.openrouter_provider:
            payload.provider = ProviderConfig(
                allow_fallbacks=self._config.openrouter_provider_allow_fallbacks,
                order=[self._config.openrouter_provider],
            )

        try:
            response = self._make_request(headers, payload.to_dict())
            generated_text = self._extract_response(response)
            logger.debug("Generated response: %s...", generated_text[:100])
            return generated_text
        except Exception as e:
            logger.error("OpenRouter API request failed: %s", e)
            raise

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        tool_choice: dict[str, Any] | str | None = None,
    ) -> LLMResponse:
        """Generate a response with function calling support.

        Args:
            prompt: The input prompt
            tools: List of tool/function definitions
            tool_choice: Optional tool choice constraint

        Returns:
            LLMResponse with content and tool_calls

        Raises:
            Exception: If API request fails
        """
        logger.debug("Generating response with tools for prompt: %s...", prompt[:100])

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        payload = GenerationPayload(
            model=self._config.openrouter_model,
            messages=[GenerationMessage(role="user", content=prompt)],
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            tools=[asdict(tool) for tool in tools],
            tool_choice=tool_choice,
        )

        if self._config.openrouter_provider:
            payload.provider = ProviderConfig(
                allow_fallbacks=self._config.openrouter_provider_allow_fallbacks,
                order=[self._config.openrouter_provider],
            )

        try:
            response = self._make_request(headers, payload.to_dict())
            result = self._extract_response_with_tools(response)
            logger.debug("Generated response with %d tool calls", len(result.tool_calls))
            return result
        except Exception as e:
            logger.error("OpenRouter API request with tools failed: %s", e)
            raise

    def _make_request(
        self, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
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
            logger.error("OpenRouter API error: %d", response.status_code)
            logger.error("Response body: %s", response.text)
            logger.error("Request payload: %s", payload)

        response.raise_for_status()
        return response.json()

    def _extract_response(self, response: dict[str, Any]) -> str:
        """Extract generated text from API response.

        Args:
            response: API response dictionary

        Returns:
            Generated text

        Raises:
            KeyError: If response format is unexpected
        """
        try:
            logger.debug("Full API response: %s", response)
            content = response["choices"][0]["message"]["content"]
            finish_reason = response["choices"][0].get("finish_reason")

            logger.debug("Extracted content length: %d", len(content))
            logger.debug("Finish reason: %s", finish_reason)

            if finish_reason == "length":
                logger.warning("Response was truncated due to length limit!")

            return content
        except (KeyError, IndexError) as e:
            logger.error("Unexpected API response format: %s", response)
            raise ValueError(f"Unexpected API response format: {e}") from e

    def _extract_response_with_tools(self, response: dict[str, Any]) -> LLMResponse:
        """Extract response with tool calls from API response.

        Args:
            response: API response dictionary

        Returns:
            LLMResponse with content and tool_calls

        Raises:
            KeyError: If response format is unexpected
        """
        try:
            message = response["choices"][0]["message"]
            content = message.get("content", "")
            tool_calls_raw: list[dict[str, Any]] = message.get("tool_calls", [])

            tool_calls: list[ToolCall] = []
            for tool_call in tool_calls_raw:
                arguments_str = tool_call["function"]["arguments"]
                arguments = json.loads(arguments_str)
                tool_calls.append(
                    ToolCall(
                        function_name=tool_call["function"]["name"],
                        arguments=arguments,
                        call_id=tool_call.get("id"),
                    )
                )

            logger.debug("Extracted %d tool calls", len(tool_calls))
            return LLMResponse(content=content, tool_calls=tool_calls)
        except (KeyError, IndexError) as e:
            logger.error("Unexpected API response format: %s", response)
            raise ValueError(f"Unexpected API response format: {e}") from e

    def close(self) -> None:
        """Clean up resources.

        No cleanup required for HTTP-based provider.
        """
        logger.info("Closing OpenRouter provider")
