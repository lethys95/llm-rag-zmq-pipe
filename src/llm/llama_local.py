"""Local LLM provider using llama-cpp-python."""

from __future__ import annotations

import logging
from textwrap import dedent
from typing import Any

from src.config.settings import LocalLLMConfig
from src.llm.base import BaseLLM, LLMResponse, ToolDefinition

logger = logging.getLogger(__name__)


class LlamaLocalLLM(BaseLLM):
    """Local LLM provider using llama-cpp-python.

    Instantiate once at the composition root — loading a local model is expensive
    and should only happen once per process.
    """

    def __init__(self, config: LocalLLMConfig) -> None:
        if not config.model_path:
            raise ValueError("model_path is required for local LLM")

        self._config = config
        self._model = self._load_model()

    def _load_model(self) -> Any:
        logger.info("Loading model from %s", self._config.model_path)
        logger.info(
            dedent("""\
                Configuration: n_ctx=%d, \
                n_threads=%d, \
                n_gpu_layers=%d"""),
            self._config.n_ctx,
            self._config.n_threads,
            self._config.n_gpu_layers,
        )

        try:
            from llama_cpp import Llama  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                dedent("""\
                    llama-cpp-python is not installed. \
                    Please run ./setup.sh to install it with CUDA support.""")
            ) from e

        try:
            model = Llama(
                model_path=self._config.model_path,
                n_ctx=self._config.n_ctx,
                n_threads=self._config.n_threads,
                n_gpu_layers=self._config.n_gpu_layers,
                verbose=False,
            )
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}") from e

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        logger.debug("Generating response for prompt: %.100s...", prompt)

        assert self._model is not None, "Model not loaded"

        response = self._model(
            prompt,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            top_k=self._config.top_k,
            echo=False,
        )

        generated_text = response["choices"][0]["text"]
        logger.debug("Generated response: %.100s...", generated_text)

        return generated_text

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        tool_choice: dict[str, Any] | str | None = None,
    ) -> LLMResponse:
        raise NotImplementedError("LlamaLocalLLM does not support tool calling")

    def close(self) -> None:
        logger.info("Closing local LLM provider")
        self._model = None
