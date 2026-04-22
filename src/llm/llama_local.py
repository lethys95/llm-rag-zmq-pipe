"""Local LLM provider using llama-cpp-python."""

from __future__ import annotations

import logging
from textwrap import dedent
from typing import Any

from src.config.settings import settings
from src.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class LlamaLocalLLM(BaseLLM):
    """Local LLM provider using llama-cpp-python.

    Configuration is sourced from settings.py. This class is a singleton
    to ensure the model is never loaded more than once.
    """

    _instance: LlamaLocalLLM | None = None
    _initialized: bool = False

    def __new__(cls) -> LlamaLocalLLM:
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        if not settings.primary_llm.model_path:
            raise ValueError("model_path is not configured in settings.primary_llm")

        self._model = self._load_model()
        self._initialized = True

    def _load_model(self) -> Any:
        model_path = settings.primary_llm.model_path
        assert model_path is not None, "model_path is not configured"

        logger.info("Loading model from %s", model_path)
        logger.info(
            dedent("""\
                Configuration: n_ctx=%d, \
                n_threads=%d, \
                n_gpu_layers=%d"""),
            settings.n_ctx,
            settings.n_threads,
            settings.n_gpu_layers,
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
                model_path=model_path,
                n_ctx=settings.n_ctx,
                n_threads=settings.n_threads,
                n_gpu_layers=settings.n_gpu_layers,
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
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            echo=False,
        )

        generated_text = response["choices"][0]["text"]
        logger.debug("Generated response: %.100s...", generated_text)

        return generated_text

    def close(self) -> None:
        logger.info("Closing local LLM provider")
        self._model = None
