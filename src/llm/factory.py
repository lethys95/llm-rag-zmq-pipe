"""Factory for creating LLM provider instances."""

import logging

from src.config.settings import Settings
from src.llm.base import BaseLLM
from src.llm.llama_local import LlamaLocalLLM
from src.llm.openrouter import OpenRouterLLM

logger = logging.getLogger(__name__)


def create_llm_provider(
    provider_type: str,
    model_path: str = None,
    openrouter_model: str = None,
    openrouter_provider: str = None,
    openrouter_provider_sort: str = "throughput",
    openrouter_provider_allow_fallbacks: bool = True,
    n_ctx: int = 2048,
    n_threads: int = 4,
    n_gpu_layers: int = -1,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.95,
    top_k: int = 40,
) -> BaseLLM:
    """Create an LLM provider based on parameters.
    
    Factory function that instantiates the appropriate LLM provider
    based on the provided parameters.
    
    Args:
        provider_type: Type of provider ("llama" or "openrouter")
        model_path: Path to local model (for llama provider)
        openrouter_model: Model identifier (for openrouter provider)
        openrouter_provider: Specific provider to use (e.g., "Cerebras")
        openrouter_provider_sort: How to sort providers ("throughput", "latency", "cost")
        openrouter_provider_allow_fallbacks: Allow fallback providers if primary unavailable
        n_ctx: Context window size
        n_threads: CPU threads
        n_gpu_layers: GPU layers
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling
        top_k: Top-k sampling
        
    Returns:
        Initialized LLM provider instance
        
    Raises:
        ValueError: If provider type is unknown or missing required parameters
        ImportError: If required dependencies are not installed
    """
    provider = provider_type.lower()
    
    if provider == "llama":
        logger.info(f"Creating local Llama provider with model: {model_path}")
        if not model_path:
            raise ValueError("model_path is required for llama provider")
            
        return LlamaLocalLLM(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
        )
    
    elif provider == "openrouter":
        logger.info(f"Creating OpenRouter provider with model: {openrouter_model}")
        return OpenRouterLLM(
            model=openrouter_model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=openrouter_provider,
            provider_sort=openrouter_provider_sort,
            provider_allow_fallbacks=openrouter_provider_allow_fallbacks,
        )
    
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            "Supported providers: llama, openrouter"
        )


def create_primary_llm(settings: Settings) -> BaseLLM:
    """Create primary LLM provider from settings.
    
    Args:
        settings: Application settings
        
    Returns:
        Initialized primary LLM provider
    """
    return create_llm_provider(
        provider_type=settings.primary_llm.provider,
        model_path=settings.primary_llm.model_path,
        openrouter_model=settings.primary_llm.openrouter_model,
        openrouter_provider=settings.primary_llm.openrouter_provider,
        openrouter_provider_sort=settings.primary_llm.openrouter_provider_sort,
        openrouter_provider_allow_fallbacks=settings.primary_llm.openrouter_provider_allow_fallbacks,
        n_ctx=settings.n_ctx,
        n_threads=settings.n_threads,
        n_gpu_layers=settings.n_gpu_layers,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        top_p=settings.top_p,
        top_k=settings.top_k,
    )


def create_sentiment_llm(settings: Settings) -> BaseLLM:
    """Create sentiment analysis LLM provider from settings.
    
    Args:
        settings: Application settings
        
    Returns:
        Initialized sentiment LLM provider
    """
    return create_llm_provider(
        provider_type=settings.sentiment_llm.provider,
        model_path=settings.sentiment_llm.model_path,
        openrouter_model=settings.sentiment_llm.openrouter_model,
        openrouter_provider=settings.sentiment_llm.openrouter_provider,
        openrouter_provider_sort=settings.sentiment_llm.openrouter_provider_sort,
        openrouter_provider_allow_fallbacks=settings.sentiment_llm.openrouter_provider_allow_fallbacks,
        n_ctx=settings.n_ctx,
        n_threads=settings.n_threads,
        n_gpu_layers=settings.n_gpu_layers,
        temperature=0.3,
        max_tokens=1024,
        top_p=settings.top_p,
        top_k=settings.top_k,
    )


def create_interpreter_llm(settings: Settings) -> BaseLLM:
    """Create context interpreter LLM provider from settings.
    
    Args:
        settings: Application settings
        
    Returns:
        Initialized context interpreter LLM provider
    """
    return create_llm_provider(
        provider_type=settings.interpreter_llm.provider,
        model_path=settings.interpreter_llm.model_path,
        openrouter_model=settings.interpreter_llm.openrouter_model,
        openrouter_provider=settings.interpreter_llm.openrouter_provider,
        openrouter_provider_sort=settings.interpreter_llm.openrouter_provider_sort,
        openrouter_provider_allow_fallbacks=settings.interpreter_llm.openrouter_provider_allow_fallbacks,
        n_ctx=settings.n_ctx,
        n_threads=settings.n_threads,
        n_gpu_layers=settings.n_gpu_layers,
        temperature=0.5,
        max_tokens=1024,
        top_p=settings.top_p,
        top_k=settings.top_k,
    )
