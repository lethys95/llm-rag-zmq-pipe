"""Configuration loader with cascading precedence: CLI → ENV → file → defaults."""

import json
import os
from pathlib import Path

from .defaults import DEFAULT_CONFIG
from .settings import (
    Settings,
    LLMConfig,
    QdrantConfig,
    ConversationStoreConfig,
    MemoryDecayConfig,
    DetoxConfig,
)


def load_config(
    cli_args: dict[str, any] | None = None,
    config_file: Path | None = None
) -> Settings:
    """Load configuration with cascading precedence.
    
    Precedence order (highest to lowest):
    1. CLI arguments (if provided and not None)
    2. Environment variables (LLM_RAG_PIPE_INPUT_ADDRESS)
    3. Config file values (if file exists)
    4. Default values
    
    Args:
        cli_args: Dictionary of CLI arguments (non-None values override lower precedence)
        config_file: Path to JSON configuration file
        
    Returns:
        Validated Settings instance
        
    Raises:
        FileNotFoundError: If config_file is specified but doesn't exist
        ValueError: If configuration is invalid
        json.JSONDecodeError: If config file has invalid JSON
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_file is not None:
        config.update(_load_config_file(config_file))
    
    config.update(_load_env_variables())
    
    if cli_args is not None:
        config.update(_filter_none_values(cli_args))
    
    settings = _create_settings_from_flat_config(config)
    settings.validate()
    
    return settings


def _create_settings_from_flat_config(config: dict[str, any]) -> Settings:
    """Transform flat config dictionary into nested Settings structure.
    
    Args:
        config: Flat configuration dictionary
        
    Returns:
        Settings instance with nested configuration objects
    """
    primary_llm = LLMConfig(
        provider=config["primary_llm_provider"],
        model_path=config["primary_model_path"],
        openrouter_model=config["primary_openrouter_model"],
        openrouter_provider=config["primary_openrouter_provider"],
        openrouter_provider_sort=config["primary_openrouter_provider_sort"],
        openrouter_provider_allow_fallbacks=config["primary_openrouter_provider_allow_fallbacks"],
    )
    
    sentiment_llm = LLMConfig(
        provider=config["sentiment_llm_provider"],
        model_path=config["sentiment_model_path"],
        openrouter_model=config["sentiment_openrouter_model"],
        openrouter_provider=config["sentiment_openrouter_provider"],
        openrouter_provider_sort=config["sentiment_openrouter_provider_sort"],
        openrouter_provider_allow_fallbacks=config["sentiment_openrouter_provider_allow_fallbacks"],
    )
    
    interpreter_llm = LLMConfig(
        provider=config["interpreter_llm_provider"],
        model_path=config["interpreter_model_path"],
        openrouter_model=config["interpreter_openrouter_model"],
        openrouter_provider=config["interpreter_openrouter_provider"],
        openrouter_provider_sort=config["interpreter_openrouter_provider_sort"],
        openrouter_provider_allow_fallbacks=config["interpreter_openrouter_provider_allow_fallbacks"],
    )
    
    qdrant = QdrantConfig(
        collection_name=config["qdrant_collection_name"],
        embedding_dim=config["qdrant_embedding_dim"],
        url=config["qdrant_url"],
        api_key=config["qdrant_api_key"],
        path=config["qdrant_path"],
    )
    
    conversation_store = ConversationStoreConfig(
        db_path=config["conversation_db_path"],
        max_messages=config["conversation_db_max_messages"],
        context_limit=config["conversation_context_limit"],
    )
    
    memory_decay = MemoryDecayConfig(
        half_life_days=config["memory_half_life_days"],
        chrono_weight=config["chrono_weight"],
        retrieval_threshold=config["memory_retrieval_threshold"],
        prune_threshold=config["memory_prune_threshold"],
        max_documents=config["max_context_documents"],
    )
    
    detox = DetoxConfig(
        idle_trigger_minutes=config["detox_idle_trigger_minutes"],
        min_interval_minutes=config["detox_min_interval_minutes"],
        max_duration_minutes=config["detox_max_duration_minutes"],
    )
    
    return Settings(
        input_endpoint=config["input_endpoint"],
        output_endpoint=config["output_endpoint"],
        primary_llm=primary_llm,
        sentiment_llm=sentiment_llm,
        interpreter_llm=interpreter_llm,
        n_ctx=config["n_ctx"],
        n_threads=config["n_threads"],
        n_gpu_layers=config["n_gpu_layers"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        top_p=config["top_p"],
        top_k=config["top_k"],
        rag_enabled=config["rag_enabled"],
        rag_type=config["rag_type"],
        qdrant=qdrant,
        conversation_store=conversation_store,
        memory_decay=memory_decay,
        detox=detox,
        enable_sentiment_analysis=config["enable_sentiment_analysis"],
        enable_context_interpreter=config["enable_context_interpreter"],
        sentiment_max_retries=config["sentiment_max_retries"],
        sentiment_retry_delay=config["sentiment_retry_delay"],
        log_level=config["log_level"],
    )


def _load_config_file(config_file: Path) -> dict[str, any]:
    """Load configuration from JSON file.
    
    Args:
        config_file: Path to JSON configuration file
        
    Returns:
        Dictionary of configuration values
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file has invalid JSON
    """
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        return json.load(f)


def _load_env_variables() -> dict[str, any]:
    """Load configuration from environment variables.
    
    Supported environment variables:
    - LLM_RAG_PIPE_INPUT_ADDRESS: Input endpoint address (e.g., tcp://*:5555)
    - TTS_INPUT_ADDRESS: Output endpoint address (e.g., tcp://localhost:20501)
    
    Returns:
        Dictionary of configuration values from environment
    """
    env_config = {}
    
    input_address = os.environ.get('LLM_RAG_PIPE_INPUT_ADDRESS')
    if input_address:
        env_config['input_endpoint'] = input_address
    
    output_address = os.environ.get('TTS_INPUT_ADDRESS')
    if output_address:
        env_config['output_endpoint'] = output_address
    
    return env_config


def _filter_none_values(d: dict[str, any]) -> dict[str, any]:
    """Filter out None values from dictionary.
    
    Args:
        d: Input dictionary
        
    Returns:
        New dictionary with None values removed
    """
    return {k: v for k, v in d.items() if v is not None}
