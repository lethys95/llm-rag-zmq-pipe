"""Shared pytest fixtures and configuration for all tests."""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.config.settings import Settings, LLMConfig, QdrantConfig, ConversationStoreConfig, MemoryDecayConfig


@pytest.fixture
def sample_llm_config() -> LLMConfig:
    """Provide a sample LLM configuration for testing."""
    return LLMConfig(
        provider="openrouter",
        model_path=None,
        openrouter_model="anthropic/claude-3.5-sonnet",
        openrouter_provider=None,
        openrouter_provider_sort="throughput",
        openrouter_provider_allow_fallbacks=True,
    )


@pytest.fixture
def sample_qdrant_config() -> QdrantConfig:
    """Provide a sample Qdrant configuration for testing."""
    return QdrantConfig(
        collection_name="test_conversations",
        embedding_dim=384,
        url="http://localhost:6333",
        api_key=None,
        path=None,
    )


@pytest.fixture
def sample_conversation_store_config() -> ConversationStoreConfig:
    """Provide a sample conversation store configuration for testing."""
    return ConversationStoreConfig(
        db_path=":memory:",
        max_messages=100,
        context_limit=10,
    )


@pytest.fixture
def sample_memory_decay_config() -> MemoryDecayConfig:
    """Provide a sample memory decay configuration for testing."""
    return MemoryDecayConfig(
        half_life_days=30.0,
        chrono_weight=0.7,
        retrieval_threshold=0.3,
        prune_threshold=0.05,
        max_documents=5,
    )


@pytest.fixture
def sample_settings(
    sample_llm_config: LLMConfig,
    sample_qdrant_config: QdrantConfig,
    sample_conversation_store_config: ConversationStoreConfig,
    sample_memory_decay_config: MemoryDecayConfig,
) -> Settings:
    """Provide a complete sample Settings instance for testing."""
    return Settings(
        input_endpoint="tcp://*:5555",
        output_endpoint="tcp://localhost:20501",
        primary_llm=sample_llm_config,
        sentiment_llm=sample_llm_config,
        interpreter_llm=sample_llm_config,
        n_ctx=4096,
        n_threads=4,
        n_gpu_layers=0,
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9,
        top_k=40,
        rag_enabled=True,
        rag_type="qdrant",
        qdrant=sample_qdrant_config,
        conversation_store=sample_conversation_store_config,
        memory_decay=sample_memory_decay_config,
        enable_sentiment_analysis=True,
        enable_context_interpreter=True,
        sentiment_max_retries=3,
        sentiment_retry_delay=1.0,
        log_level="INFO",
    )


@pytest.fixture
def temp_config_file() -> Path:
    """Create a temporary config file for testing."""
    config_data = {
        "input_endpoint": "tcp://*:6666",
        "output_endpoint": "tcp://localhost:6667",
        "primary_llm_provider": "openrouter",
        "primary_openrouter_model": "anthropic/claude-3.5-sonnet",
        "rag_enabled": True,
        "rag_type": "qdrant",
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        path = Path(f.name)
    
    yield path
    
    # Cleanup
    if path.exists():
        path.unlink()


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set up mock environment variables for testing."""
    env_vars = {
        "LLM_RAG_PIPE_INPUT_ADDRESS": "tcp://*:7777",
        "TTS_INPUT_ADDRESS": "tcp://localhost:7778",
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars


@pytest.fixture(autouse=True)
def clean_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Automatically clean known environment variables before each test.
    
    This prevents environment variable pollution between tests and from the system.
    Tests that need specific env vars should use the mock_env_vars fixture.
    """
    env_vars_to_clean = [
        "LLM_RAG_PIPE_INPUT_ADDRESS",
        "LLM_RAG_PIPE_OUTPUT_ADDRESS",
        "TTS_INPUT_ADDRESS",
        "TTS_OUTPUT_ADDRESS",
        "OPENROUTER_API_KEY",
    ]
    
    for var in env_vars_to_clean:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def temp_db_path() -> Path:
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        path = Path(f.name)
    
    yield path
    
    # Cleanup
    if path.exists():
        path.unlink()
