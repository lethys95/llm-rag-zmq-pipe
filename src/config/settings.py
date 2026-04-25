"""Settings dataclass for type-safe configuration.

This module provides a singleton Settings instance that can be imported
and used throughout the application.
"""

from dataclasses import dataclass, field, replace
import logging
import os
from textwrap import dedent


@dataclass
class LLMConfig:
    """Configuration for an LLM provider.

    `model` is the remote model identifier (e.g. OpenRouter model ID).
    `model_path` is the local model file path, used when provider is 'llama'.
    The openrouter_provider_* fields control routing within OpenRouter (e.g. to Cerebras).
    """

    provider: str
    model_path: str | None
    model: str
    openrouter_provider: str | None
    openrouter_provider_sort: str
    openrouter_provider_allow_fallbacks: bool


DEFAULT_WORKER_LLM_PROFILE = LLMConfig(
    provider="openrouter",
    model_path=None,
    model="openai/gpt-oss-120b",
    openrouter_provider="Cerebras",
    openrouter_provider_sort="throughput",
    openrouter_provider_allow_fallbacks=True,
)


@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database."""

    collection_name: str
    embedding_dim: int
    url: str | None
    api_key: str | None
    path: str | None


@dataclass
class ConversationStoreConfig:
    """
    Main storage for conversation through sqlite3 db
    """

    db_path: str
    max_messages: int
    context_limit: int


@dataclass
class MemoryDecayConfig:
    """
    Configuration for memory decay process.
    Intended to be used for running through the vector db, figuring out which
    memories should be decayed, etc.
    """

    half_life_days: float
    chrono_weight: float
    retrieval_threshold: float
    prune_threshold: float
    max_documents: int


@dataclass
class DetoxConfig:
    """Configuration for the detox protocol."""

    idle_trigger_minutes: int
    min_interval_minutes: int
    max_duration_minutes: int


@dataclass
class SentimentAnalysisConfig:
    """Configuration for sentiment analysis."""

    max_retries: int
    retry_delay: float


@dataclass
class NudgingConfig:
    """
    Nudging refers to the effect of pushing the AI or user slightly towards a neutral.
    """

    nudge_strength: float
    max_companion_drift: float
    base_user_influence: float
    base_companion_influence: float
    max_trust_boost: float


@dataclass
class MemoryConsolidationConfig:
    """
    Configuration for memory consolidation process.
    Intended to be used for running through the vector db, figuring out which
    memories should be preserved, etc.
    """

    consolidation_threshold: float
    max_memories_per_batch: int


@dataclass
class Settings:
    """Application settings with type validation."""

    zmq_input_endpoint: str = os.environ.get("LLM_RAG_PIPE_INPUT_ADDRESS", "tcp://*:5555")
    zmq_output_endpoint: str = os.environ.get("TTS_INPUT_ADDRESS", "tcp://localhost:20501")

    primary_llm: LLMConfig = field(
        default_factory=lambda: replace(
            DEFAULT_WORKER_LLM_PROFILE,
            model="z-ai/glm-4.7",
        )
    )

    worker_llm: LLMConfig = field(default_factory=lambda: DEFAULT_WORKER_LLM_PROFILE)

    n_ctx: int = 80000
    n_threads: int = 4
    n_gpu_layers: int = -1
    temperature: float = 0.7
    max_tokens: int = 8000
    top_p: float = 0.95
    top_k: int = 40

    rag_enabled: bool = True
    rag_type: str = "qdrant"
    rag_embedding_model = "all-MiniLM-L6-v2"
    openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
    openrouter_api_url: str = "https://openrouter.ai/api/v1/chat/completions"

    qdrant: QdrantConfig = field(
        default_factory=lambda: QdrantConfig(
            collection_name="llm_rag_memories",
            embedding_dim=384,
            url=None,
            api_key=None,
            path=None,
        )
    )

    conversation_store: ConversationStoreConfig = field(
        default_factory=lambda: ConversationStoreConfig(
            db_path="./data/conversations.db",
            max_messages=200,
            context_limit=15,
        )
    )

    memory_decay: MemoryDecayConfig = field(
        default_factory=lambda: MemoryDecayConfig(
            half_life_days=30.0,
            chrono_weight=1.0,
            retrieval_threshold=0.15,
            prune_threshold=0.05,
            max_documents=25,
        )
    )

    detox: DetoxConfig = field(
        default_factory=lambda: DetoxConfig(
            idle_trigger_minutes=60,
            min_interval_minutes=120,
            max_duration_minutes=30,
        )
    )

    sentiment: SentimentAnalysisConfig = field(
        default_factory=lambda: SentimentAnalysisConfig(
            max_retries=3,
            retry_delay=0.5,
        )
    )

    nudging: NudgingConfig = field(
        default_factory=lambda: NudgingConfig(
            nudge_strength=0.15,
            max_companion_drift=0.3,
            base_user_influence=0.3,
            base_companion_influence=0.7,
            max_trust_boost=0.3,
        )
    )

    memory_consolidation: MemoryConsolidationConfig = field(
        default_factory=lambda: MemoryConsolidationConfig(
            consolidation_threshold=0.7,
            max_memories_per_batch=10,
        )
    )

    enable_sentiment_analysis: bool = True

    log_level = logging.INFO

    def _validate_llms(self) -> None:
        """Validate LLM configurations."""
        for name, llm_config in [
            ("primary", self.primary_llm),
            ("worker", self.worker_llm),
        ]:
            if llm_config.provider not in ["llama", "llama_local", "openrouter"]:
                raise ValueError(
                    f"Invalid {name}_llm provider: {llm_config.provider}. "
                    "Must be 'llama', 'llama_local', or 'openrouter'"
                )

            if (
                llm_config.provider in ["llama", "llama_local"]
                and not llm_config.model_path
            ):
                raise ValueError(
                    f"{name}_llm model_path is required when using llama provider"
                )

    def _validate_rag(self) -> None:
        """Validate RAG configuration."""
        if self.rag_enabled and self.rag_type not in ["qdrant", "none"]:
            raise ValueError(
                f"Invalid rag_type: {self.rag_type}. Must be 'qdrant' or 'none'"
            )

    def _validate_memory_decay(self) -> None:
        """Validate memory decay configuration."""
        if self.memory_decay.half_life_days <= 0:
            raise ValueError(
                f"memory_decay.half_life_days must be positive, got {self.memory_decay.half_life_days}"
            )

        if not 0.0 <= self.memory_decay.chrono_weight <= 2.0:
            raise ValueError(
                dedent(f"""
                    memory_decay.chrono_weight must be between 0.0 and 2.0,
                    got {self.memory_decay.chrono_weight}""")
            )

        if not 0.0 <= self.memory_decay.retrieval_threshold <= 1.0:
            raise ValueError(
                dedent(f"""
                    memory_decay.retrieval_threshold must be between 0.0 and 1.0,
                    got {self.memory_decay.retrieval_threshold}""")
            )

        if not 0.0 <= self.memory_decay.prune_threshold <= 1.0:
            raise ValueError(
                dedent(f"""
                    memory_decay.prune_threshold must be between 0.0 and 1.0,
                    got {self.memory_decay.prune_threshold}""")
            )

        if self.memory_decay.max_documents <= 0:
            raise ValueError(
                f"memory_decay.max_documents must be positive, got {self.memory_decay.max_documents}"
            )

    def _validate_detox(self) -> None:
        """Validate detox configuration."""
        if self.detox.idle_trigger_minutes <= 0:
            raise ValueError(
                f"detox.idle_trigger_minutes must be positive, got {self.detox.idle_trigger_minutes}"
            )

        if self.detox.min_interval_minutes <= 0:
            raise ValueError(
                f"detox.min_interval_minutes must be positive, got {self.detox.min_interval_minutes}"
            )

        if self.detox.max_duration_minutes <= 0:
            raise ValueError(
                dedent(f"""
                    detox.max_duration_minutes must be positive,
                    got {self.detox.max_duration_minutes}.""")
            )

    def _validate_generation_params(self) -> None:
        """Validate generation parameters."""
        if self.n_ctx <= 0:
            raise ValueError(f"n_ctx must be positive, got {self.n_ctx}")

        if self.n_threads <= 0:
            raise ValueError(f"n_threads must be positive, got {self.n_threads}")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                dedent(f"""
                    temperature must be between 0.0 and 2.0,
                    got {self.temperature}.""")
            )

        if self.max_tokens <= 0:
            raise ValueError(
                dedent(f"""
                    max_tokens must be positive, got {self.max_tokens}.""")
            )

    def validate(self) -> None:
        """Validate configuration settings.

        Raises:
            ValueError: If configuration is invalid.
        """
        self._validate_llms()
        self._validate_rag()
        self._validate_memory_decay()
        self._validate_detox()
        self._validate_generation_params()


# Singleton instance - import this to access settings throughout the app
settings = Settings()
