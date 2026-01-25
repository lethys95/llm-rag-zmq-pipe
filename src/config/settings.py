"""Settings dataclass for type-safe configuration."""

from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    
    provider: str
    model_path: str | None
    openrouter_model: str
    openrouter_provider: str | None
    openrouter_provider_sort: str
    openrouter_provider_allow_fallbacks: bool


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
    """Configuration for conversation storage (SQLite)."""
    
    db_path: str
    max_messages: int
    context_limit: int


@dataclass
class MemoryDecayConfig:
    """Configuration for memory decay algorithm."""
    
    half_life_days: float
    chrono_weight: float
    retrieval_threshold: float
    prune_threshold: float
    max_documents: int


@dataclass
class DetoxConfig:
    """Configuration for detox protocol."""
    
    idle_trigger_minutes: int
    min_interval_minutes: int
    max_duration_minutes: int


@dataclass
class Settings:
    """Application settings with type validation."""
    
    input_endpoint: str
    output_endpoint: str
    
    primary_llm: LLMConfig
    sentiment_llm: LLMConfig
    interpreter_llm: LLMConfig
    
    n_ctx: int
    n_threads: int
    n_gpu_layers: int
    temperature: float
    max_tokens: int
    top_p: float
    top_k: int
    
    rag_enabled: bool
    rag_type: str
    
    qdrant: QdrantConfig
    conversation_store: ConversationStoreConfig
    memory_decay: MemoryDecayConfig
    detox: DetoxConfig
    
    enable_sentiment_analysis: bool
    enable_context_interpreter: bool
    
    sentiment_max_retries: int
    sentiment_retry_delay: float
    
    log_level: str
    
    def validate(self) -> None:
        """Validate configuration settings.
        
        Raises:
            ValueError: If configuration is invalid.
        """
        for name, llm_config in [
            ("primary", self.primary_llm),
            ("sentiment", self.sentiment_llm),
            ("interpreter", self.interpreter_llm),
        ]:
            if llm_config.provider not in ["llama", "llama_local", "openrouter"]:
                raise ValueError(
                    f"Invalid {name}_llm provider: {llm_config.provider}. "
                    "Must be 'llama', 'llama_local', or 'openrouter'"
                )
            
            if llm_config.provider in ["llama", "llama_local"] and not llm_config.model_path:
                raise ValueError(
                    f"{name}_llm model_path is required when using llama provider"
                )
        
        if self.rag_enabled and self.rag_type not in ["qdrant", "none"]:
            raise ValueError(
                f"Invalid rag_type: {self.rag_type}. "
                "Must be 'qdrant' or 'none'"
            )
        
        if self.memory_decay.half_life_days <= 0:
            raise ValueError(
                f"memory_decay.half_life_days must be positive, got {self.memory_decay.half_life_days}"
            )
        
        if not 0.0 <= self.memory_decay.chrono_weight <= 2.0:
            raise ValueError(
                f"memory_decay.chrono_weight must be between 0.0 and 2.0, got {self.memory_decay.chrono_weight}"
            )
        
        if not 0.0 <= self.memory_decay.retrieval_threshold <= 1.0:
            raise ValueError(
                f"memory_decay.retrieval_threshold must be between 0.0 and 1.0, got {self.memory_decay.retrieval_threshold}"
            )
        
        if not 0.0 <= self.memory_decay.prune_threshold <= 1.0:
            raise ValueError(
                f"memory_decay.prune_threshold must be between 0.0 and 1.0, got {self.memory_decay.prune_threshold}"
            )
        
        if self.memory_decay.max_documents <= 0:
            raise ValueError(
                f"memory_decay.max_documents must be positive, got {self.memory_decay.max_documents}"
            )
        
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
                f"detox.max_duration_minutes must be positive, got {self.detox.max_duration_minutes}"
            )
        
        if self.n_ctx <= 0:
            raise ValueError(f"n_ctx must be positive, got {self.n_ctx}")
        
        if self.n_threads <= 0:
            raise ValueError(f"n_threads must be positive, got {self.n_threads}")
        
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )
        
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
