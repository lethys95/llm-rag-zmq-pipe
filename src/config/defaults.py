"""Default configuration values."""

DEFAULT_CONFIG: dict[str, any] = {
    "input_endpoint": "tcp://*:5555",
    "output_endpoint": "tcp://localhost:20501",
    
    "primary_llm_provider": "openrouter",
    "primary_model_path": None,
    "primary_openrouter_model": "z-ai/glm-4.7",
    "primary_openrouter_provider": "Cerebras",
    "primary_openrouter_provider_sort": "throughput",
    "primary_openrouter_provider_allow_fallbacks": True,
    
    "sentiment_llm_provider": "openrouter",
    "sentiment_model_path": None,
    "sentiment_openrouter_model": "openai/gpt-oss-120b",
    "sentiment_openrouter_provider": "Cerebras",
    "sentiment_openrouter_provider_sort": "throughput",
    "sentiment_openrouter_provider_allow_fallbacks": True,
    
    "interpreter_llm_provider": "openrouter",
    "interpreter_model_path": None,
    "interpreter_openrouter_model": "openai/gpt-oss-120b",
    "interpreter_openrouter_provider": "Cerebras",
    "interpreter_openrouter_provider_sort": "throughput",
    "interpreter_openrouter_provider_allow_fallbacks": True,
    
    "n_ctx": 2048,
    "n_threads": 4,
    "n_gpu_layers": -1,
    "temperature": 0.7,
    "max_tokens": 8000,
    "top_p": 0.95,
    "top_k": 40,
    
    "rag_enabled": True,
    "rag_type": "qdrant",
    
    "qdrant_collection_name": "llm_rag_memories",
    "qdrant_embedding_dim": 384,
    "qdrant_url": None,
    "qdrant_api_key": None,
    "qdrant_path": None,
    
    "conversation_db_path": "./data/conversations.db",
    "conversation_db_max_messages": 200,
    "conversation_context_limit": 15,
    
    "memory_half_life_days": 30.0,
    "chrono_weight": 1.0,
    "memory_retrieval_threshold": 0.15,
    "memory_prune_threshold": 0.05,
    "max_context_documents": 25,
    
    "enable_sentiment_analysis": True,
    "enable_context_interpreter": True,
    
    "sentiment_max_retries": 3,
    "sentiment_retry_delay": 0.5,
    
    "log_level": "INFO",
}
