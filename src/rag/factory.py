"""Factory for creating RAG provider instances."""

import logging

from ..config.settings import Settings
from .base import BaseRAG
from .qdrant_connector import QdrantRAG
from .selector import RAGSelector

logger = logging.getLogger(__name__)


def create_rag_provider(
    settings: Settings,
    rag_type: str | None = None,
    **kwargs
) -> BaseRAG:
    """Create a RAG provider based on settings.
    
    Factory function that instantiates the appropriate RAG provider.
    Currently supports Qdrant vector database for semantic search.
    
    Args:
        settings: Application settings
        rag_type: Optional override for RAG type (currently only "qdrant" supported)
        **kwargs: Additional arguments passed to RAG constructor
        
    Returns:
        Initialized RAG provider instance
        
    Raises:
        ValueError: If RAG is disabled or unknown RAG type specified
    """
    if not settings.rag_enabled:
        raise ValueError(
            "RAG is disabled in settings. Set 'rag_enabled' to true to use RAG functionality."
        )
    
    # Determine which RAG provider to use (default to qdrant)
    provider_type = rag_type or settings.rag_type
    
    if provider_type == "qdrant":
        logger.info("Creating Qdrant RAG provider")
        
        collection_name = kwargs.get("collection_name", settings.qdrant.collection_name)
        embedding_dim = kwargs.get("embedding_dim", settings.qdrant.embedding_dim)
        url = kwargs.get("url", settings.qdrant.url)
        api_key = kwargs.get("api_key", settings.qdrant.api_key)
        path = kwargs.get("path", settings.qdrant.path)
        
        selector = RAGSelector(
            max_documents=kwargs.get("max_documents", settings.memory_decay.max_documents),
            min_score=kwargs.get("min_score", settings.memory_decay.retrieval_threshold),
            max_age_hours=kwargs.get("max_age_hours", None)
        )
        
        return QdrantRAG(
            collection_name=collection_name,
            embedding_dim=embedding_dim,
            url=url,
            api_key=api_key,
            path=path,
            selector=selector
        )
    
    else:
        raise ValueError(
            f"Unknown RAG type '{provider_type}'. Only 'qdrant' is currently supported."
        )
