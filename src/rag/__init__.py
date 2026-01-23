"""RAG (Retrieval-Augmented Generation) abstraction and implementations.

Uses Qdrant vector database for semantic search with memory decay algorithms.
"""

from .base import BaseRAG
from .factory import create_rag_provider
from .selector import RAGDocument, RAGSelector
from .qdrant_connector import QdrantRAG
from .algorithms import MemoryDecayAlgorithm

__all__ = [
    "BaseRAG",
    "create_rag_provider",
    "QdrantRAG",
    "RAGDocument",
    "RAGSelector",
    "MemoryDecayAlgorithm",
]
