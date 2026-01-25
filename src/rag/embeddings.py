"""Embedding service for generating text embeddings."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Self

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers.
    
    This uses a singleton pattern to cache the model and avoid reloading.
    The default model is 'all-MiniLM-L6-v2' which produces 384-dimensional
    embeddings and is optimized for semantic similarity.
    """
    
    _instance: Self | None = None
    _lock: Lock = Lock()
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        self._model_lock = Lock()
        
        logger.info(f"Embedding service initialized with model: {model_name}")
    
    @classmethod
    def get_instance(cls, model_name: str = "all-MiniLM-L6-v2") -> Self:
        """Get or create the singleton instance.
        
        Args:
            model_name: Model to use (only used on first creation)
            
        Returns:
            Singleton EmbeddingService instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_name)
        return cls._instance
    
    def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model."""
        if self.model is None:
            with self._model_lock:
                if self.model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                        logger.info(f"Loading sentence-transformers model: {self.model_name}")
                        self.model = SentenceTransformer(self.model_name)
                        logger.info("Model loaded successfully")
                    except ImportError:
                        logger.error(
                            "sentence-transformers not installed. "
                            "Install with: pip install sentence-transformers"
                        )
                        raise
                    except Exception as e:
                        logger.error(f"Error loading embedding model: {e}", exc_info=True)
                        raise
    
    def encode(self, text: str) -> list[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            ImportError: If sentence-transformers is not installed
            Exception: If encoding fails
        """
        self._load_model()
        
        try:
            # sentence-transformers returns numpy array, convert to list
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error encoding text: {e}", exc_info=True)
            raise
    
    def encode_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (default: 32)
            
        Returns:
            List of embedding vectors
            
        Raises:
            ImportError: If sentence-transformers is not installed
            Exception: If encoding fails
        """
        self._load_model()
        
        try:
            # Process in batches for efficiency
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                show_progress_bar=len(texts) > 100
            )
            
            # Convert numpy arrays to lists
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Error encoding batch: {e}", exc_info=True)
            raise
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        self._load_model()
        return self.model.get_sentence_embedding_dimension()
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        with cls._lock:
            cls._instance = None
