"""Embedding service for generating text embeddings."""

from __future__ import annotations
from sentence_transformers import SentenceTransformer
import logging
from threading import Lock
from typing import Self, cast

import numpy as np
from numpy.typing import NDArray
from src.config.settings import Settings

logger = logging.getLogger(__name__)

# Is this singleton properly implemented?


class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers.

    This uses a singleton pattern to cache the model and avoid reloading.
    The default model is 'all-MiniLM-L6-v2' which produces 384-dimensional
    embeddings and is optimized for semantic similarity.
    """

    _instance: Self | None = None
    _lock: Lock = Lock()
    _settings: Settings = Settings()
    _model_name: str = _settings.rag_embedding_model
    _model_lock = Lock()
    
    _model: SentenceTransformer | None = None

    @classmethod
    def get_instance(cls) -> Self:
        """Get or create the singleton instance.

        Args:
            model_name: Model to use (only used on first creation)

        Returns:
            Singleton EmbeddingService instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    try:
                        logger.info(
                            "Loading sentence-transformers model: %s", self._model_name
                        )
                        self._model = SentenceTransformer(self._model_name)
                        logger.info("Model loaded successfully")
                    except ImportError:
                        logger.error(
                            "sentence-transformers not installed. "
                            "Install with: pip install sentence-transformers"
                        )
                        raise
                    except Exception as e:
                        logger.error(
                            "Error loading embedding model: %s", e, exc_info=True
                        )
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
            model = self._model
            if model is None:
                raise RuntimeError("Model not loaded")
            embedding = cast(
                NDArray[np.float32],
                model.encode(text, convert_to_tensor=False),  # type: ignore[reportUnknownMemberType]
            )
            return embedding.tolist()
        except Exception as e:
            logger.error("Error encoding text: %s", e, exc_info=True)
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
            model = self._model
            if model is None:
                raise RuntimeError("Model not loaded")
            # Process in batches for efficiency
            embeddings = cast(
                NDArray[np.float32],
                model.encode(  # type: ignore[reportUnknownMemberType]
                    texts,
                    batch_size=batch_size,
                    convert_to_tensor=False,
                    show_progress_bar=len(texts) > 100,
                ),
            )

            # Convert numpy arrays to lists
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error("Error encoding batch: %s", e, exc_info=True)
            raise

    def get_dimension(self) -> int | None:
        """Get the dimension of embeddings produced by this model.

        Returns:
            Embedding dimension
        """
        self._load_model()
        model = self._model
        if model is None:
            raise RuntimeError("Model not loaded")
        return model.get_sentence_embedding_dimension()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        with cls._lock:
            cls._instance = None
