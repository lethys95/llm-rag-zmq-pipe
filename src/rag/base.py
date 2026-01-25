"""Abstract base class for RAG providers."""

from abc import ABC, abstractmethod
from .selector import RAGDocument


class BaseRAG(ABC):
    """Abstract base class for RAG (Retrieval-Augmented Generation) providers.

    This class defines the interface that all RAG providers must implement,
    ensuring loose coupling and enabling easy integration of different
    retrieval systems (e.g., Qdrant, Pinecone, etc.).
    """

    @abstractmethod
    def retrieve(self, query: str) -> str:
        """Retrieve relevant context for the given query.

        Args:
            query: The search query to retrieve context for

        Returns:
            Retrieved context as a string, to be included in the LLM prompt

        Raises:
            Exception: If retrieval fails
        """
        pass

    @abstractmethod
    def retrieve_documents(self, query: str, top_k: int = 5) -> list["RAGDocument"]:
        """Retrieve relevant documents as structured objects.

        This method returns RAGDocument objects instead of formatted strings,
        allowing for more advanced processing (e.g., context interpretation).

        Args:
            query: The search query to retrieve documents for
            top_k: Maximum number of documents to retrieve

        Returns:
            List of RAGDocument objects with content, metadata, and scores

        Raises:
            Exception: If retrieval fails
        """
        pass

    @abstractmethod
    def store(
        self,
        text: str,
        embedding: list[float],
        metadata: dict[str, object] | None = None,
        point_id: str | None = None
    ) -> str:
        """Store a document with its embedding.
        
        Args:
            text: The text content to store
            embedding: The embedding vector for the text
            metadata: Optional metadata to store with the document
            point_id: Optional specific ID for the point (auto-generated if None)
            
        Returns:
            The ID of the stored point
        """
        pass
    
    @abstractmethod
    def update_access_count(self, point_id: str) -> None:
        """Increment access count for a document.
        
        Args:
            point_id: The point ID to update
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up resources and close connections.
        
        This method should be called when the RAG provider is no longer needed.
        """
        pass
