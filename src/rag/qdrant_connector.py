"""Qdrant vector database RAG implementation."""

import logging
from typing import Any
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
)

from .base import BaseRAG
from .selector import RAGDocument, RAGSelector
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class QdrantRAG(BaseRAG):
    """Qdrant-based RAG implementation.
    
    This implementation uses Qdrant vector database for storing and retrieving
    embeddings. It supports both local (in-memory/disk) and remote Qdrant instances.
    """
    
    def __init__(
        self,
        collection_name: str,
        embedding_dim: int = 384,
        url: str | None = None,
        api_key: str | None = None,
        path: str | None = None,
        distance: Distance = Distance.COSINE,
        selector: RAGSelector | None = None,
        embedding_service: EmbeddingService | None = None
    ):
        """Initialize the Qdrant RAG provider.
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_dim: Dimension of embeddings (default: 384 for all-MiniLM-L6-v2)
            url: URL for remote Qdrant instance (e.g., "http://localhost:6333")
            api_key: API key for remote Qdrant instance
            path: Path for local Qdrant storage (mutually exclusive with url)
            distance: Distance metric (COSINE, EUCLID, DOT, etc.)
            selector: RAGSelector for filtering/ranking results (optional)
            embedding_service: Optional EmbeddingService instance (will create singleton if None)
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.distance = distance
        self.selector = selector or RAGSelector(max_documents=5, min_score=0.5)
        
        # Initialize embedding service
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        
        # Initialize Qdrant client
        if url:
            logger.info(f"Connecting to remote Qdrant at {url}")
            self.client = QdrantClient(url=url, api_key=api_key)
        elif path:
            logger.info(f"Using local Qdrant storage at {path}")
            self.client = QdrantClient(path=path)
        else:
            logger.info("Using in-memory Qdrant storage")
            self.client = QdrantClient(":memory:")
        
        # Create collection if it doesn't exist
        self._ensure_collection()
        
        logger.info(f"Qdrant RAG initialized with collection '{collection_name}'")
    
    def _ensure_collection(self) -> None:
        """Ensure the collection exists, create if not."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection '{self.collection_name}'")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=self.distance
                    )
                )
                logger.info(f"Collection '{self.collection_name}' created successfully")
            else:
                logger.debug(f"Collection '{self.collection_name}' already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}", exc_info=True)
            raise
    
    def store(
        self,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        point_id: str | None = None
    ) -> str:
        """Store a document with its embedding in Qdrant.
        
        Args:
            text: The text content to store
            embedding: The embedding vector for the text
            metadata: Optional metadata to store with the document
            point_id: Optional specific ID for the point (auto-generated if None)
            
        Returns:
            The ID of the stored point
            
        Raises:
            Exception: If storage fails
        """
        try:
            # Prepare metadata with text content
            payload = metadata or {}
            payload["text"] = text
            payload["timestamp"] = payload.get("timestamp", datetime.now().isoformat())
            
            # Generate ID if not provided
            import uuid
            pid = point_id or str(uuid.uuid4())
            
            # Create point
            point = PointStruct(
                id=pid,
                vector=embedding,
                payload=payload
            )
            
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.debug(f"Stored document with ID: {pid}")
            return pid
            
        except Exception as e:
            logger.error(f"Error storing document: {e}", exc_info=True)
            raise
    
    def retrieve(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        filter_conditions: Filter | None = None
    ) -> str:
        """Retrieve relevant context for the given query.
        
        This method searches the vector database and returns formatted context.
        Note: If query_embedding is not provided, you must handle embedding
        generation externally and call retrieve_documents directly.
        
        Args:
            query: The search query
            query_embedding: Optional pre-computed embedding for the query
            limit: Maximum number of results to retrieve
            score_threshold: Minimum similarity score threshold
            filter_conditions: Optional Qdrant filter conditions
            
        Returns:
            Retrieved context as a formatted string
            
        Raises:
            ValueError: If query_embedding is None (embedding not implemented here)
        """
        if query_embedding is None:
            logger.warning(
                "No query embedding provided. Returning empty context. "
                "You must provide embeddings externally or implement an embedding model."
            )
            return ""
        
        # Retrieve documents using embedding
        documents = self.retrieve_documents_with_embedding(
            query_embedding=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions
        )
        
        # Select and format
        selected = self.selector.select(documents, query=query)
        formatted = self.selector.format_for_llm(selected)
        
        logger.info(f"Retrieved and formatted {len(selected)} documents for query")
        return formatted
    
    def retrieve_documents(
        self,
        query: str,
        top_k: int = 5
    ) -> list[RAGDocument]:
        """Retrieve relevant documents as structured objects.
        
        This method generates embeddings automatically using the embedding service
        and retrieves semantically similar documents from Qdrant.
        
        Args:
            query: The search query to retrieve documents for
            top_k: Maximum number of documents to retrieve
            
        Returns:
            List of RAGDocument objects
        """
        try:
            # Generate embedding for query
            logger.debug(f"Generating embedding for query: {query[:100]}...")
            query_embedding = self.embedding_service.encode(query)
            
            # Retrieve documents using embedding
            return self.retrieve_documents_with_embedding(
                query_embedding=query_embedding,
                limit=top_k
            )
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}", exc_info=True)
            return []
    
    def retrieve_documents_with_embedding(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float | None = None,
        filter_conditions: Filter | None = None
    ) -> list[RAGDocument]:
        """Retrieve documents from Qdrant as RAGDocument objects using embeddings.
        
        This method requires a pre-computed embedding vector.
        
        Args:
            query_embedding: The embedding vector for the query
            limit: Maximum number of results to retrieve
            score_threshold: Minimum similarity score threshold
            filter_conditions: Optional Qdrant filter conditions
            
        Returns:
            List of RAGDocument objects
            
        Raises:
            Exception: If retrieval fails
        """
        try:
            # Search Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_conditions
            )
            
            # Convert to RAGDocument objects
            documents = []
            for result in results:
                # Extract text from payload
                text = result.payload.get("text", "")
                
                # Build metadata (exclude text to avoid duplication)
                metadata = {k: v for k, v in result.payload.items() if k != "text"}
                metadata["point_id"] = result.id
                
                # Create RAGDocument
                doc = RAGDocument(
                    content=text,
                    score=result.score,
                    metadata=metadata
                )
                documents.append(doc)
            
            logger.debug(f"Retrieved {len(documents)} documents from Qdrant")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}", exc_info=True)
            raise
    
    def delete(self, point_ids: list[str]) -> None:
        """Delete documents from Qdrant by their IDs.
        
        Args:
            point_ids: List of point IDs to delete
            
        Raises:
            Exception: If deletion fails
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            logger.info(f"Deleted {len(point_ids)} documents")
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}", exc_info=True)
            raise
    
    def count(self) -> int:
        """Get the total number of documents in the collection.
        
        Returns:
            Number of documents
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            count = collection_info.points_count
            logger.debug(f"Collection contains {count} documents")
            return count
            
        except Exception as e:
            logger.error(f"Error counting documents: {e}", exc_info=True)
            return 0
    
    def close(self) -> None:
        """Clean up resources and close connections.
        
        Closes the Qdrant client connection.
        """
        logger.info("Closing Qdrant RAG connection")
        try:
            self.client.close()
        except Exception as e:
            logger.warning(f"Error closing Qdrant client: {e}")
