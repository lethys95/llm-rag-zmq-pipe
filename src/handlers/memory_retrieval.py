import logging

from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.rag.base import BaseRAG
from src.rag.embeddings import EmbeddingService
from src.rag.algorithms.memory_chrono_decay import MemoryDecayAlgorithm
from src.rag.selector import RAGDocument

logger = logging.getLogger(__name__)

# Retrieve more than the final limit from Qdrant to give the decay
# algorithm enough candidates to work with.
_QDRANT_FETCH_MULTIPLIER = 3


class MemoryRetrievalHandler:
    """Retrieves and ranks memories relevant to the current message.

    Flow:
        1. Encode the query with the embedding service
        2. Fetch candidates from Qdrant, optionally filtered by memory_owner
        3. Re-rank and prune with MemoryDecayAlgorithm
        4. Return the top N documents
    """

    def __init__(
        self,
        rag: BaseRAG,
        embedding_service: EmbeddingService,
        memory_decay: MemoryDecayAlgorithm,
    ) -> None:
        self.rag = rag
        self.embedding_service = embedding_service
        self.memory_decay = memory_decay

    def retrieve(self, query: str, memory_owner: str | None = None) -> list[RAGDocument]:
        """Retrieve memories relevant to query, filtered by owner and ranked by decay score.

        Args:
            query: The user's message or enriched query string
            memory_owner: If provided, only retrieve memories belonging to this speaker

        Returns:
            Ranked list of relevant RAGDocuments after decay filtering
        """
        try:
            embedding = self.embedding_service.encode(query)

            filter_conditions = None
            if memory_owner:
                filter_conditions = Filter(
                    must=[FieldCondition(key="memory_owner", match=MatchValue(value=memory_owner))]
                )

            fetch_limit = self.memory_decay.max_documents * _QDRANT_FETCH_MULTIPLIER
            candidates = self.rag.retrieve_documents_with_embedding(
                query_embedding=embedding,
                limit=fetch_limit,
                filter_conditions=filter_conditions,
            )

            results = self.memory_decay.filter_and_rank(candidates)

            logger.debug(
                "Memory retrieval: %d candidates → %d after decay (owner=%s)",
                len(candidates), len(results), memory_owner,
            )
            return results

        except Exception:
            logger.exception("MemoryRetrievalHandler failed for query: %.80s", query)
            return []
