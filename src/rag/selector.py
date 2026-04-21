"""RAG result selection and pruning utilities."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class RAGDocument:
    """Represents a document retrieved from RAG.

    This is the expected structure for documents returned by RAG queries.
    """

    content: str
    score: float
    metadata: dict[str, str | float | int | bool | list | None]

    def __post_init__(self):
        """Validate the document after initialization."""
        if not 0.0 <= self.score <= 1.0:
            logger.warning(f"Score {self.score} outside expected range [0.0, 1.0]")


class RAGSelector:
    """Utility for selecting and pruning RAG results.

    This class provides methods to filter and rank RAG results based on
    various criteria, helping to select the most relevant context for
    LLM consumption without exceeding token limits.
    """

    def __init__(
        self,
        max_documents: int | None = None,
        min_score: float = 0.0,
        max_age_hours: float | None = None,
    ):
        """Initialize the RAG selector.

        Args:
            max_documents: Maximum number of documents to return (None = no limit)
            min_score: Minimum similarity score threshold (0.0-1.0)
            max_age_hours: Maximum age in hours for time-sensitive filtering (None = no limit)
        """
        self.max_documents = max_documents
        self.min_score = min_score
        self.max_age_hours = max_age_hours

        logger.info(
            f"RAG selector initialized: max_docs={max_documents}, "
            f"min_score={min_score}, max_age_hours={max_age_hours}"
        )

    def select(
        self, documents: list[RAGDocument], query: str | None = None
    ) -> list[RAGDocument]:
        """Select and prune documents based on configured criteria.

        This method applies multiple filtering strategies:
        1. Score-based filtering (remove low-relevance docs)
        2. Time-based filtering (remove stale docs if configured)
        3. Limit to max_documents (keep top-ranked)

        Args:
            documents: List of RAGDocument objects to filter
            query: Optional query string for advanced filtering

        Returns:
            Filtered and sorted list of RAGDocument objects
        """
        logger.debug(f"Selecting from {len(documents)} documents")

        if not documents:
            logger.debug("No documents to select from")
            return []

        # Apply filters sequentially
        filtered = self._filter_by_score(documents)
        filtered = self._filter_by_age(filtered)
        filtered = self._limit_count(filtered)

        logger.info(f"Selected {len(filtered)} documents from {len(documents)} total")

        return filtered

    def _filter_by_score(self, documents: list[RAGDocument]) -> list[RAGDocument]:
        """Filter documents by minimum score threshold.

        Args:
            documents: Documents to filter

        Returns:
            Documents meeting score threshold
        """
        filtered = [doc for doc in documents if doc.score >= self.min_score]

        if len(filtered) < len(documents):
            logger.debug(
                f"Score filtering removed {len(documents) - len(filtered)} documents "
                f"(threshold: {self.min_score})"
            )

        return filtered

    def _parse_timestamp(self, timestamp: str | float | int | None) -> datetime | None:
        """Parse a timestamp in various formats.

        Args:
            timestamp: Timestamp to parse (str, int, float, or other)

        Returns:
            Parsed datetime object, or None if parsing fails
        """
        if timestamp is None:
            return None

        try:
            if isinstance(timestamp, str):
                return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            elif isinstance(timestamp, (int, float)):
                return datetime.fromtimestamp(timestamp)
            else:
                return None
        except (ValueError, OSError) as e:
            logger.warning(f"Failed to parse timestamp {timestamp}: {e}")
            return None

    def _filter_by_age(self, documents: list[RAGDocument]) -> list[RAGDocument]:
        """Filter documents by maximum age.

        Args:
            documents: Documents to filter

        Returns:
            Documents within age threshold
        """
        if self.max_age_hours is None:
            return documents

        cutoff = datetime.now() - timedelta(hours=self.max_age_hours)
        filtered = []

        for doc in documents:
            timestamp = doc.metadata.get("timestamp")
            doc_time = self._parse_timestamp(timestamp)

            # Keep document if no timestamp or if it's within age limit
            if doc_time is None or doc_time >= cutoff:
                filtered.append(doc)

        if len(filtered) < len(documents):
            logger.debug(
                f"Age filtering removed {len(documents) - len(filtered)} documents "
                f"(max_age: {self.max_age_hours}h)"
            )

        return filtered

    def _limit_count(self, documents: list[RAGDocument]) -> list[RAGDocument]:
        """Limit documents to maximum count, keeping highest-scored.

        Args:
            documents: Documents to limit

        Returns:
            Top N documents by score
        """
        if self.max_documents is None or len(documents) <= self.max_documents:
            return documents

        # Sort by score descending and take top N
        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)
        limited = sorted_docs[: self.max_documents]

        logger.debug(
            f"Count limiting reduced {len(documents)} documents to {len(limited)}"
        )

        return limited

    def _calculate_recency_boost(
        self, doc_time: datetime, boost_factor: float, now: datetime | None = None
    ) -> float:
        """Calculate recency boost multiplier for a document.

        Args:
            doc_time: Document timestamp
            boost_factor: Maximum boost factor to apply
            now: Current time (defaults to datetime.now())

        Returns:
            Recency boost multiplier (1.0 = no boost)
        """
        if now is None:
            now = datetime.now()

        age_hours = (now - doc_time).total_seconds() / 3600

        # Boost documents less than 24 hours old
        if age_hours < 24:
            recency_multiplier = 1.0 + (1.0 - age_hours / 24) * (boost_factor - 1.0)
            logger.debug(
                f"Document age: {age_hours:.1f}h, boost: {recency_multiplier:.3f}"
            )
            return recency_multiplier

        return 1.0

    def _calculate_adjusted_score(
        self, doc: RAGDocument, boost_factor: float, now: datetime
    ) -> float:
        """Calculate adjusted score for a document with recency boost.

        Args:
            doc: Document to score
            boost_factor: Recency boost factor
            now: Current time

        Returns:
            Adjusted score
        """
        adjusted_score = doc.score
        timestamp = doc.metadata.get("timestamp")

        if timestamp:
            doc_time = self._parse_timestamp(timestamp)

            if doc_time:
                recency_multiplier = self._calculate_recency_boost(
                    doc_time, boost_factor, now
                )
                adjusted_score *= recency_multiplier

                if recency_multiplier > 1.0:
                    logger.debug(
                        f"Boosted recent doc: {doc.score:.3f} -> {adjusted_score:.3f}"
                    )

        return adjusted_score

    def rank_by_relevance(
        self,
        documents: list[RAGDocument],
        boost_recent: bool = False,
        recent_boost_factor: float = 1.2,
    ) -> list[RAGDocument]:
        """Re-rank documents with optional recency boosting.

        This can be used to adjust ranking after initial selection,
        for example to boost recent documents.

        Args:
            documents: Documents to rank
            boost_recent: Whether to boost recent documents
            recent_boost_factor: Multiplier for recent documents (e.g., 1.2 = 20% boost)

        Returns:
            Ranked documents (sorted by adjusted score)
        """
        if not boost_recent or not documents:
            return sorted(documents, key=lambda d: d.score, reverse=True)

        # Calculate recency-adjusted scores
        now = datetime.now()
        ranked = []

        for doc in documents:
            adjusted_score = self._calculate_adjusted_score(
                doc, recent_boost_factor, now
            )
            ranked.append((adjusted_score, doc))

        # Sort by adjusted score and return original documents
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked]

    def _filter_metadata(
        self, metadata: dict[str, str | float | int | bool | list | None]
    ) -> dict[str, str | float | int | bool | list | None]:
        """Filter out large metadata fields.

        Args:
            metadata: Metadata dictionary to filter

        Returns:
            Filtered metadata dictionary
        """
        # Skip large fields like embeddings and vectors
        excluded_fields = {"embedding", "vector"}
        return {k: v for k, v in metadata.items() if k not in excluded_fields}

    def _format_single_document(
        self, doc: RAGDocument, index: int, include_metadata: bool
    ) -> str:
        """Format a single document for LLM consumption.

        Args:
            doc: Document to format
            index: Document index (for numbering)
            include_metadata: Whether to include metadata

        Returns:
            Formatted document string
        """
        if include_metadata:
            filtered_meta = self._filter_metadata(doc.metadata)
            meta_str = ", ".join(f"{k}={v}" for k, v in filtered_meta.items())
            return f"[Document {index} | Score: {doc.score:.3f} | {meta_str}]\n{doc.content}\n"
        else:
            return f"{doc.content}\n"

    def format_for_llm(
        self, documents: list[RAGDocument], include_metadata: bool = False
    ) -> str:
        """Format selected documents for LLM consumption.

        Args:
            documents: Documents to format
            include_metadata: Whether to include metadata in output

        Returns:
            Formatted string suitable for LLM context
        """
        if not documents:
            return ""

        formatted_parts = [
            self._format_single_document(doc, i, include_metadata)
            for i, doc in enumerate(documents, 1)
        ]

        result = "\n".join(formatted_parts)
        logger.debug(
            f"Formatted {len(documents)} documents into {len(result)} characters"
        )

        return result
