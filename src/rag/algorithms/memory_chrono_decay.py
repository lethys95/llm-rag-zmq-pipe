"""Memory decay algorithms for time-based RAG document scoring."""

import logging
from datetime import datetime, timezone
from math import exp, log

from src.rag.selector import RAGDocument

logger = logging.getLogger(__name__)


# Standalone functions for mathematical calculations
# These are used by the MemoryDecayAlgorithm class and can be tested independently


def calculate_time_decay(
    created_at: datetime, current_time: datetime, half_life_days: float
) -> float:
    """Calculate exponential time decay for a memory.

    Args:
        created_at: When the memory was created
        current_time: Current reference time
        half_life_days: Time in days for memory to decay to 50%

    Returns:
        Decay factor between 0.0 and 1.0
    """
    # Calculate age in days
    age = current_time - created_at
    age_days = age.total_seconds() / 86400.0

    # Handle edge case: future memories
    if age_days < 0:
        return 1.0

    # Exponential decay: 2^(-age/half_life)
    # Equivalent to: e^(-age * ln(2)/half_life)
    lambda_decay = log(2) / half_life_days
    decay = exp(-lambda_decay * age_days)

    return decay


def calculate_access_boost(access_count: int, retrieval_count: int) -> float:
    """Calculate boost to memory score based on access frequency.

    Args:
        access_count: Number of times this specific memory was accessed
        retrieval_count: Total number of retrievals performed

    Returns:
        Boost factor (0.0+)
    """
    if access_count == 0:
        return 0.0

    # Logarithmic boost based on both access count and retrieval context
    # More accesses = higher boost
    # More total retrievals = higher boost (memory is more important in active system)

    # Base boost from access count
    access_factor = log(1 + access_count)

    # Additional boost from retrieval context
    if retrieval_count > 0:
        retrieval_factor = log(1 + retrieval_count / 10.0)
    else:
        retrieval_factor = 0.0

    boost = access_factor + retrieval_factor

    return boost


def calculate_memory_score(
    created_at: datetime,
    current_time: datetime,
    half_life_days: float,
    access_count: int,
    retrieval_count: int,
    chrono_weight: float,
) -> float:
    """Calculate combined memory score with time decay and access boost.

    Args:
        created_at: When the memory was created
        current_time: Current reference time
        half_life_days: Time in days for memory to decay to 50%
        access_count: Number of times this memory was accessed
        retrieval_count: Total number of retrievals performed
        chrono_weight: Weight for chronological decay (0.0-1.0)
            Higher = decay matters more, access boost matters less

    Returns:
        Combined memory score
    """
    # Calculate components
    time_decay = calculate_time_decay(created_at, current_time, half_life_days)
    access_boost = calculate_access_boost(access_count, retrieval_count)

    # Combine with weighted average
    # chrono_weight controls the balance between time decay and access boost
    chrono_component = time_decay * chrono_weight
    access_component = (1.0 + access_boost) * (1.0 - chrono_weight)

    score = chrono_component + access_component

    return score


class MemoryDecayAlgorithm:
    """Algorithm for calculating time-decayed memory scores.

    This algorithm mimics human memory by applying exponential decay
    to document relevance based on age and chrono_relevance. Documents
    with high chrono_relevance (e.g., family loss) persist much longer
    than documents with low chrono_relevance (e.g., breakfast menu).
    """

    def __init__(
        self,
        memory_half_life_days: float = 30.0,
        chrono_weight: float = 1.0,
        retrieval_threshold: float = 0.15,
        prune_threshold: float = 0.05,
        max_documents: int = 25,
    ):
        """Initialize the memory decay algorithm.

        Args:
            memory_half_life_days: Time in days for 50% decay at mid chrono_relevance.
                Higher = memories last longer overall.
            chrono_weight: Multiplier for chrono_relevance effect (0.0-2.0).
                1.0 = normal, >1.0 = amplify differences, <1.0 = dampen differences.
            retrieval_threshold: Minimum score to include in retrieval (0.0-1.0)
            prune_threshold: Minimum score to keep in database (0.0-1.0)
            max_documents: Maximum documents to return after scoring
        """
        self.memory_half_life_days = memory_half_life_days
        self.chrono_weight = chrono_weight
        self.retrieval_threshold = retrieval_threshold
        self.prune_threshold = prune_threshold
        self.max_documents = max_documents

        # Calculate lambda from half-life
        self.lambda_decay = log(2) / memory_half_life_days

        logger.info(
            f"Memory decay algorithm initialized: half_life={memory_half_life_days}d, "
            f"chrono_weight={chrono_weight}, threshold={retrieval_threshold}"
        )

    def calculate_memory_score(
        self,
        relevance: float,
        chrono_relevance: float,
        timestamp: datetime,
        current_time: datetime | None = None,
    ) -> float:
        """Calculate time-decayed memory score for a single document.

        Uses exponential decay where high chrono_relevance = low decay rate.
        Mimics human memory: important events persist, mundane things fade.

        Args:
            relevance: Base importance/impact (0.0-1.0)
            chrono_relevance: Temporal persistence (0.0-1.0)
                High (0.9) = persists long (family loss)
                Low (0.1) = fades fast (breakfast menu)
            timestamp: When the document was created
            current_time: Reference time (defaults to now)

        Returns:
            Memory score (0.0-1.0)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Calculate age in days
        age = current_time - timestamp
        age_days = age.total_seconds() / 86400.0  # Convert to days

        # Handle edge cases
        if age_days < 0:
            logger.warning(f"Document from future detected: {timestamp}")
            age_days = 0

        # Apply chrono_weight to amplify or dampen effect
        weighted_chrono = chrono_relevance * self.chrono_weight

        # Clamp to [0, 1] to prevent inversion
        weighted_chrono = max(0.0, min(1.0, weighted_chrono))

        # Inverse: high chrono_relevance = low decay rate
        decay_rate = 1.0 - weighted_chrono

        # Exponential decay: e^(-decay_rate * lambda * age)
        decay_multiplier = exp(-decay_rate * self.lambda_decay * age_days)

        # Final score
        score = relevance * decay_multiplier

        logger.debug(
            f"Memory score: relevance={relevance:.3f}, chrono={chrono_relevance:.3f}, "
            f"age={age_days:.1f}d → score={score:.3f}"
        )

        return score

    def score_document(
        self, document: RAGDocument, current_time: datetime | None = None
    ) -> float:
        """Calculate memory score for a RAGDocument.

        Extracts relevance, chrono_relevance, and timestamp from document metadata.

        Args:
            document: Document to score
            current_time: Reference time (defaults to now)

        Returns:
            Memory score (0.0-1.0), or document.score if metadata missing
        """
        # Try to extract sentiment metadata
        relevance = document.metadata.get("relevance")
        chrono_relevance = document.metadata.get("chrono_relevance")
        timestamp_str = document.metadata.get("timestamp")

        # If metadata is missing, return original similarity score
        if relevance is None or chrono_relevance is None or timestamp_str is None:
            logger.debug(
                f"Document missing metadata, using raw similarity score: {document.score:.3f}"
            )
            return document.score

        # Parse timestamp
        try:
            if isinstance(timestamp_str, str):
                # Handle ISO format with or without timezone
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                # Make timezone-aware if naive
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            elif isinstance(timestamp_str, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp_str, tz=timezone.utc)
            elif isinstance(timestamp_str, datetime):
                timestamp = timestamp_str
                # Make timezone-aware if naive
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                logger.warning(f"Unknown timestamp format: {type(timestamp_str)}")
                return document.score
        except (ValueError, OSError) as e:
            logger.warning(f"Failed to parse timestamp: {e}")
            return document.score

        # Calculate memory score
        return self.calculate_memory_score(
            relevance=relevance,
            chrono_relevance=chrono_relevance,
            timestamp=timestamp,
            current_time=current_time,
        )

    def score_documents(
        self, documents: list[RAGDocument], current_time: datetime | None = None
    ) -> list[tuple[RAGDocument, float]]:
        """Batch score all documents with memory decay.

        Args:
            documents: List of documents to score
            current_time: Reference time (defaults to now)

        Returns:
            List of (document, memory_score) tuples
        """
        scored = []
        for doc in documents:
            score = self.score_document(doc, current_time)
            scored.append((doc, score))

        logger.debug(f"Scored {len(documents)} documents")
        return scored

    def filter_and_rank(
        self,
        documents: list[RAGDocument],
        threshold: float | None = None,
        max_docs: int | None = None,
        current_time: datetime | None = None,
    ) -> list[RAGDocument]:
        """Filter documents by memory score and return top N.

        This is the main entry point for retrieval filtering.

        Args:
            documents: Documents to filter and rank
            threshold: Minimum score (defaults to self.retrieval_threshold)
            max_docs: Maximum documents to return (defaults to self.max_documents)
            current_time: Reference time (defaults to now)

        Returns:
            Filtered and ranked documents
        """
        if threshold is None:
            threshold = self.retrieval_threshold

        if max_docs is None:
            max_docs = self.max_documents

        # Score all documents
        scored = self.score_documents(documents, current_time)

        # Filter by threshold
        filtered = [(doc, score) for doc, score in scored if score >= threshold]

        # Sort by score descending
        filtered.sort(key=lambda x: x[1], reverse=True)

        # Limit to max_docs
        limited = filtered[:max_docs]

        logger.info(
            f"Memory filtering: {len(documents)} → {len(filtered)} (threshold) → "
            f"{len(limited)} (top {max_docs})"
        )

        # Return just the documents
        return [doc for doc, score in limited]

    def identify_prunable(
        self,
        documents: list[RAGDocument],
        prune_threshold: float | None = None,
        current_time: datetime | None = None,
    ) -> list[str]:
        """Identify documents that should be pruned from the database.

        Returns point IDs of documents with memory score below prune_threshold.
        These can be deleted from the vector database.

        Args:
            documents: Documents to evaluate for pruning
            prune_threshold: Minimum score to keep (defaults to self.prune_threshold)
            current_time: Reference time (defaults to now)

        Returns:
            List of point IDs to delete
        """
        if prune_threshold is None:
            prune_threshold = self.prune_threshold

        # Score all documents
        scored = self.score_documents(documents, current_time)

        # Find docs below threshold
        prunable_ids = []
        for doc, score in scored:
            if score < prune_threshold:
                point_id = doc.metadata.get("point_id")
                if point_id:
                    prunable_ids.append(str(point_id))

        logger.info(
            f"Identified {len(prunable_ids)} documents for pruning "
            f"(threshold: {prune_threshold})"
        )

        return prunable_ids

    def get_decay_stats(
        self, documents: list[RAGDocument], current_time: datetime | None = None
    ) -> dict:
        """Get statistics about memory decay for a set of documents.

        Useful for debugging and tuning parameters.

        Args:
            documents: Documents to analyze
            current_time: Reference time (defaults to now)

        Returns:
            Dictionary with statistics
        """
        if not documents:
            return {
                "total_documents": 0,
                "avg_memory_score": 0.0,
                "above_retrieval_threshold": 0,
                "below_prune_threshold": 0,
            }

        scored = self.score_documents(documents, current_time)
        scores = [score for _, score in scored]

        return {
            "total_documents": len(documents),
            "avg_memory_score": sum(scores) / len(scores),
            "min_memory_score": min(scores),
            "max_memory_score": max(scores),
            "above_retrieval_threshold": sum(
                1 for s in scores if s >= self.retrieval_threshold
            ),
            "below_prune_threshold": sum(1 for s in scores if s < self.prune_threshold),
            "memory_half_life_days": self.memory_half_life_days,
            "chrono_weight": self.chrono_weight,
        }
