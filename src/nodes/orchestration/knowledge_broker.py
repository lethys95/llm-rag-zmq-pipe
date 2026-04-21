"""Knowledge broker for accumulating context across nodes."""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from src.models.sentiment import DialogueInput, SentimentAnalysis
from src.models.memory import TrustAnalysis
from src.models.analysis import MemoryEvaluation, NeedsAnalysis
from src.rag.selector import RAGDocument

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetadata:
    """Tracks node execution lifecycle."""

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_nodes_executed: int = 0
    failed_nodes: list[str] = field(default_factory=list)
    skipped_nodes: list[str] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)
    durations: dict[str, float] = field(default_factory=dict)

    def record_execution(
        self, node_name: str, status: str, duration: float | None = None
    ) -> None:
        """Record node execution details."""
        self.execution_order.append(node_name)
        self.total_nodes_executed += 1

        if status == "failed":
            self.failed_nodes.append(node_name)
        elif status == "skipped":
            self.skipped_nodes.append(node_name)

        if duration is not None:
            self.durations[node_name] = duration
            logger.debug(f"Recorded '{node_name}' ({status}) in {duration:.3f}s")
        else:
            logger.debug(f"Recorded '{node_name}' ({status})")


@dataclass
class KnowledgeBroker:
    """Central knowledge pool with strongly-typed fields.

    Direct attribute access replaces stringly-typed get/add methods.
    Each field represents data produced by specific nodes in the pipeline.
    """

    dialogue_input: DialogueInput | None = None
    sentiment_analysis: SentimentAnalysis | None = None
    primary_response: str | None = None
    ack_status: str | None = None
    ack_message: str | None = None
    zmq_identity: list[bytes] | None = None
    idle_time_minutes: float | None = None

    # Memory-related fields
    retrieved_documents: list[RAGDocument] = field(default_factory=list)
    evaluated_memories: list[tuple[RAGDocument, MemoryEvaluation]] = field(
        default_factory=list
    )
    conversation_history: list[SentimentAnalysis] = field(default_factory=list)

    # Trust analysis field
    trust_analysis: TrustAnalysis | None = None

    # Needs analysis field
    needs_analysis: NeedsAnalysis | None = None

    # Detox results field
    detox_results: dict[str, object] = field(default_factory=dict)

    metadata: ExecutionMetadata = field(default_factory=ExecutionMetadata)

    def __post_init__(self) -> None:
        logger.debug("Knowledge broker initialized")

    def record_node_execution(
        self, node_name: str, status: str, duration: float | None = None
    ) -> None:
        """Delegate to metadata tracker."""
        self.metadata.record_execution(node_name, status, duration)

    def get_execution_summary(self) -> dict[str, object]:
        """Return execution summary as dict for serialization."""
        return {
            "total_nodes_executed": self.metadata.total_nodes_executed,
            "execution_order": self.metadata.execution_order.copy(),
            "failed_nodes": self.metadata.failed_nodes.copy(),
            "skipped_nodes": self.metadata.skipped_nodes.copy(),
        }

    def get_analyzed_context(self) -> dict:
        """Get all analyzed context from various nodes.

        Consolidates data from sentiment_analysis, evaluated_memories,
        trust_analysis, needs_analysis, and other nodes into a single context
        dictionary for use in primary response generation.

        Returns:
            Dictionary with all analyzed context data
        """
        context = {}

        if self.sentiment_analysis:
            context["sentiment"] = self.sentiment_analysis

        if self.evaluated_memories:
            context["evaluated_memories"] = [
                (doc, evaluation) for doc, evaluation in self.evaluated_memories
            ]

        if self.trust_analysis:
            context["trust_analysis"] = self.trust_analysis

        if self.needs_analysis:
            context["needs_analysis"] = self.needs_analysis

        if self.conversation_history:
            context["conversation_history"] = self.conversation_history

        if self.idle_time_minutes is not None:
            context["idle_time_minutes"] = self.idle_time_minutes

        if self.detox_results:
            context["detox_results"] = self.detox_results

        return context

    def __repr__(self) -> str:
        field_count = sum(
            1
            for field_name in [
                "dialogue_input",
                "sentiment_analysis",
                "primary_response",
                "ack_status",
                "ack_message",
                "zmq_identity",
                "idle_time_minutes",
                "retrieved_documents",
                "evaluated_memories",
                "conversation_history",
                "trust_analysis",
                "needs_analysis",
                "detox_results",
            ]
            if getattr(self, field_name) is not None
        )
        return f"<KnowledgeBroker fields={field_count}>"
