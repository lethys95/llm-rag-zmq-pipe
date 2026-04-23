"""Knowledge broker — shared workspace for a single request.

Not a pipeline. The coordinator decides which nodes run and in what order.
Nodes read from and write to this workspace freely. Fields are typed to
enforce what each node produces, not to imply execution sequence.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from src.models.sentiment import DialogueInput
from src.models.user_fact import UserFact
from src.models.analysis import MemoryEvaluation, NeedsAnalysis
from src.models.response_strategy import ResponseStrategy
from src.rag.selector import RAGDocument

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetadata:
    """Tracks node execution within a single request."""

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_nodes_executed: int = 0
    failed_nodes: list[str] = field(default_factory=list)
    skipped_nodes: list[str] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)
    durations: dict[str, float] = field(default_factory=dict)

    def record_execution(
        self, node_name: str, status: str, duration: float | None = None
    ) -> None:
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
    """Shared workspace for a single request turn.

    Every node reads from and writes to this object. The coordinator
    decides what runs and when — there is no implied sequence in the
    field ordering here.
    """

    # --- Input ---
    dialogue_input: DialogueInput | None = None
    zmq_identity: list[bytes] | None = None
    idle_time_minutes: float | None = None

    # --- Classifier outputs ---
    # emotional_state is commented out pending clarification of its role
    # in the advisor architecture. Classifiers produce it; advisors should
    # consume it. Whether it belongs as a first-class broker field or is
    # internal to advisor nodes is an open question.
    # emotional_state: EmotionalState | None = None

    user_facts: list[UserFact] = field(default_factory=list)

    # --- Memory ---
    retrieved_documents: list[RAGDocument] = field(default_factory=list)
    evaluated_memories: list[tuple[RAGDocument, MemoryEvaluation]] = field(
        default_factory=list
    )

    # --- Analysis ---
    needs_analysis: NeedsAnalysis | None = None
    response_strategy: ResponseStrategy | None = None

    # --- Detox ---
    # Placeholder — detox design is not settled. See PSYCHOLOGY.md.
    detox_results: dict[str, object] = field(default_factory=dict)

    # --- Output ---
    primary_response: str | None = None
    ack_status: str | None = None
    ack_message: str | None = None

    metadata: ExecutionMetadata = field(default_factory=ExecutionMetadata)

    def __post_init__(self) -> None:
        logger.debug("Knowledge broker initialized")

    def record_node_execution(
        self, node_name: str, status: str, duration: float | None = None
    ) -> None:
        self.metadata.record_execution(node_name, status, duration)

    def get_state_summary(self) -> str:
        """Produce a concise but substantive summary of what this turn has produced.

        Written for the coordinator — gives it enough signal to decide what
        to run next without dumping raw data.
        """
        lines = []

        if self.user_facts:
            claims = "; ".join(f.claim for f in self.user_facts[:5])
            lines.append(f"User facts ({len(self.user_facts)} extracted): {claims}")
        else:
            lines.append("User facts: none extracted yet")

        if self.retrieved_documents:
            lines.append(f"Retrieved memories: {len(self.retrieved_documents)} documents")
        else:
            lines.append("Retrieved memories: none")

        if self.evaluated_memories:
            lines.append(f"Evaluated memories: {len(self.evaluated_memories)} assessed")
        else:
            lines.append("Evaluated memories: not yet run")

        if self.needs_analysis:
            n = self.needs_analysis
            top = ", ".join(n.primary_needs) if n.primary_needs else "none"
            lines.append(
                f"Needs analysis: primary={top}, urgency={n.need_urgency:.2f} — \"{n.context_summary}\""
            )
        else:
            lines.append("Needs analysis: not yet run")

        if self.response_strategy:
            s = self.response_strategy
            lines.append(f"Response strategy: {s.approach} / {s.tone}")
        else:
            lines.append("Response strategy: not yet selected")

        if self.detox_results:
            lines.append("Calibration notes: present")

        if self.primary_response:
            lines.append(f"Primary response: generated ({len(self.primary_response)} chars)")
        else:
            lines.append("Primary response: not yet generated")

        return "\n".join(lines)

    def get_analyzed_context(self) -> dict:
        """Collect broker fields for use by the primary response handler.

        Returns only what is currently populated. Does not include raw
        classifier values that belong to the advisor layer.
        """
        context = {}

        if self.user_facts:
            context["user_facts"] = self.user_facts

        if self.evaluated_memories:
            context["evaluated_memories"] = self.evaluated_memories

        if self.needs_analysis:
            context["needs_analysis"] = self.needs_analysis

        if self.response_strategy:
            context["response_strategy"] = self.response_strategy

        if self.idle_time_minutes is not None:
            context["idle_time_minutes"] = self.idle_time_minutes

        if self.detox_results:
            context["detox_results"] = self.detox_results

        return context

    def get_execution_summary(self) -> dict[str, object]:
        return {
            "total_nodes_executed": self.metadata.total_nodes_executed,
            "execution_order": self.metadata.execution_order.copy(),
            "failed_nodes": self.metadata.failed_nodes.copy(),
            "skipped_nodes": self.metadata.skipped_nodes.copy(),
        }

    def __repr__(self) -> str:
        populated = [
            name for name in [
                "dialogue_input",
                "zmq_identity",
                "user_facts",
                "retrieved_documents",
                "evaluated_memories",
                "needs_analysis",
                "response_strategy",
                "primary_response",
            ]
            if getattr(self, name, None)
        ]
        return f"<KnowledgeBroker populated={populated}>"
