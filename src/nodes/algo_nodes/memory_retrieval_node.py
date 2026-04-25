import asyncio

from src.handlers.memory_retrieval import MemoryRetrievalHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class MemoryRetrievalNode(BaseNode):
    dependencies: list[str] = []
    min_criticality: float = 0.0

    def __init__(self, memory_retrieval_handler: MemoryRetrievalHandler) -> None:
        super().__init__()
        self.handler = memory_retrieval_handler

    def get_description(self) -> str:
        return (
            "Queries Qdrant for memories (past conversation turns and extracted user facts) semantically "
            "similar to the current message, ranked by a combined score of embedding similarity and "
            "time-decay-adjusted chrono_relevance. Writes results to broker.retrieved_documents. "
            "These raw documents feed MemoryEvaluationNode (which re-ranks them with LLM judgment) "
            "and MemoryAdvisorNode (which synthesises them into primary LLM guidance). "
            "NeedsAnalysisNode also reads retrieved_documents to detect patterns across history. "
            "Returns PARTIAL (not FAILED) if no relevant memories exist — the user may simply be new. "
            "Can run in parallel with EmotionalStateNode and MessageAnalysisNode — has no broker deps."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        if not broker.dialogue_input:
            return NodeResult(status=NodeStatus.FAILED, error="No dialogue_input in broker")

        docs = await asyncio.to_thread(
            self.handler.retrieve,
            query=broker.dialogue_input.content,
            memory_owner=broker.dialogue_input.speaker,
        )

        broker.retrieved_documents = docs
        if not docs:
            return NodeResult(status=NodeStatus.PARTIAL, metadata={"retrieved": 0})
        return NodeResult(status=NodeStatus.SUCCESS, metadata={"retrieved": len(docs)})
