import asyncio

from src.handlers.memory_evaluation import MemoryEvaluationHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class MemoryEvaluationNode(BaseNode):
    dependencies: list[str] = ["MemoryRetrievalNode"]
    min_criticality: float = 0.2

    def __init__(self, memory_evaluation_handler: MemoryEvaluationHandler) -> None:
        super().__init__()
        self.handler = memory_evaluation_handler

    def get_description(self) -> str:
        return (
            "Re-ranks broker.retrieved_documents using LLM judgment and writes broker.evaluated_memories "
            "as a list of (document, MemoryEvaluation) pairs. Each MemoryEvaluation carries: "
            "relevance score (how directly this memory bears on the current message), "
            "chrono_relevance (is this still true today?), and reasoning (why this memory matters now). "
            "If broker.emotional_state is present, evaluation weights emotional resonance — "
            "memories from similar emotional moments score higher. "
            "Without this node, MemoryAdvisorNode falls back to raw similarity scores. "
            "Skip if broker.retrieved_documents is empty (returns SUCCESS with evaluated=0)."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        if not broker.dialogue_input:
            return NodeResult(status=NodeStatus.FAILED, error="No dialogue_input in broker")

        if not broker.retrieved_documents:
            broker.evaluated_memories = []
            return NodeResult(status=NodeStatus.SUCCESS, metadata={"evaluated": 0})

        results = await asyncio.to_thread(
            self.handler.evaluate,
            message=broker.dialogue_input.content,
            documents=broker.retrieved_documents,
            emotional_state=broker.emotional_state,
        )

        broker.evaluated_memories = results
        return NodeResult(
            status=NodeStatus.SUCCESS,
            metadata={"evaluated": len(results)},
        )
