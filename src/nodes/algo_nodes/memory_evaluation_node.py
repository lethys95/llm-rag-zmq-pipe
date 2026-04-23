from src.handlers.memory_evaluation import MemoryEvaluationHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class MemoryEvaluationNode(BaseNode):

    def __init__(self, memory_evaluation_handler: MemoryEvaluationHandler) -> None:
        super().__init__()
        self.handler = memory_evaluation_handler

    def get_description(self) -> str:
        return (
            "Evaluate retrieved memories for relevance to the current message. "
            "Requires memory retrieval to have run first. "
            "Produces evaluated_memories for use by the primary response."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        if not broker.dialogue_input:
            return NodeResult(status=NodeStatus.FAILED, error="No dialogue_input in broker")

        if not broker.retrieved_documents:
            broker.evaluated_memories = []
            return NodeResult(status=NodeStatus.SUCCESS, metadata={"evaluated": 0})

        results = self.handler.evaluate(
            message=broker.dialogue_input.content,
            documents=broker.retrieved_documents,
            emotional_state=None,  # emotional_state broker field suspended — see knowledge_broker.py
        )

        broker.evaluated_memories = results
        return NodeResult(
            status=NodeStatus.SUCCESS,
            metadata={"evaluated": len(results)},
        )
