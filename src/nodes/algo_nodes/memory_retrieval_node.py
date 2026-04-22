from src.handlers.memory_retrieval import MemoryRetrievalHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class MemoryRetrievalNode(BaseNode):

    def __init__(self, memory_retrieval_handler: MemoryRetrievalHandler) -> None:
        super().__init__()
        self.handler = memory_retrieval_handler

    def get_description(self) -> str:
        return (
            "Retrieve past memories relevant to the current message, ranked by "
            "semantic similarity and time decay. Run before NeedsAnalysis and "
            "ResponseStrategy so they have historical context."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        if not broker.dialogue_input:
            return NodeResult(status=NodeStatus.FAILED, error="No dialogue_input in broker")

        docs = self.handler.retrieve(
            query=broker.dialogue_input.content,
            memory_owner=broker.dialogue_input.speaker,
        )

        broker.retrieved_documents = docs
        return NodeResult(status=NodeStatus.SUCCESS, metadata={"retrieved": len(docs)})
