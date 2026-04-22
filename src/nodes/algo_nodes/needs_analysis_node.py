from src.handlers.needs_analysis import NeedsAnalysisHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class NeedsAnalysisNode(BaseNode):

    def __init__(self, needs_analysis_handler: NeedsAnalysisHandler) -> None:
        super().__init__()
        self.handler = needs_analysis_handler

    def get_description(self) -> str:
        return (
            "Identify the user's active psychological needs using Maslow's framework. "
            "Requires emotional state and memory retrieval to have run first. "
            "Output informs response strategy selection."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        if not broker.dialogue_input:
            return NodeResult(status=NodeStatus.FAILED, error="No dialogue_input in broker")

        result = self.handler.analyze(
            message=broker.dialogue_input.content,
            speaker=broker.dialogue_input.speaker,
            emotional_state=broker.emotional_state,
            retrieved_documents=broker.retrieved_documents,
        )

        if result is None:
            return NodeResult(status=NodeStatus.FAILED, error="NeedsAnalysisHandler returned None")

        broker.needs_analysis = result
        return NodeResult(
            status=NodeStatus.SUCCESS,
            metadata={"primary_needs": result.primary_needs, "urgency": result.need_urgency},
        )
