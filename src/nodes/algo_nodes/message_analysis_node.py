from src.handlers.user_fact_extraction import UserFactExtractionHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class MessageAnalysisNode(BaseNode):
    """Extracts atomic facts about the user from the incoming message.

    EmotionalStateHandler is temporarily disconnected — emotional_state's
    role in the broker is under review. The handler and model exist but
    the field is not on the broker until the advisor architecture clarifies
    where classifier outputs should live.
    """

    def __init__(
        self,
        user_fact_extraction_handler: UserFactExtractionHandler,
    ) -> None:
        super().__init__()
        self.user_fact_extraction_handler = user_fact_extraction_handler

    def get_description(self) -> str:
        return (
            "Extract atomic facts about the user from their message and store "
            "them to Qdrant. Run early — facts inform memory retrieval and advisors."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        if not broker.dialogue_input:
            return NodeResult(status=NodeStatus.FAILED, error="No dialogue_input in broker")

        message = broker.dialogue_input.content
        speaker = broker.dialogue_input.speaker

        user_facts = self.user_fact_extraction_handler.extract(
            message, speaker, emotional_state=broker.emotional_state
        )
        broker.user_facts = user_facts

        if not user_facts:
            return NodeResult(status=NodeStatus.PARTIAL)

        return NodeResult(status=NodeStatus.SUCCESS)
