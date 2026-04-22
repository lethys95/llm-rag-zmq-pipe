from src.handlers.emotional_state import EmotionalStateHandler
from src.handlers.user_fact_extraction import UserFactExtractionHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class MessageAnalysisNode(BaseNode):
    """Analyses an incoming message for emotional state and extractable user facts.

    Runs two handlers in sequence:
    - EmotionalStateHandler  → broker.emotional_state  (session-scoped, VAD model)
    - UserFactExtractionHandler → broker.user_facts    (stored to Qdrant per fact)

    Partial success is acceptable — if one handler fails the other's output
    is still written to the broker.
    """

    def __init__(
        self,
        emotional_state_handler: EmotionalStateHandler,
        user_fact_extraction_handler: UserFactExtractionHandler,
    ) -> None:
        super().__init__()
        self.emotional_state_handler = emotional_state_handler
        self.user_fact_extraction_handler = user_fact_extraction_handler

    def get_description(self) -> str:
        return (
            "Analyse the user's message for emotional state (VAD model) and extract "
            "atomic facts about the user. Run early — output informs response strategy "
            "and memory retrieval."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        if not broker.dialogue_input:
            return NodeResult(status=NodeStatus.FAILED, error="No dialogue_input in broker")

        message = broker.dialogue_input.content
        speaker = broker.dialogue_input.speaker

        emotional_state = self.emotional_state_handler.analyze(message)
        user_facts = self.user_fact_extraction_handler.extract(message, speaker)

        broker.emotional_state = emotional_state
        broker.user_facts = user_facts

        if emotional_state is None and not user_facts:
            return NodeResult(status=NodeStatus.FAILED, error="Both handlers returned nothing")

        if emotional_state is None or not user_facts:
            return NodeResult(status=NodeStatus.PARTIAL)

        return NodeResult(status=NodeStatus.SUCCESS)
