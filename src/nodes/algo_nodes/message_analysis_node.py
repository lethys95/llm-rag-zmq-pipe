import asyncio

from src.handlers.user_fact_extraction import UserFactExtractionHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class MessageAnalysisNode(BaseNode):
    dependencies: list[str] = ["EmotionalStateNode"]  # optional but stamps facts with VAD
    min_criticality: float = 0.0
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
            "Extracts atomic facts stated or strongly implied by the user's message and writes them to "
            "broker.user_facts. Each fact carries: claim (a concise declarative statement about the user), "
            "chrono_relevance (0.0=ephemeral, 1.0=permanent life fact), and subject category. "
            "If EmotionalStateNode has already run, facts are also stamped with the turn's VAD scores so "
            "the memory layer can later surface emotionally resonant history. "
            "Facts are persisted to Qdrant after the response is sent (via ConversationStorage) — they are "
            "NOT stored inline by this node. Skip for messages that convey no information about the user "
            "(e.g. pure greetings, clarification requests with no self-disclosure)."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        if not broker.dialogue_input:
            return NodeResult(status=NodeStatus.FAILED, error="No dialogue_input in broker")

        message = broker.dialogue_input.content
        speaker = broker.dialogue_input.speaker

        user_facts = await asyncio.to_thread(
            self.user_fact_extraction_handler.extract,
            message, speaker, emotional_state=broker.emotional_state,
        )
        broker.user_facts = user_facts

        if not user_facts:
            return NodeResult(status=NodeStatus.PARTIAL)

        return NodeResult(status=NodeStatus.SUCCESS)
