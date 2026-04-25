import asyncio

from src.handlers.emotional_state import EmotionalStateHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class EmotionalStateNode(BaseNode):
    dependencies: list[str] = []
    min_criticality: float = 0.0

    def __init__(self, emotional_state_handler: EmotionalStateHandler) -> None:
        super().__init__()
        self.handler = emotional_state_handler

    def get_description(self) -> str:
        return (
            "Classifies the emotional charge of the user's message and writes broker.emotional_state. "
            "Produces: valence (−1=very negative .. +1=very positive), arousal (0=calm .. 1=activated), "
            "dominance (0=submissive .. 1=in control), confidence score, and a human-readable emotional summary. "
            "Run this early — NeedsAnalysisNode, MemoryEvaluationNode, and MessageAnalysisNode all read "
            "broker.emotional_state to contextualise their own output. Without it, those nodes operate on "
            "surface text only. Skip only for internal/system events that carry no user emotional content."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        if not broker.dialogue_input:
            return NodeResult(status=NodeStatus.FAILED, error="No dialogue_input in broker")

        result = await asyncio.to_thread(self.handler.analyze, broker.dialogue_input.content)

        if result is None:
            return NodeResult(status=NodeStatus.PARTIAL, error="EmotionalStateHandler returned None")

        broker.emotional_state = result
        return NodeResult(
            status=NodeStatus.SUCCESS,
            metadata={
                "valence": result.valence,
                "arousal": result.arousal,
                "dominance": result.dominance,
                "confidence": result.confidence,
            },
        )
