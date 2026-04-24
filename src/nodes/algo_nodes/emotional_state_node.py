from src.handlers.emotional_state import EmotionalStateHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class EmotionalStateNode(BaseNode):

    def __init__(self, emotional_state_handler: EmotionalStateHandler) -> None:
        super().__init__()
        self.handler = emotional_state_handler

    def get_description(self) -> str:
        return (
            "Assess the emotional state of the current message using VAD and categorical emotion scores. "
            "Run early — before MessageAnalysisNode so facts can be stamped with turn-level emotional context. "
            "Output informs NeedsAnalysis, ResponseStrategy, and MemoryEvaluation."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        if not broker.dialogue_input:
            return NodeResult(status=NodeStatus.FAILED, error="No dialogue_input in broker")

        result = self.handler.analyze(broker.dialogue_input.content)

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
