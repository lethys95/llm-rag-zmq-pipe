import asyncio

from src.handlers.needs_analysis import NeedsAnalysisHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class NeedsAnalysisNode(BaseNode):
    dependencies: list[str] = ["EmotionalStateNode", "MemoryRetrievalNode"]
    min_criticality: float = 0.0

    def __init__(self, needs_analysis_handler: NeedsAnalysisHandler) -> None:
        super().__init__()
        self.handler = needs_analysis_handler

    def get_description(self) -> str:
        return (
            "Identifies which psychological needs are active in this turn and writes broker.needs_analysis. "
            "Uses a Maslow-adjacent framework: belonging, meaning, esteem, autonomy, safety, "
            "physiological stability, and growth. Produces: primary_needs (list of active needs), "
            "need_urgency (0.0=no urgency, 1.0=crisis-level), and context_summary (human-readable "
            "interpretation of what the user is actually dealing with). "
            "Also detects motivational interviewing signals: change talk, sustain talk, readiness stage, "
            "emotional disclosure level, and whether the user is seeking advice or just to be heard. "
            "Reads broker.emotional_state for valence/arousal cues (more accurate with it). "
            "Reads broker.retrieved_documents for pattern-detection across history. "
            "ResponseStrategyNode, NeedsAdvisorNode, and StrategyAdvisorNode all depend on this output."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        if not broker.dialogue_input:
            return NodeResult(status=NodeStatus.FAILED, error="No dialogue_input in broker")

        result = await asyncio.to_thread(
            self.handler.analyze,
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
