import asyncio

from src.handlers.response_strategy import ResponseStrategyHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class ResponseStrategyNode(BaseNode):
    dependencies: list[str] = ["NeedsAnalysisNode"]
    min_criticality: float = 0.3

    def __init__(self, response_strategy_handler: ResponseStrategyHandler) -> None:
        super().__init__()
        self.handler = response_strategy_handler

    def get_description(self) -> str:
        return (
            "Selects the therapeutic/communicative approach best suited to this turn and writes "
            "broker.response_strategy. Approach options include: active listening, validation, "
            "psychoeducation, motivational interviewing, problem-solving, gentle challenge, "
            "normalisation, and others. Also selects tone (warm, playful, direct, gentle) "
            "and pacing guidance. The selection is driven by broker.needs_analysis (primary_needs + "
            "need_urgency) and broker.emotional_state (valence/arousal). "
            "StrategyAdvisorNode translates this output into concrete behavioural instructions for the "
            "primary LLM — run both together. Without needs_analysis, the strategy defaults to generic "
            "active listening and will not reflect the user's actual state."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        if not broker.dialogue_input:
            return NodeResult(status=NodeStatus.FAILED, error="No dialogue_input in broker")

        result = await asyncio.to_thread(
            self.handler.select,
            needs_analysis=broker.needs_analysis,
            emotional_state=broker.emotional_state,
        )

        if result is None:
            return NodeResult(status=NodeStatus.FAILED, error="ResponseStrategyHandler returned None")

        broker.response_strategy = result
        return NodeResult(
            status=NodeStatus.SUCCESS,
            metadata={"approach": result.approach, "tone": result.tone},
        )
