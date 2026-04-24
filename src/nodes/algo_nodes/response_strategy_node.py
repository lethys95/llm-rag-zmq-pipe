from src.handlers.response_strategy import ResponseStrategyHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class ResponseStrategyNode(BaseNode):

    def __init__(self, response_strategy_handler: ResponseStrategyHandler) -> None:
        super().__init__()
        self.handler = response_strategy_handler

    def get_description(self) -> str:
        return (
            "Select a therapeutic response approach based on needs analysis and emotional state. "
            "Run after NeedsAnalysisNode. Output is injected into the primary response system prompt."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        if not broker.dialogue_input:
            return NodeResult(status=NodeStatus.FAILED, error="No dialogue_input in broker")

        result = self.handler.select(
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
