from src.handlers.strategy_advisor import StrategyAdvisorHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class StrategyAdvisorNode(BaseNode):

    def __init__(self, strategy_advisor_handler: StrategyAdvisorHandler) -> None:
        super().__init__()
        self.handler = strategy_advisor_handler

    def get_description(self) -> str:
        return (
            "Translate the selected response strategy into natural language guidance for the primary LLM. "
            "Must run after ResponseStrategyNode — it has no output without it. "
            "Pair it with NeedsAdvisorNode: if NeedsAdvisorNode ran, run this too. "
            "Potency scales with need urgency."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        need_urgency = broker.needs_analysis.need_urgency if broker.needs_analysis else 0.0
        output = self.handler.advise(
            response_strategy=broker.response_strategy,
            need_urgency=need_urgency,
        )
        broker.advisor_outputs.append(output)
        return NodeResult(
            status=NodeStatus.SUCCESS,
            metadata={"advisor": output.advisor, "potency": output.potency},
        )
