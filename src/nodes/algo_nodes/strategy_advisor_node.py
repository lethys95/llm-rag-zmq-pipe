from src.handlers.strategy_advisor import StrategyAdvisorHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class StrategyAdvisorNode(BaseNode):
    dependencies: list[str] = ["ResponseStrategyNode"]
    min_criticality: float = 0.3

    def __init__(self, strategy_advisor_handler: StrategyAdvisorHandler) -> None:
        super().__init__()
        self.handler = strategy_advisor_handler

    def get_description(self) -> str:
        return (
            "Translates broker.response_strategy into concrete behavioural instructions for the primary LLM "
            "and appends an AdvisorOutput to broker.advisor_outputs. "
            "Instructions are approach-specific: for motivational interviewing, tells the primary to use "
            "open questions, reflective listening, and to avoid the righting reflex; for validation, "
            "tells it to reflect the emotion before doing anything else; for psychoeducation, tells it "
            "how to frame information without being patronising; and so on. "
            "Potency scales with need_urgency from broker.needs_analysis (if present). "
            "Must run after ResponseStrategyNode — without broker.response_strategy the advice is empty. "
            "Run alongside NeedsAdvisorNode for full coverage of what the primary LLM needs to know."
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
