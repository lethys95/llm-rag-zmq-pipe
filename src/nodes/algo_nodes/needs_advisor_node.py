from src.handlers.needs_advisor import NeedsAdvisorHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class NeedsAdvisorNode(BaseNode):
    dependencies: list[str] = ["NeedsAnalysisNode"]
    min_criticality: float = 0.3

    def __init__(self, needs_advisor_handler: NeedsAdvisorHandler) -> None:
        super().__init__()
        self.handler = needs_advisor_handler

    def get_description(self) -> str:
        return (
            "Translates broker.needs_analysis into behavioural guidance for the primary LLM and "
            "appends an AdvisorOutput to broker.advisor_outputs. "
            "The advice tells the primary LLM what the user actually needs from this interaction: "
            "e.g. 'the user is showing elevated belonging need — acknowledge their experience and "
            "make them feel heard before offering any perspective', or 'autonomy need is high — "
            "avoid directive language, offer choices, follow the user's lead'. "
            "Potency reflects need_urgency: 0.0 for a casual light message, 1.0 for acute distress. "
            "Run together with StrategyAdvisorNode — they are complementary. "
            "If NeedsAnalysisNode did not run, the advice is empty and potency is 0."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        output = self.handler.advise(needs_analysis=broker.needs_analysis)
        broker.advisor_outputs.append(output)
        return NodeResult(
            status=NodeStatus.SUCCESS,
            metadata={"advisor": output.advisor, "potency": output.potency},
        )
