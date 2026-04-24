from src.handlers.needs_advisor import NeedsAdvisorHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class NeedsAdvisorNode(BaseNode):

    def __init__(self, needs_advisor_handler: NeedsAdvisorHandler) -> None:
        super().__init__()
        self.handler = needs_advisor_handler

    def get_description(self) -> str:
        return (
            "Translate needs analysis into natural language guidance for the primary LLM. "
            "Run after NeedsAnalysisNode. Potency reflects need urgency — low for casual "
            "messages, high when the user is in distress. Skip if NeedsAnalysisNode didn't run."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        output = self.handler.advise(needs_analysis=broker.needs_analysis)
        broker.advisor_outputs.append(output)
        return NodeResult(
            status=NodeStatus.SUCCESS,
            metadata={"advisor": output.advisor, "potency": output.potency},
        )
