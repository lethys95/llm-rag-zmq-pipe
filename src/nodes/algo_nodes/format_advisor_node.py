from src.handlers.format_advisor import FormatAdvisorHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class FormatAdvisorNode(BaseNode):
    dependencies: list[str] = []
    min_criticality: float = 0.0

    def __init__(self, format_advisor_handler: FormatAdvisorHandler) -> None:
        super().__init__()
        self.handler = format_advisor_handler

    def get_description(self) -> str:
        return (
            "Produces format and length instructions for the primary LLM based on the delivery mode "
            "and appends an AdvisorOutput to broker.advisor_outputs. "
            "Spoken mode: short natural sentences, no markdown, no bullet lists, no emojis, "
            "no parenthetical asides, speech-rhythm pacing. "
            "Text mode: may use light markdown where it genuinely helps clarity, can be longer. "
            "Both modes: match conversational register (don't be formal if the user is casual), "
            "don't pad or repeat yourself. "
            "Run before PrimaryResponseNode. No broker dependencies — works from broker.dialogue_input.mode "
            "which is always populated. Always run once per turn."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        mode = broker.dialogue_input.mode if broker.dialogue_input else "spoken"
        output = self.handler.advise(mode=mode)
        broker.advisor_outputs.append(output)
        return NodeResult(
            status=NodeStatus.SUCCESS,
            metadata={"advisor": output.advisor, "mode": mode, "potency": output.potency},
        )
