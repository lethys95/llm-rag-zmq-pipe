"""Primary response processing node wrapping PrimaryResponseHandler."""

import asyncio

from src.handlers.primary_response import PrimaryResponseHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.orchestration.node_registry_decorator import register_node
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.core.result import NodeResult, NodeStatus


@register_node
class PrimaryResponseNode(BaseNode):
    min_criticality: float = 0.0
    dependencies: list[str] = [
        "FormatAdvisorNode",
        "MemoryAdvisorNode",
        "NeedsAdvisorNode",
        "StrategyAdvisorNode",
    ]

    def __init__(self, primary_response_handler: PrimaryResponseHandler) -> None:
        super().__init__()
        self.handler = primary_response_handler

    def get_description(self) -> str:
        return (
            "Calls the primary LLM to generate the companion's response and writes broker.primary_response. "
            "The primary LLM is the creative, empathetic entity — all other nodes exist to inform it. "
            "It reads all broker.advisor_outputs (assembled guidance from memory, needs, strategy, format "
            "advisors), conversation history, and broker.dialogue_input. The richer the advisor context, "
            "the more targeted the response. Run LAST — after all relevant advisors have run. "
            "Run exactly once per turn; if broker.primary_response is already set, do not run again. "
            "If no advisors have run the response will still be generated, just without specialised guidance."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        dialogue_input = broker.dialogue_input
        if not dialogue_input:
            return NodeResult(
                status=NodeStatus.FAILED, error="No dialogue_input in broker"
            )

        try:
            response = await asyncio.to_thread(
                self.handler.generate_response,
                prompt=dialogue_input.content,
                broker=broker,
                system_prompt_override=dialogue_input.system_prompt_override,
            )
            broker.primary_response = response
            return NodeResult(status=NodeStatus.SUCCESS, data={"response": response})
        except Exception as e:
            return NodeResult(status=NodeStatus.FAILED, error=str(e))
