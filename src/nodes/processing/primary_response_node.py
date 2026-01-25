"""Primary response processing node wrapping PrimaryResponseHandler."""

from src.handlers.primary_response import PrimaryResponseHandler
from src.nodes.core.base import BaseNode
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.core.result import NodeResult, NodeStatus


class PrimaryResponseNode(BaseNode):
    """Node that generates primary response using provided handler."""

    def __init__(self, handler: PrimaryResponseHandler, **kwargs):
        super().__init__(
            name="primary_response", priority=5, queue_type="immediate", **kwargs
        )
        self.handler = handler

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        dialogue_input = broker.dialogue_input
        if not dialogue_input:
            return NodeResult(
                status=NodeStatus.FAILED, error="No dialogue_input in broker"
            )

        try:
            # Get analyzed context from broker
            analyzed_context = broker.get_analyzed_context()
            
            response = self.handler.generate_response(
                prompt=dialogue_input.content,
                use_rag=False,
                system_prompt_override=dialogue_input.system_prompt_override,
                analyzed_context=analyzed_context,
                broker=broker
            )
            broker.primary_response = response
            return NodeResult(status=NodeStatus.SUCCESS, data={"response": response})
        except Exception as e:
            return NodeResult(status=NodeStatus.FAILED, error=str(e))