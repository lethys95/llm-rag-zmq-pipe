"""Ack preparation node for setting acknowledgment data."""

from src.nodes.core.base_node import BaseNode
from src.nodes.orchestration.node_registry_decorator import register_node
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.core.result import NodeResult, NodeStatus


# What is this?
@register_node
class AckPreparationNode(BaseNode):
    """Node that prepares acknowledgment message from sentiment data."""

    def __init__(self, **kwargs):
        super().__init__(name="ack_preparation", **kwargs)

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        sentiment = broker.sentiment_analysis
        ack_msg = "Request processed successfully"
        if sentiment:
            ack_msg += f" | Sentiment: {sentiment.sentiment}"
        broker.ack_status = "success"
        broker.ack_message = ack_msg
        return NodeResult(status=NodeStatus.SUCCESS)
