"""Ack preparation node for setting acknowledgment data."""

from src.nodes.core.base import BaseNode
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.core.result import NodeResult, NodeStatus


class AckPreparationNode(BaseNode):
    """Node that prepares acknowledgment message from sentiment data."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="ack_preparation",
            priority=3,
            queue_type="immediate",
            dependencies=["receive_message"],
            **kwargs
        )
    
    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        sentiment = broker.sentiment_analysis
        ack_msg = "Request processed successfully"
        if sentiment:
            ack_msg += f" | Sentiment: {sentiment.sentiment}"
        broker.ack_status = 'success'
        broker.ack_message = ack_msg
        return NodeResult(status=NodeStatus.SUCCESS)
