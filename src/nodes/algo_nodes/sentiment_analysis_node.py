from src.handlers.sentiment_analysis import SentimentAnalysisHandler
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import register_node


@register_node
class SentimentAnalysisNode(BaseNode):

    def __init__(self, sentiment_analysis_handler: SentimentAnalysisHandler) -> None:
        super().__init__()
        self.handler = sentiment_analysis_handler

    def get_description(self) -> str:
        return (
            "Analyse the emotional tone and relevance of the user's message. "
            "Produces a SentimentAnalysis stored in the broker for downstream nodes."
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        if not broker.dialogue_input:
            return NodeResult(status=NodeStatus.FAILED, error="No dialogue_input in broker")

        result = self.handler.analyze(
            message=broker.dialogue_input.content,
            speaker=broker.dialogue_input.speaker,
        )

        if result is None:
            return NodeResult(status=NodeStatus.FAILED, error="Sentiment analysis returned None")

        broker.sentiment_analysis = result
        return NodeResult(status=NodeStatus.SUCCESS)
