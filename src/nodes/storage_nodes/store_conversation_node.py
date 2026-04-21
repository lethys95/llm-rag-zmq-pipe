"""Conversation storage node for persisting messages."""

from dataclasses import dataclass, field
from datetime import datetime

from src.nodes.core.base_node import BaseNode
from src.nodes.orchestration.node_registry_decorator import register_node
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.storage.conversation_store import ConversationStore
from src.rag.base import BaseRAG
from src.rag.embeddings import EmbeddingService
from src.models.sentiment import SentimentAnalysis


@dataclass
class ConversationMetadata:
    """Metadata for storing conversation in RAG."""

    timestamp: str
    speaker: str
    relevance: float = 0.5
    chrono_relevance: float = 0.5
    sentiment: str | None = None
    topics: list[str] = field(default_factory=list)
    context_summary: str | None = None

@register_node
class StoreConversationNode(BaseNode):
    """Node that stores conversation in SQLite and Qdrant."""

    def __init__(
        self,
        conversation_store: ConversationStore,
        rag: BaseRAG,
        embedding_service: EmbeddingService,
    ) -> None:
        super().__init__()
        self.conversation_store = conversation_store
        self.rag_provider = rag
        self.embedding_service = embedding_service

    def get_description(self) -> str:
        return "Persist the conversation turn (user message + response) to SQLite and Qdrant. Run after response is sent."

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        try:
            dialogue_input = broker.dialogue_input
            response = broker.primary_response
            sentiment = broker.sentiment_analysis
            timestamp = datetime.now().isoformat()

            # Store in SQLite
            self.conversation_store.add_message(
                speaker=dialogue_input.speaker,
                message=dialogue_input.content,
                response=response,
                timestamp=timestamp,
            )

            # Store in Qdrant
            conversation_text = f"{dialogue_input.speaker}: {dialogue_input.content}\nAssistant: {response}"
            embedding = self.embedding_service.encode(conversation_text)
            metadata_obj = self._prepare_metadata(
                timestamp, dialogue_input.speaker, sentiment
            )

            self.rag_provider.store(
                text=conversation_text,
                embedding=embedding,
                metadata=metadata_obj.__dict__,
            )

            return NodeResult(status=NodeStatus.SUCCESS, metadata={"stored": True})
        except Exception as e:
            return NodeResult(status=NodeStatus.FAILED, error=str(e))

    def _prepare_metadata(
        self, timestamp: str, speaker: str, sentiment: SentimentAnalysis | None = None
    ) -> ConversationMetadata:
        if sentiment:
            return ConversationMetadata(
                timestamp=timestamp,
                speaker=speaker,
                relevance=(
                    sentiment.relevance if sentiment.relevance is not None else 0.5
                ),
                chrono_relevance=(
                    sentiment.chrono_relevance
                    if sentiment.chrono_relevance is not None
                    else 0.5
                ),
                sentiment=sentiment.sentiment,
                topics=sentiment.key_topics or [],
                context_summary=sentiment.context_summary,
            )
        else:
            return ConversationMetadata(timestamp=timestamp, speaker=speaker)
