"""Conversation persistence — stores each turn to SQLite and Qdrant."""

import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.storage.conversation_store import ConversationStore
from src.rag.base import BaseRAG
from src.rag.embeddings import EmbeddingService
from src.models.emotional_state import EmotionalState


@dataclass
class ConversationMetadata:
    timestamp: str
    memory_owner: str
    valence: float | None = None
    arousal: float | None = None
    dominance: float | None = None
    emotional_summary: str | None = None


class ConversationStorage(BaseNode):
    """Persists a conversation turn to SQLite (recent history) and Qdrant (long-term memory).

    Not coordinator-selectable. Run by the orchestrator as a background task
    after the response has been forwarded to TTS.
    """

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
        return "Persist conversation turn to SQLite and Qdrant. Runs as background task after response is sent."

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        try:
            dialogue_input = broker.dialogue_input
            response = broker.primary_response
            emotional_state = None  # broker field suspended — see knowledge_broker.py
            timestamp = datetime.now().isoformat()

            self.conversation_store.add_message(
                speaker=dialogue_input.speaker,
                message=dialogue_input.content,
                response=response,
                timestamp=timestamp,
            )

            conversation_text = f"{dialogue_input.speaker}: {dialogue_input.content}\nAssistant: {response}"
            embedding = self.embedding_service.encode(conversation_text)
            metadata_obj = self._prepare_metadata(timestamp, dialogue_input.speaker, emotional_state)

            self.rag_provider.store(
                text=conversation_text,
                embedding=embedding,
                metadata=metadata_obj.__dict__,
            )

            for fact in broker.user_facts:
                try:
                    fact_embedding = self.embedding_service.encode(fact.claim)
                    self.rag_provider.store(
                        text=fact.claim,
                        embedding=fact_embedding,
                        metadata={
                            "timestamp": timestamp,
                            "memory_owner": fact.memory_owner,
                            "claim": fact.claim,
                            "valence": fact.valence,
                            "arousal": fact.arousal,
                            "dominance": fact.dominance,
                            "chrono_relevance": fact.chrono_relevance,
                            "subject": fact.subject,
                        },
                    )
                except Exception:
                    logger.exception("Failed to store fact: %s", fact.claim)

            return NodeResult(status=NodeStatus.SUCCESS, metadata={"stored": True})
        except Exception as e:
            return NodeResult(status=NodeStatus.FAILED, error=str(e))

    def _prepare_metadata(
        self, timestamp: str, speaker: str, emotional_state: EmotionalState | None = None
    ) -> ConversationMetadata:
        if emotional_state:
            return ConversationMetadata(
                timestamp=timestamp,
                memory_owner=speaker,
                valence=emotional_state.valence,
                arousal=emotional_state.arousal,
                dominance=emotional_state.dominance,
                emotional_summary=emotional_state.summary,
            )
        return ConversationMetadata(timestamp=timestamp, memory_owner=speaker)
