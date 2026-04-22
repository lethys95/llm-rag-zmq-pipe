import json
import logging
import time
from datetime import datetime
from textwrap import dedent

from pydantic import ValidationError

from src.llm.base import BaseLLM
from src.rag.base import BaseRAG
from src.rag.embeddings import EmbeddingService
from src.models.user_fact import UserFact

logger = logging.getLogger(__name__)


class UserFactExtractionHandler:
    """Extracts atomic facts about the user from a message.

    Each extracted fact is stored as a separate vector point in Qdrant,
    allowing precise semantic retrieval of specific user knowledge.

    Only directly stated or strongly implied facts are extracted.
    Loose inferences are explicitly excluded to prevent the companion
    from accumulating hallucinated beliefs about the user.
    """

    SYSTEM_PROMPT = dedent("""\
        You are an information extraction assistant. Extract atomic facts about the user from their message.

        Rules:
        - Only extract facts that are directly stated or strongly implied
        - No loose inference — if the user says "making pizza tonight" do NOT infer "likes cooking on weekdays"
        - Each fact must be about the user specifically, phrased as "user [verb] [subject]"
        - Assign chrono_relevance based on how stable/long-lasting the fact is:
            0.9+ = stable preference or significant life event ("user lost their mother")
            0.5  = moderately stable ("user is stressed about work this week")
            0.1  = ephemeral ("user is making pizza tonight")

        Response format — a JSON array:
        [
          {
            "claim": "user likes pepperoni pizza",
            "sentiment": "positive",
            "confidence": 0.9,
            "chrono_relevance": 0.85,
            "subject": "food preferences"
          }
        ]

        sentiment must be exactly "positive", "negative", or "neutral".
        Return [] if no facts can be extracted.
        Respond ONLY with a valid JSON array. No explanation, no extra text.""")

    def __init__(
        self,
        llm_provider: BaseLLM,
        rag_provider: BaseRAG,
        embedding_service: EmbeddingService | None = None,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ) -> None:
        self.llm = llm_provider
        self.rag = rag_provider
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def extract(self, message: str, speaker: str) -> list[UserFact]:
        prompt = f"{self.SYSTEM_PROMPT}\n\nUser message:\n{message}\n\nJSON array:"

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.llm.generate(prompt, json_mode=True)
                facts = self._parse(raw, speaker)
                if facts is not None:
                    logger.debug("Extracted %d facts from message", len(facts))
                    if facts:
                        self._store(facts)
                    return facts
            except Exception:
                logger.exception("UserFactExtractionHandler attempt %d/%d failed", attempt, self.max_retries)

            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        logger.error("UserFactExtractionHandler failed after %d attempts", self.max_retries)
        return []

    def _parse(self, raw: str, speaker: str) -> list[UserFact] | None:
        try:
            text = raw.strip()
            start, end = text.find("["), text.rfind("]")
            if start == -1 or end == -1:
                raise json.JSONDecodeError("No JSON array found", text, 0)
            items = json.loads(text[start:end + 1])
            facts = []
            for item in items:
                item["memory_owner"] = speaker
                facts.append(UserFact(**item))
            return facts
        except (json.JSONDecodeError, ValidationError, TypeError, KeyError) as e:
            logger.error("UserFactExtractionHandler parse error: %s", e)
            return None

    def _store(self, facts: list[UserFact]) -> None:
        timestamp = datetime.now().isoformat()
        for fact in facts:
            try:
                embedding = self.embedding_service.encode(fact.claim)
                self.rag.store(
                    text=fact.claim,
                    embedding=embedding,
                    metadata={
                        "timestamp": timestamp,
                        "memory_owner": fact.memory_owner,
                        "claim": fact.claim,
                        "sentiment": fact.sentiment,
                        "confidence": fact.confidence,
                        "chrono_relevance": fact.chrono_relevance,
                        "subject": fact.subject,
                    },
                )
            except Exception:
                logger.exception("Failed to store fact: %s", fact.claim)
