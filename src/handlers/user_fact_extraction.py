import json
import logging
import time
from textwrap import dedent

from pydantic import ValidationError

from src.config.settings import WorkerCallConfig
from src.handlers.handler_registry_decorator import register_handler
from src.llm.base import BaseLLM
from src.models.emotional_state import EmotionalState
from src.models.user_fact import UserFact

logger = logging.getLogger(__name__)


@register_handler
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

        Response format:
        {
          "facts": [
            {
              "claim": "user likes pepperoni pizza",
              "chrono_relevance": 0.85,
              "subject": "food preferences"
            }
          ]
        }

        Return {"facts": []} if no facts can be extracted.
        Respond ONLY with valid JSON. No explanation, no extra text.""")

    def __init__(self, worker_llm: BaseLLM, worker_call: WorkerCallConfig) -> None:
        self.llm = worker_llm
        self.max_retries = worker_call.max_retries
        self.retry_delay = worker_call.retry_delay

    def extract(
        self,
        message: str,
        speaker: str,
        emotional_state: EmotionalState | None = None,
    ) -> list[UserFact]:
        prompt = f"{self.SYSTEM_PROMPT}\n\nUser message:\n{message}\n\nJSON:"

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.llm.generate(prompt, json_mode=True)
                facts = self._parse(raw, speaker, emotional_state)
                if facts is not None:
                    logger.debug("Extracted %d facts from message", len(facts))
                    return facts
            except Exception:
                logger.exception("UserFactExtractionHandler attempt %d/%d failed", attempt, self.max_retries)

            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        logger.error("UserFactExtractionHandler failed after %d attempts", self.max_retries)
        return []

    def _parse(
        self,
        raw: str,
        speaker: str,
        emotional_state: EmotionalState | None,
    ) -> list[UserFact] | None:
        try:
            text = raw.strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                raise json.JSONDecodeError("No JSON object found", text, 0)
            data = json.loads(text[start:end + 1])
            items = data.get("facts", [])
            facts = []
            for item in items:
                item["memory_owner"] = speaker
                if emotional_state is not None:
                    item["valence"] = emotional_state.valence
                    item["arousal"] = emotional_state.arousal
                    item["dominance"] = emotional_state.dominance
                facts.append(UserFact(**item))
            return facts
        except (json.JSONDecodeError, ValidationError, TypeError, KeyError) as e:
            logger.error("UserFactExtractionHandler parse error: %s", e)
            return None

