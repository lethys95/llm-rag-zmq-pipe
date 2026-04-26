import json
import logging
import time
from textwrap import dedent

from pydantic import ValidationError

from src.config.settings import WorkerCallConfig
from src.handlers.handler_registry_decorator import register_handler
from src.llm.base import BaseLLM
from src.models.emotional_state import EmotionalState

logger = logging.getLogger(__name__)


@register_handler
class EmotionalStateHandler:
    """Analyses the emotional state of a message using fixed emotion categories and VAD.

    Session-scoped output — not stored to RAG. Drives response strategy
    and crisis detection for the current turn.

    Uses json_mode on the LLM call as the primary JSON enforcement mechanism,
    with retry logic as fallback.
    """

    SYSTEM_PROMPT = dedent("""\
        You are an emotional state analyser. Analyse the user's message and return a JSON object.

        Return ONLY this JSON structure with exactly these fields:
        {
          "joy": 0.0,
          "sadness": 0.0,
          "grief": 0.0,
          "anger": 0.0,
          "frustration": 0.0,
          "fear": 0.0,
          "anxiety": 0.0,
          "disgust": 0.0,
          "guilt": 0.0,
          "shame": 0.0,
          "loneliness": 0.0,
          "overwhelm": 0.0,
          "contentment": 0.0,
          "confusion": 0.0,
          "valence": 0.0,
          "arousal": 0.0,
          "dominance": 0.0,
          "confidence": 0.0,
          "summary": "..."
        }

        Emotion scores (all 0.0–1.0, default 0.0, only raise what is present):
          joy          — happiness, delight, pleasure
          sadness      — general unhappiness, sorrow, low mood
          grief        — loss-specific sorrow (death, breakup, significant loss)
          anger        — irritation to rage, feeling wronged or violated
          frustration  — blocked goal, things not working as expected
          fear         — acute threat, dread, feeling unsafe
          anxiety      — diffuse worry, nervousness, anticipatory distress without specific cause
          disgust      — revulsion, strong disapproval
          guilt        — "I did something wrong" — behaviour-focused self-reproach
          shame        — "I am wrong/bad/broken" — identity-level self-reproach
          loneliness   — isolation, disconnection, feeling unseen or uncared for
          overwhelm    — too much to process or handle
          contentment  — calm satisfaction, quiet positive state
          confusion    — uncertainty, disorientation, not knowing what to think or feel

        VAD dimensions:
          valence   — -1.0 (very negative) to 1.0 (very positive)
          arousal   — 0.0 (calm, subdued) to 1.0 (intense, activated)
          dominance — 0.0 (powerless, controlled) to 1.0 (in control, autonomous)

        confidence — how reliably can you score this from text alone?
          0.9+ — explicit emotional language, clear unambiguous context
          0.5  — some emotional signal but ambiguous tone or short message
          0.2  — very short, possible sarcasm, minimal signal

        summary — one plain-language sentence describing the dominant emotional state.

        Example input:  "I can't stop thinking about my mum. It's been three months."
        Example output:
        {
          "joy": 0.0, "sadness": 0.6, "grief": 0.85, "anger": 0.0,
          "frustration": 0.0, "fear": 0.0, "anxiety": 0.2, "disgust": 0.0,
          "guilt": 0.1, "shame": 0.0, "loneliness": 0.5, "overwhelm": 0.0,
          "contentment": 0.0, "confusion": 0.0,
          "valence": -0.7, "arousal": 0.3, "dominance": 0.2,
          "confidence": 0.85,
          "summary": "Ongoing grief after losing their mother, with loneliness and mild guilt."
        }

        Return ONLY valid JSON. No explanation, no extra text.""")

    def __init__(self, worker_llm: BaseLLM, worker_call: WorkerCallConfig) -> None:
        self.llm = worker_llm
        self.max_retries = worker_call.max_retries
        self.retry_delay = worker_call.retry_delay

    def analyze(self, message: str) -> EmotionalState | None:
        prompt = f"{self.SYSTEM_PROMPT}\n\nUser message:\n{message}\n\nJSON:"

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.llm.generate(prompt, json_mode=True)
                result = self._parse(raw)
                if result:
                    logger.debug(
                        "EmotionalState: valence=%.2f arousal=%.2f dominance=%.2f confidence=%.2f",
                        result.valence, result.arousal, result.dominance, result.confidence,
                    )
                    return result
            except Exception:
                logger.exception("EmotionalStateHandler attempt %d/%d failed", attempt, self.max_retries)

            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        logger.error("EmotionalStateHandler failed after %d attempts", self.max_retries)
        return None

    def _parse(self, raw: str) -> EmotionalState | None:
        try:
            text = raw.strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                raise json.JSONDecodeError("No JSON object found", text, 0)
            data = json.loads(text[start:end + 1])
            return EmotionalState(**data)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            logger.error("EmotionalStateHandler parse error: %s", e)
            return None
