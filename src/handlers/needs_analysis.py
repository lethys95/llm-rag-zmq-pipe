import json
import logging
import time
from textwrap import dedent

from pydantic import ValidationError

from src.llm.base import BaseLLM
from src.models.analysis import NeedsAnalysis
from src.models.emotional_state import EmotionalState
from src.rag.selector import RAGDocument

logger = logging.getLogger(__name__)


class NeedsAnalysisHandler:
    """Identifies the user's active psychological needs from message context and memory.

    Applies Maslow's hierarchy as dimensional scoring — each need category
    gets a 0.0–1.0 score reflecting how activated it is right now (phasic).

    Inputs considered:
      - Current message
      - EmotionalState (VAD + categorical emotions)
      - Retrieved memory facts (past context about this user)

    NOTE: The system prompt is a working stub. It will need tuning once
    real outputs can be reviewed against actual conversations.
    """

    SYSTEM_PROMPT = dedent("""\
        You are a psychological needs analyst. Identify which of the user's psychological
        needs are currently active based on their message, emotional state, and past context.

        Use Maslow's hierarchy as dimensional scoring — each need gets a score from 0.0 to 1.0
        based on how strongly it appears to be activated right now. Most scores will be 0.0.

        Need dimensions:
          physiological — hunger, sleep deprivation, pain, physical distress
          safety        — financial stress, housing insecurity, health concerns, feeling threatened
          belonging     — loneliness, desire for connection, relationship difficulties, feeling unseen
          esteem        — feeling unvalued, lacking recognition, competence doubts, low self-worth
          autonomy      — feeling controlled, trapped, lacking agency or self-direction
          meaning       — lack of purpose, existential emptiness, directionlessness, grief of identity
          growth        — desire to learn, improve, create, or become more than current self

        need_urgency: how pressing is the dominant need right now?
          0.9+ = crisis or acute distress requiring immediate attention
          0.6  = clearly present and affecting the user significantly
          0.3  = background concern, not immediately distressing
          0.1  = mild or hypothetical

        need_persistence: how chronic does this pattern appear to be?
          0.9  = longstanding pattern evident across multiple memories
          0.5  = recurrent but not defining
          0.1  = appears situational or one-off
          (Note: this is assessed from available context, not computed — treat as approximate)

        primary_needs: list the 1–3 most activated need names (highest scores)
        unmet_needs: subset of primary_needs that appear unmet rather than just active

        Return ONLY this JSON structure:
        {
          "physiological": 0.0,
          "safety": 0.0,
          "belonging": 0.0,
          "esteem": 0.0,
          "autonomy": 0.0,
          "meaning": 0.0,
          "growth": 0.0,
          "primary_needs": [],
          "unmet_needs": [],
          "need_urgency": 0.0,
          "need_persistence": 0.0,
          "context_summary": "..."
        }

        context_summary: 1–2 sentences describing what the user appears to need and why,
        written as if briefing the companion — not clinical language.

        Example:
        Input: User says "I don't know what to do with myself lately."
               Emotional state: sadness 0.5, loneliness 0.6, confusion 0.4, valence -0.5
               Memories: user lost their mother 3 months ago, has mentioned feeling purposeless twice

        Output:
        {
          "physiological": 0.0, "safety": 0.0, "belonging": 0.7, "esteem": 0.2,
          "autonomy": 0.1, "meaning": 0.8, "growth": 0.2,
          "primary_needs": ["meaning", "belonging"],
          "unmet_needs": ["meaning", "belonging"],
          "need_urgency": 0.5,
          "need_persistence": 0.8,
          "context_summary": "They are struggling with purpose and connection following their mother's death. The feeling of directionlessness has come up repeatedly and appears chronic rather than situational."
        }

        Return ONLY valid JSON. No explanation, no extra text.""")

    def __init__(
        self,
        llm_provider: BaseLLM,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ) -> None:
        self.llm = llm_provider
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def analyze(
        self,
        message: str,
        speaker: str,
        emotional_state: EmotionalState | None = None,
        retrieved_documents: list[RAGDocument] | None = None,
    ) -> NeedsAnalysis | None:
        prompt = self._build_prompt(message, emotional_state, retrieved_documents or [])

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.llm.generate(prompt, json_mode=True)
                result = self._parse(raw, speaker)
                if result:
                    logger.debug(
                        "NeedsAnalysis: primary=%s urgency=%.2f persistence=%.2f",
                        result.primary_needs, result.need_urgency, result.need_persistence,
                    )
                    return result
            except Exception:
                logger.exception("NeedsAnalysisHandler attempt %d/%d failed", attempt, self.max_retries)

            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        logger.error("NeedsAnalysisHandler failed after %d attempts", self.max_retries)
        return None

    def _build_prompt(
        self,
        message: str,
        emotional_state: EmotionalState | None,
        retrieved_documents: list[RAGDocument],
    ) -> str:
        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Current message:\n{message}\n\n"
            f"Emotional state:\n{self._format_emotional_state(emotional_state)}\n\n"
            f"Relevant memories:\n{self._format_memories(retrieved_documents)}\n\n"
            f"JSON:"
        )

    def _format_emotional_state(self, state: EmotionalState | None) -> str:
        if not state:
            return "Not available."
        emotion_fields = {
            k: v for k, v in state.model_dump().items()
            if k not in ("valence", "arousal", "dominance", "confidence", "summary")
            and isinstance(v, float) and v > 0.05
        }
        dominant = sorted(emotion_fields.items(), key=lambda x: x[1], reverse=True)[:4]
        emotion_str = ", ".join(f"{k}: {v:.2f}" for k, v in dominant) if dominant else "neutral"
        return (
            f"Dominant emotions: {emotion_str}\n"
            f"Valence: {state.valence:.2f}  Arousal: {state.arousal:.2f}  "
            f"Dominance: {state.dominance:.2f}\n"
            f"Summary: {state.summary or 'N/A'}"
        )

    def _format_memories(self, docs: list[RAGDocument]) -> str:
        if not docs:
            return "No relevant memories available."
        lines = []
        for doc in docs:
            subject = doc.metadata.get("subject", "")
            valence = doc.metadata.get("valence")
            parts = []
            if subject:
                parts.append(subject)
            if valence is not None:
                parts.append(f"valence={float(valence):.2f}")
            tag = f" [{', '.join(parts)}]" if parts else ""
            lines.append(f"- {doc.content}{tag}")
        return "\n".join(lines)

    def _parse(self, raw: str, speaker: str) -> NeedsAnalysis | None:
        try:
            text = raw.strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                raise json.JSONDecodeError("No JSON object found", text, 0)
            data = json.loads(text[start:end + 1])
            data["memory_owner"] = speaker
            return NeedsAnalysis(**data)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            logger.error("NeedsAnalysisHandler parse error: %s", e)
            return None
