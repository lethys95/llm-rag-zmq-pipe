import json
import logging
import time
from textwrap import dedent

from pydantic import ValidationError

from src.llm.base import BaseLLM
from src.models.analysis import NeedsAnalysis
from src.models.emotional_state import EmotionalState
from src.models.response_strategy import ResponseStrategy, VALID_APPROACHES, VALID_TONES

logger = logging.getLogger(__name__)


class ResponseStrategyHandler:
    """Selects a therapeutic approach based on the user's active needs and emotional state.

    Output feeds directly into the primary response system prompt via
    ResponseStrategy.system_prompt_addition.

    NOTE: System prompt is a working stub — approach descriptions and
    system_prompt_addition phrasing will need tuning from real outputs.
    """

    SYSTEM_PROMPT = dedent("""\
        You are a therapeutic approach selector for an AI companion. Based on the user's
        psychological needs and emotional state, select the most appropriate response strategy.

        Available approaches:
          reflective_listening     — Mirror and validate feelings. For belonging needs, acute distress,
                                     when the user needs to feel heard above all else.
          socratic_questioning     — Ask questions that guide self-discovery. For autonomy needs,
                                     when the user is capable of reasoning through their situation.
          cognitive_reframing      — Gently challenge distorted thinking. For esteem needs, catastrophising,
                                     or persistent negative interpretations of events.
          behavioral_activation    — Encourage small concrete actions. For low motivation, depression-like
                                     patterns, meaning needs with an action component.
          acceptance_and_validation — Accept the situation without trying to fix it. For grief, loss,
                                      or situations where the user cannot change the outcome.
          meaning_making           — Explore purpose, values, and what matters. For meaning and growth
                                     needs, existential concerns, identity questions.
          practical_problem_solving — Work through a concrete problem together. For safety needs,
                                      actionable issues where practical help is genuinely wanted.

        Available tones:
          empathetic_warm    — Warm and caring. Default for emotional content.
          curious_gentle     — Interested and exploratory. Good for socratic work.
          grounding_steady   — Calm and stabilising. For high arousal or crisis-adjacent states.
          playful_light      — Light and easy. Only appropriate when urgency is low and mood allows.
          direct_honest      — Clear and straightforward. For practical problem-solving.

        system_prompt_addition: 1–3 sentences of behavioral instruction for the companion,
        written as internal guidance — not shown to the user. Should be specific and actionable.
        Avoid clinical language. Write as a brief to a perceptive friend, not a therapist.

        Examples of good system_prompt_additions:
          "Validate their grief before anything else. Don't suggest silver linings or fixes.
           Ask what they miss most."
          "They're caught in a loop of self-blame. Gently introduce doubt about their interpretation
           without dismissing their feelings."
          "Keep it light — they need a break from heavy thinking. Be curious and warm."

        Return ONLY this JSON:
        {
          "approach": "...",
          "tone": "...",
          "needs_focus": [],
          "system_prompt_addition": "...",
          "reasoning": "..."
        }

        reasoning: 1–2 sentences explaining why this approach fits — for internal review only.

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

    def select(
        self,
        needs_analysis: NeedsAnalysis | None,
        emotional_state: EmotionalState | None,
    ) -> ResponseStrategy | None:
        if not needs_analysis and not emotional_state:
            logger.warning("ResponseStrategyHandler: no input available, cannot select")
            return None

        prompt = self._build_prompt(needs_analysis, emotional_state)

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.llm.generate(prompt, json_mode=True)
                result = self._parse(raw)
                if result:
                    logger.debug(
                        "ResponseStrategy: approach=%s tone=%s",
                        result.approach, result.tone,
                    )
                    return result
            except Exception:
                logger.exception("ResponseStrategyHandler attempt %d/%d failed", attempt, self.max_retries)

            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        logger.error("ResponseStrategyHandler failed after %d attempts", self.max_retries)
        return None

    def _build_prompt(
        self,
        needs_analysis: NeedsAnalysis | None,
        emotional_state: EmotionalState | None,
    ) -> str:
        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Needs analysis:\n{self._format_needs(needs_analysis)}\n\n"
            f"Emotional state:\n{self._format_emotional_state(emotional_state)}\n\n"
            f"JSON:"
        )

    def _format_needs(self, needs: NeedsAnalysis | None) -> str:
        if not needs:
            return "Not available."
        active = {
            k: v for k, v in needs.model_dump().items()
            if k in ("physiological", "safety", "belonging", "esteem",
                     "autonomy", "meaning", "growth")
            and isinstance(v, float) and v > 0.1
        }
        scored = sorted(active.items(), key=lambda x: x[1], reverse=True)
        lines = [f"  {k}: {v:.2f}" for k, v in scored]
        return (
            f"Active needs:\n" + ("\n".join(lines) if lines else "  none") + "\n"
            f"Primary: {needs.primary_needs}\n"
            f"Unmet: {needs.unmet_needs}\n"
            f"Urgency: {needs.need_urgency:.2f}  Persistence: {needs.need_persistence:.2f}\n"
            f"Summary: {needs.context_summary}"
        )

    def _format_emotional_state(self, state: EmotionalState | None) -> str:
        if not state:
            return "Not available."
        dominant = sorted(
            [(k, v) for k, v in state.model_dump().items()
             if k not in ("valence", "arousal", "dominance", "confidence", "summary")
             and isinstance(v, float) and v > 0.05],
            key=lambda x: x[1], reverse=True,
        )[:3]
        emotion_str = ", ".join(f"{k}: {v:.2f}" for k, v in dominant) or "neutral"
        return (
            f"Emotions: {emotion_str}\n"
            f"Valence: {state.valence:.2f}  Arousal: {state.arousal:.2f}  "
            f"Dominance: {state.dominance:.2f}\n"
            f"Confidence: {state.confidence:.2f}"
        )

    def _parse(self, raw: str) -> ResponseStrategy | None:
        try:
            text = raw.strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                raise json.JSONDecodeError("No JSON object found", text, 0)
            data = json.loads(text[start:end + 1])
            return ResponseStrategy(**data)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            logger.error("ResponseStrategyHandler parse error: %s", e)
            return None
