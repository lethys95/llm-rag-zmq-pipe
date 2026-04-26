"""Primary response handler using composition."""

import logging
from textwrap import dedent

from src.handlers.handler_registry_decorator import register_handler
from src.llm.base import BaseLLM
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker

logger = logging.getLogger(__name__)


@register_handler
class PrimaryResponseHandler:
    """Handler for generating primary responses using a large LLM.

    Composes a BaseLLM provider and builds the full prompt from context
    assembled by the node pipeline via KnowledgeBroker.
    """

    SYSTEM_PROMPT_WITH_CONTEXT = "You are an AI companion. You're here to provide emotional support, to listen, provide guidance, etc."

    SYSTEM_PROMPT_WITHOUT_CONTEXT = "You are an AI companion. You're here to provide emotional support, to listen, provide guidance, etc."

    def __init__(self, primary_llm: BaseLLM) -> None:
        self.llm = primary_llm

    def generate_response(
        self,
        prompt: str,
        broker: KnowledgeBroker,
        system_prompt_override: str | None = None,
    ) -> str:
        """Generate a response to user prompt.

        Args:
            prompt: The user's prompt/question
            broker: KnowledgeBroker providing all context assembled by the pipeline
            system_prompt_override: Optional override for system prompt persona

        Returns:
            Generated response string
        """
        logger.debug("Generating primary response for prompt: %s...", prompt[:100])

        try:
            analyzed_context = broker.get_analyzed_context()
            context = self._format_analyzed_context(analyzed_context) if analyzed_context else None

            full_prompt = self._build_prompt(prompt, context, system_prompt_override, analyzed_context)
            response = self.llm.generate(full_prompt)

            logger.info("Primary response generated (length: %s)", len(response))
            return response

        except Exception as e:
            logger.error("Error generating primary response: %s", e, exc_info=True)
            raise

    def _format_analyzed_context(self, analyzed_context: dict) -> str:
        """Format analyzed context from various nodes into coherent string."""
        parts = []

        # Add emotional state
        if "emotional_state" in analyzed_context:
            state = analyzed_context["emotional_state"]
            emotion_parts = []
            dominant = sorted(
                [(k, v) for k, v in state.model_dump().items()
                 if k not in ("valence", "arousal", "dominance", "confidence", "summary")
                 and isinstance(v, float) and v > 0.1],
                key=lambda x: x[1], reverse=True,
            )[:3]
            if dominant:
                emotion_parts.append("Emotions: " + ", ".join(f"{k}: {v:.2f}" for k, v in dominant))
            if state.summary:
                emotion_parts.append(f"State: {state.summary}")
            if emotion_parts:
                parts.append("\n".join(emotion_parts))

        # Add needs analysis only when no needs advisor has translated it to natural language
        if "needs_analysis" in analyzed_context:
            advisor_names = {o.advisor for o in (analyzed_context.get("advisor_outputs") or [])}
            if "needs" not in advisor_names:
                needs = analyzed_context["needs_analysis"]
                need_parts = []
                if needs.primary_needs:
                    need_parts.append(f"Primary needs: {', '.join(needs.primary_needs)}")
                if needs.unmet_needs:
                    need_parts.append(f"Unmet: {', '.join(needs.unmet_needs)}")
                if needs.need_urgency > 0.6:
                    need_parts.append(f"Urgency: {needs.need_urgency:.2f}")
                if needs.context_summary:
                    need_parts.append(needs.context_summary)
                if need_parts:
                    parts.append("\n".join(need_parts))

        # Add retrieved user facts
        if "user_facts" in analyzed_context:
            facts = analyzed_context["user_facts"]
            if facts:
                fact_lines = [f"- {f.claim}" for f in facts[:8]]
                parts.append("What I know about you:\n" + "\n".join(fact_lines))

        # Add idle time
        if "idle_time_minutes" in analyzed_context:
            idle = analyzed_context["idle_time_minutes"]
            if idle > 0:
                parts.append(f"User idle time: {idle:.1f} minutes")

        if parts:
            return "\n\n---\n\n".join(parts)

        return ""

    def _format_advisor_outputs(self, outputs: list) -> str:
        if not outputs:
            return ""
        relevant = [o for o in outputs if o.potency >= 0.3]
        if not relevant:
            return ""
        relevant.sort(key=lambda o: o.potency, reverse=True)
        lines = ["\nAdvisory context:"]
        for o in relevant:
            lines.append(f"[{o.advisor}] {o.advice}")
        return "\n".join(lines) + "\n"

    def _build_prompt(
        self,
        prompt: str,
        context: str | None,
        system_prompt_override: str | None = None,
        analyzed_context: dict | None = None,
    ) -> str:
        """Build full prompt with optional context and system prompt override."""
        default_with_context = self.SYSTEM_PROMPT_WITH_CONTEXT
        default_without_context = self.SYSTEM_PROMPT_WITHOUT_CONTEXT

        # Append strategy_addition only when the strategy advisor hasn't handled it
        strategy_addition = ""
        if analyzed_context and "response_strategy" in analyzed_context:
            advisor_names = {o.advisor for o in (analyzed_context.get("advisor_outputs") or [])}
            if "strategy" not in advisor_names:
                strategy_addition = "\n\n" + analyzed_context["response_strategy"].system_prompt_addition

        advisor_guidance = ""
        if analyzed_context and "advisor_outputs" in analyzed_context:
            advisor_guidance = self._format_advisor_outputs(analyzed_context["advisor_outputs"])

        if context and context.strip():
            system_prompt = (system_prompt_override or default_with_context) + strategy_addition
            logger.debug(
                "Built augmented prompt with context%s",
                " and custom system prompt" if system_prompt_override else "",
            )
            return dedent(f"""
                {system_prompt}

                Context:
                {context}
                {advisor_guidance}
                User Question:
                {prompt}

                Assistant Response:""")

        system_prompt = (system_prompt_override or default_without_context) + strategy_addition
        if advisor_guidance:
            return dedent(f"""
                {system_prompt}
                {advisor_guidance}
                User Question:
                {prompt}

                Assistant Response:""")

        logger.debug(
            "Built prompt without context%s",
            " and custom system prompt" if system_prompt_override else "",
        )
        return dedent(f"""
            {system_prompt}

            User Question:
            {prompt}

            Assistant Response:""")
