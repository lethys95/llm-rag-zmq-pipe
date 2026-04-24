import logging

from src.models.advisor import AdvisorOutput
from src.models.response_strategy import ResponseStrategy

logger = logging.getLogger(__name__)

_POTENCY_FLOOR = 0.4


class StrategyAdvisorHandler:
    """Translates ResponseStrategy into natural language guidance for the primary LLM.

    No LLM call required — ResponseStrategy.system_prompt_addition is already
    behavioral instruction in natural language, written as a brief to the companion.

    This replaces the pattern of appending system_prompt_addition directly to the
    system prompt. Strategy guidance belongs in the advisor layer alongside memory
    and needs context — not embedded in the persona prompt.

    Potency has a floor of 0.4: strategy guidance is always at least somewhat relevant
    because even a casual response has a chosen approach. When needs urgency is high,
    it's more important that the companion follow the strategy carefully.
    """

    def advise(
        self,
        response_strategy: ResponseStrategy | None,
        need_urgency: float = 0.0,
    ) -> AdvisorOutput:
        if not response_strategy:
            return AdvisorOutput(
                advisor="strategy",
                advice="No response strategy selected.",
                potency=0.0,
            )

        potency = max(_POTENCY_FLOOR, need_urgency)
        logger.debug(
            "StrategyAdvisor: approach=%s tone=%s potency=%.2f",
            response_strategy.approach, response_strategy.tone, potency,
        )

        return AdvisorOutput(
            advisor="strategy",
            advice=response_strategy.system_prompt_addition,
            potency=potency,
        )
