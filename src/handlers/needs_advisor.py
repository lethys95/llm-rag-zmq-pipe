import logging

from src.handlers.handler_registry_decorator import register_handler
from src.models.advisor import AdvisorOutput
from src.models.analysis import NeedsAnalysis

logger = logging.getLogger(__name__)


@register_handler
class NeedsAdvisorHandler:
    """Translates NeedsAnalysis into natural language guidance for the primary LLM.

    No LLM call required — NeedsAnalysis.context_summary is already natural language,
    written as a brief to a friend rather than clinical notation.

    Potency is derived from need_urgency: how pressing the need is right now is also
    how much this advisory context should influence the response. A casual message with
    no activated needs → potency near 0. Acute distress → potency near 1.

    When no needs analysis is available (analysis didn't run, or wasn't needed),
    potency is 0.0 and the advice is minimal.
    """

    def advise(self, needs_analysis: NeedsAnalysis | None) -> AdvisorOutput:
        if not needs_analysis:
            return AdvisorOutput(
                advisor="needs",
                advice="No needs analysis available for this interaction.",
                potency=0.0,
            )

        logger.debug(
            "NeedsAdvisor: urgency=%.2f primary=%s",
            needs_analysis.need_urgency, needs_analysis.primary_needs,
        )

        return AdvisorOutput(
            advisor="needs",
            advice=needs_analysis.context_summary,
            potency=needs_analysis.need_urgency,
        )
