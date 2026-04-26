import logging

from src.handlers.handler_registry_decorator import register_handler
from src.models.advisor import AdvisorOutput

logger = logging.getLogger(__name__)

_SPOKEN_ADVICE = (
    "Respond in natural spoken language. "
    "Carefully consider the length of your response — less is often more in spoken conversation. "
    "Leave space for the user to speak. "
    "Do not use lists, bullet points, emojis, or any formatting that does not translate to speech."
)

_TEXT_ADVICE = (
    "Respond in clear written language. "
    "Use formatting where it genuinely aids clarity, but do not overstructure. "
    "Carefully consider the length of your response — less is often more."
)

_POTENCY = 0.65


@register_handler
class FormatAdvisorHandler:
    """Produces format and length guidance for the primary LLM based on delivery medium.

    Rule-based — no LLM call. Reads the dialogue mode ('spoken' or 'text') and
    produces advisory guidance on register, length, and formatting constraints.

    Potency is fixed at a moderate level: format guidance is always relevant but
    should not override emotional or strategic context.
    """

    def advise(self, mode: str) -> AdvisorOutput:
        if mode == "spoken":
            advice = _SPOKEN_ADVICE
        else:
            advice = _TEXT_ADVICE

        logger.debug("FormatAdvisor: mode=%s potency=%.2f", mode, _POTENCY)
        return AdvisorOutput(
            advisor="format",
            advice=advice,
            potency=_POTENCY,
        )
