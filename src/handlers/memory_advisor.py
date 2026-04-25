import json
import logging
import time
from textwrap import dedent

from pydantic import ValidationError, BaseModel, Field

from src.llm.base import BaseLLM
from src.models.advisor import AdvisorOutput
from src.models.analysis import MemoryEvaluation
from src.rag.selector import RAGDocument

logger = logging.getLogger(__name__)


class _MemoryAdvisorResponse(BaseModel):
    advice: str = Field(..., description="Natural language synthesis")
    potency: float = Field(..., ge=0.0, le=1.0)


class MemoryAdvisorHandler:
    """Synthesises retrieved memories into natural language guidance for the primary LLM.

    This is the most important advisor. The companion's character with a specific user
    is almost entirely built from memories. This handler makes those memories usable
    by the primary LLM rather than dumping raw documents at it.

    Input: evaluated memories (RAGDocument + MemoryEvaluation reasoning) + current message.
    Output: AdvisorOutput with natural language synthesis + potency.

    Potency reflects how much the retrieved memories matter to this specific moment.
    No memories → potency 0.0. Many directly relevant memories → potency near 1.0.

    NOTE: System prompt is a working stub — will need tuning from real outputs.
    """

    SYSTEM_PROMPT = dedent("""\
        You are an advisor to an AI companion. You synthesise retrieved memories about a user
        to help the companion respond appropriately.

        CRITICAL: You are NOT a participant in this conversation. You will NEVER speak directly
        to the user. You will NEVER write responses like "I'm so sorry" or "I'm here for you."
        If you return anything other than actual advisory guidance, you have failed your job.

        Your primary responsibility is to provide META-LANGUAGE: instructions and context for
        the companion, not a direct response to the user.

        You will be given:
        - The current message from the user
        - A set of past memories about this user, each with a relevance note

        Your job is to analyse what the memories reveal about this person and tell the companion
        what they should know and how they should handle this moment. Draw on patterns and context
        from the memories. Use the database content to inform your guidance.

        Example of WRONG output (direct response — never do this):
          "I'm so sorry to hear about your cat. I know how painful this must be."

        Example of CORRECT output (meta-language advisory):
          "Fetched data indicates the user formed a deep bond with their pet over several years.
           This is a critical grief moment. The companion should prioritise acknowledgement and
           emotional validation over problem-solving. Avoid minimising the loss."

        Then score potency: how much do these memories actually matter to this specific moment?
          0.9+ — multiple directly relevant memories, emotionally connected to current message
          0.6  — some useful context but not directly on-topic
          0.3  — sparse or old memories with weak connection to current message
          0.0  — no memories, or none meaningfully connected to the current moment

        Return ONLY this JSON:
        {
          "advice": "...",
          "potency": 0.0
        }

        advice: 2–4 sentences of meta-language guidance for the companion. No clinical scores or
        numbers. Do NOT address the user. Do NOT draft a response.

        If no memories are relevant, advice should be brief: "No relevant history to draw on."
        and potency should be 0.0 or close to it.

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

    def advise(
        self,
        message: str,
        evaluated_memories: list[tuple[RAGDocument, MemoryEvaluation]],
    ) -> AdvisorOutput:
        if not evaluated_memories:
            return AdvisorOutput(
                advisor="memory",
                advice="No relevant memories available for this conversation yet.",
                potency=0.0,
            )

        prompt = self._build_prompt(message, evaluated_memories)

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.llm.generate(prompt, json_mode=True)
                result = self._parse(raw)
                if result:
                    logger.debug(
                        "MemoryAdvisor: potency=%.2f, advice=%.60s...",
                        result.potency, result.advice,
                    )
                    return AdvisorOutput(
                        advisor="memory",
                        advice=result.advice,
                        potency=result.potency,
                    )
            except Exception:
                logger.exception("MemoryAdvisorHandler attempt %d/%d failed", attempt, self.max_retries)

            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        logger.error("MemoryAdvisorHandler failed after %d attempts", self.max_retries)
        return AdvisorOutput(
            advisor="memory",
            advice="Memory synthesis unavailable.",
            potency=0.0,
        )

    def _build_prompt(
        self,
        message: str,
        evaluated_memories: list[tuple[RAGDocument, MemoryEvaluation]],
    ) -> str:
        memories_text = self._format_memories(evaluated_memories)
        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Current message:\n{message}\n\n"
            f"Past memories:\n{memories_text}\n\n"
            f"JSON:"
        )

    def _format_memories(
        self, evaluated: list[tuple[RAGDocument, MemoryEvaluation]]
    ) -> str:
        lines = []
        for doc, evaluation in evaluated:
            lines.append(
                f"- {doc.content}\n"
                f"  (relevance: {evaluation.relevance:.2f} — {evaluation.reasoning})"
            )
        return "\n".join(lines)

    def _parse(self, raw: str) -> _MemoryAdvisorResponse | None:
        try:
            text = raw.strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                raise json.JSONDecodeError("No JSON object found", text, 0)
            data = json.loads(text[start:end + 1])
            return _MemoryAdvisorResponse(**data)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            logger.error("MemoryAdvisorHandler parse error: %s", e)
            return None
