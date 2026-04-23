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
        You are synthesising past memories about a user to help a companion respond well right now.

        You will be given:
        - The current message from the user
        - A set of past memories, each with a relevance reasoning note

        Your job is to write a brief natural language synthesis — not a list, not clinical notation.
        Write as if briefing a perceptive friend who is about to respond. Tell them what matters
        about this person right now, given what you know. Emphasise patterns and context over
        individual facts. Be honest about what is and isn't relevant.

        Then score potency: how much do these memories actually matter to this specific moment?
          0.9+ — multiple directly relevant memories, recent, emotionally connected to current message
          0.6  — some useful context but not directly on-topic
          0.3  — sparse or old memories with weak connection to current message
          0.0  — no memories, or none meaningfully connected to the current moment

        Return ONLY this JSON:
        {
          "advice": "...",
          "potency": 0.0
        }

        advice: 2–4 sentences. Plain language. No clinical terms. No scores or numbers.
        Write as if speaking to a friend, not writing a report.

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
