import json
import logging
import time
from textwrap import dedent

from src.config.settings import WorkerCallConfig
from src.handlers.handler_registry_decorator import register_handler
from src.llm.base import BaseLLM
from src.models.analysis import MemoryEvaluation
from src.models.emotional_state import EmotionalState
from src.rag.selector import RAGDocument

logger = logging.getLogger(__name__)


@register_handler
class MemoryEvaluationHandler:
    """Evaluates retrieved memories for relevance to the current conversational moment.

    Takes raw documents from memory retrieval and asks an LLM to assess each one:
    how relevant is it right now, how long will that relevance persist, and why.
    The reasoning is what the primary LLM reads — it should explain the connection
    in plain language, not clinical notation.
    """

    SYSTEM_PROMPT = dedent("""\
        You are evaluating which past memories are meaningful in the context of what a user is saying right now.

        For each memory, assess:
          relevance      — how directly connected is this memory to the current message and emotional state
                           0.0 = unrelated, 1.0 = directly relevant to what is happening right now
          chrono_relevance — how long will this connection persist going forward
                           0.0 = fleeting (this connection is specific to this moment only)
                           1.0 = enduring (this memory will remain relevant for a long time)
          reasoning      — one sentence explaining the connection in plain language

        Be honest. If a memory is not relevant, say so with a low score and a brief reason.
        The reasoning will be read by the companion to understand the user — keep it direct and human.

        Respond ONLY with a JSON object containing an "evaluations" key whose value is an array,
        one entry per memory, in the same order as provided:
        {
          "evaluations": [
            {"index": 0, "relevance": 0.0, "chrono_relevance": 0.0, "reasoning": "..."},
            ...
          ]
        }

        Return ONLY valid JSON. No explanation, no extra text.""")

    def __init__(self, worker_llm: BaseLLM, worker_call: WorkerCallConfig) -> None:
        self.llm = worker_llm
        self.max_retries = worker_call.max_retries
        self.retry_delay = worker_call.retry_delay

    def evaluate(
        self,
        message: str,
        documents: list[RAGDocument],
        emotional_state: EmotionalState | None = None,
    ) -> list[tuple[RAGDocument, MemoryEvaluation]]:
        if not documents:
            return []

        prompt = self._build_prompt(message, documents, emotional_state)

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.llm.generate(prompt, json_mode=True)
                result = self._parse(raw, documents)
                if result is not None:
                    logger.debug(
                        "MemoryEvaluation: %d/%d memories evaluated",
                        len(result), len(documents),
                    )
                    return result
            except Exception:
                logger.exception("MemoryEvaluationHandler attempt %d/%d failed", attempt, self.max_retries)

            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        logger.error("MemoryEvaluationHandler failed after %d attempts", self.max_retries)
        return []

    def _build_prompt(
        self,
        message: str,
        documents: list[RAGDocument],
        emotional_state: EmotionalState | None,
    ) -> str:
        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Current message:\n{message}\n\n"
            f"Emotional state:\n{self._format_emotional_state(emotional_state)}\n\n"
            f"Memories to evaluate:\n{self._format_documents(documents)}\n\n"
            f"JSON:"
        )

    def _format_emotional_state(self, state: EmotionalState | None) -> str:
        if not state:
            return "Not available."
        dominant = sorted(
            [(k, v) for k, v in state.model_dump().items()
             if k not in ("valence", "arousal", "dominance", "confidence", "summary")
             and isinstance(v, float) and v > 0.05],
            key=lambda x: x[1], reverse=True,
        )[:4]
        emotion_str = ", ".join(f"{k}: {v:.2f}" for k, v in dominant) if dominant else "neutral"
        return (
            f"Dominant emotions: {emotion_str}\n"
            f"Valence: {state.valence:.2f}  Arousal: {state.arousal:.2f}  "
            f"Dominance: {state.dominance:.2f}"
        )

    def _format_documents(self, documents: list[RAGDocument]) -> str:
        lines = []
        for i, doc in enumerate(documents):
            lines.append(f"[{i}] {doc.content}")
        return "\n".join(lines)

    def _parse(
        self,
        raw: str,
        documents: list[RAGDocument],
    ) -> list[tuple[RAGDocument, MemoryEvaluation]] | None:
        try:
            text = raw.strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                raise json.JSONDecodeError("No JSON object found", text, 0)
            data = json.loads(text[start:end + 1])
            items = data["evaluations"]

            results = []
            for item in items:
                idx = item["index"]
                if not (0 <= idx < len(documents)):
                    logger.warning("MemoryEvaluation: index %d out of range, skipping", idx)
                    continue
                evaluation = MemoryEvaluation(
                    relevance=float(item["relevance"]),
                    chrono_relevance=float(item["chrono_relevance"]),
                    reasoning=str(item["reasoning"]),
                )
                results.append((documents[idx], evaluation))
            return results

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error("MemoryEvaluationHandler parse error: %s", e)
            return None
