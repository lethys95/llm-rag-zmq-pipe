"""Coordinator — LLM-based node selection for each turn."""

import logging
from dataclasses import dataclass, field
from textwrap import dedent

from src.llm.base import BaseLLM, LLMResponse
from src.llm.tools import NO_ACTIONS_NEEDED_NODE_RESPONSE, build_select_nodes_tool
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.storage.conversation_store import ConversationStore

logger = logging.getLogger(__name__)

_CONVERSATION_HISTORY_LIMIT = 30


@dataclass
class Coordinator:
    """Decides which nodes to run next within a single turn.

    Returns a batch of node names that can be executed in parallel, or None
    when the turn is complete. Called repeatedly by the orchestrator until
    it returns None.
    """

    _llm_provider: BaseLLM
    _conversation_store: ConversationStore | None = field(default=None)

    def select_nodes(self, broker: KnowledgeBroker, registry: "NodeRegistry") -> list[str] | None:  # noqa: F821
        available = registry.get_names()
        if not available:
            logger.warning("No nodes registered, cannot select")
            return None

        prompt = self._build_prompt(broker, registry)
        response: LLMResponse = self._llm_provider.generate_with_tools(
            prompt,
            [build_select_nodes_tool(registry)],
            tool_choice={"type": "function", "function": {"name": "select_nodes"}},
        )
        batch = self._parse_tool_call(response)

        if batch:
            logger.info("Coordinator selected batch: %s", batch)
        else:
            logger.info("Coordinator: turn complete")

        return batch

    def _parse_tool_call(self, response: LLMResponse) -> list[str] | None:
        if not response.tool_calls:
            logger.warning("No tool calls in coordinator response")
            return None

        try:
            tool_call = response.tool_calls[0]
            if tool_call.function_name == "select_nodes":
                names = tool_call.arguments.get("node_names", [])
                if not names:
                    return None
                if names == [NO_ACTIONS_NEEDED_NODE_RESPONSE] or NO_ACTIONS_NEEDED_NODE_RESPONSE in names:
                    return None
                return names
            logger.warning("Unexpected tool call: %s", tool_call.function_name)
            return None
        except (KeyError, TypeError):
            logger.exception("Failed to parse coordinator tool call")
            return None

    def _build_prompt(self, broker: KnowledgeBroker, registry: "NodeRegistry") -> str:  # noqa: F821
        already_run = broker.metadata.execution_order
        already_run_str = ", ".join(already_run) if already_run else "none"
        message = broker.dialogue_input.content if broker.dialogue_input else "(no message)"

        return dedent(f"""\
            You are the coordinator for an AI companion system. Your job is to decide
            which processing nodes to run next — you may select multiple nodes to execute
            in parallel, or a single node, or signal completion.

            PARALLELISM RULES:
            - You may run multiple nodes in one batch if they are independent: none of them
              reads a broker field that another in the same batch writes.
            - Example safe parallel batch: [EmotionalStateNode, MemoryRetrievalNode] — both
              only read dialogue_input and write to different fields.
            - Example unsafe batch: [NeedsAnalysisNode, ResponseStrategyNode] — ResponseStrategyNode
              reads broker.needs_analysis which NeedsAnalysisNode writes. Run them in separate rounds.
            - Each node's "requires:" line tells you what must have been produced before that node
              will be meaningful. Honour these when batching.

            EVENT HANDLING:
            Events are not always user messages — they can be scheduled check-ins, idle-time
            reflections, internal state updates, or proactive companion initiations. The appropriate
            response to an event is not always a primary response to the user. Sometimes the right
            outcome is an internal state change only. Sometimes the right choice is to do nothing.

            Use 'complete' (alone) when: a response was sent, an internal update was made,
            or deliberate inaction was the correct choice.

            {self._format_conversation_history()}

            Current event:
            {message}

            What this turn has produced so far:
            {broker.get_state_summary()}

            Nodes already run this turn: {already_run_str}

            Available nodes (with dependencies and descriptions):
            {registry.get_menu()}

            Select a batch of independent nodes to run next, or ['complete'] if finished.
        """)

    def _format_conversation_history(self) -> str:
        if not self._conversation_store:
            return "Recent conversation history: not available"

        messages = self._conversation_store.get_recent_for_context(
            limit=_CONVERSATION_HISTORY_LIMIT
        )

        if not messages:
            return "Recent conversation history: none yet"

        lines = ["Recent conversation history (oldest first, with timestamps):"]
        for msg in messages:
            lines.append(f"  [{msg.timestamp}] {msg.speaker}: {msg.message}")
            if msg.response:
                lines.append(f"  [{msg.timestamp}] Companion: {msg.response}")

        return "\n".join(lines)
