"""Coordinator — LLM-based node selection for each turn."""

import logging
from dataclasses import dataclass, field
from textwrap import dedent

from src.llm.base import BaseLLM, LLMResponse
from src.llm.tools import build_select_nodes_tool
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.storage.conversation_store import ConversationStore

logger = logging.getLogger(__name__)

_CONVERSATION_HISTORY_LIMIT = 30


@dataclass
class Coordinator:
    """Decides which node to run next within a single turn.

    Called repeatedly by the orchestrator until it returns None.
    Has access to the broker state (what this turn has produced so far)
    and recent conversation history (what the relationship looks like).
    """

    _llm_provider: BaseLLM
    _conversation_store: ConversationStore | None = field(default=None)

    def select_node(self, broker: KnowledgeBroker, registry: "NodeRegistry") -> str | None:  # noqa: F821
        available = registry.get_names()
        if not available:
            logger.warning("No nodes registered, cannot select")
            return None

        prompt = self._build_prompt(broker, registry)
        response: LLMResponse = self._llm_provider.generate_with_tools(
            prompt,
            [build_select_nodes_tool(registry)],
            tool_choice={"type": "function", "function": {"name": "select_node"}},
        )
        node_name = self._parse_tool_call(response)

        if node_name:
            logger.info("Coordinator selected: %s", node_name)
        else:
            logger.info("Coordinator: turn complete")

        return node_name

    def _parse_tool_call(self, response: LLMResponse) -> str | None:
        if not response.tool_calls:
            logger.warning("No tool calls in coordinator response")
            return None

        try:
            tool_call = response.tool_calls[0]
            if tool_call.function_name == "select_node":
                node_name = tool_call.arguments.get("node_name")
                if node_name == "complete":
                    return None
                return node_name
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
            which single processing node should run next for this event.

            You are processing an event. Events are not always user messages — they can
            be scheduled check-ins, idle-time reflections, internal state updates, or
            proactive companion initiations. The appropriate response to an event is not
            always a primary response to the user. Sometimes the right outcome is an
            internal state change only. Sometimes it is a proactive message the companion
            initiates itself. Sometimes the right choice is to do nothing.

            Use the event content, conversation history, and what has been produced so
            far to decide what — if anything — should happen next. "complete" means this
            event has been handled appropriately, not necessarily that a response was sent.

            {self._format_conversation_history()}

            Current event:
            {message}

            What this turn has produced so far:
            {broker.get_state_summary()}

            Nodes already run this turn: {already_run_str}

            Available nodes:
            {registry.get_menu()}

            Special value:
            - "complete": this event has been handled — a response was sent, an internal
              update was made, or inaction was the deliberate choice

            Select exactly one node to run next, or "complete" if finished.
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
