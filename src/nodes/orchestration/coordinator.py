"""Coordinator — LLM-based node selection for each request."""

import logging
from dataclasses import dataclass
from textwrap import dedent

from src.llm.base import BaseLLM, LLMResponse
from src.llm.tools import build_select_nodes_tool
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker

logger = logging.getLogger(__name__)


@dataclass
class Coordinator:
    """LLM-based coordinator that decides which nodes to run next.

    Called repeatedly by the orchestrator until it returns None,
    indicating no more work is needed for this interaction.
    """

    _llm_provider: BaseLLM

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
            logger.info("Coordinator: processing complete")

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
            You are a coordinator for an AI companion system. Decide which single
            processing node should run next.

            Available nodes:
            {registry.get_menu()}

            Special value:
            - "complete": no more processing needed for this message

            User message: {message}
            Nodes already executed this turn: {already_run_str}

            Select exactly one node to execute next, or "complete" if finished.
        """)
