"""Decision engine for LLM-based node selection."""

import logging
from dataclasses import dataclass
from textwrap import dedent

from src.llm.base import (
    BaseLLM,
    LLMResponse,
)
from src.llm.tools import build_select_nodes_tool
from src.nodes.orchestration.node_options_registry import NodeOptionsRegistry

logger = logging.getLogger(__name__)


@dataclass
class DecisionEngine:
    """LLM-based reasoning engine to decide which node to run next.

    The decision engine analyzes the incoming message and current context
    to determine which single processing node should be executed next.
    It uses an LLM with function calling to make context-aware decisions.

    The orchestrator calls select_node() repeatedly until it returns None,
    indicating no more work is needed for this interaction.

    Node registration happens automatically via the @register_node decorator.
    """

    _llm_provider: BaseLLM

    def select_node(
        self,
        message: str
    ) -> str | None:
        """Select the next node to execute based on message and context.

        Uses the LLM with function calling to analyze the message and context,
        then returns a single node name to execute next. Returns None when
        no more processing is needed.

        Args:
            message: The user's message
            broker: Knowledge broker with current context

        Returns:
            Node name to execute, or None if no more work needed
        """
        registry = NodeOptionsRegistry.get_instance()
        available_nodes = registry.get_option_names()

        if not available_nodes:
            logger.warning("No nodes registered in the registry")
            return None

        #tools = self._build_tools_schema(available_nodes)
        prompt = self._build_selection_prompt(message, registry)

        response: LLMResponse = self._llm_provider.generate_with_tools(
            prompt,
            [build_select_nodes_tool(registry)],
            tool_choice={"type": "function", "function": {"name": "select_node"}},
        )
        node_name = self._parse_tool_call(response)

        if node_name:
            logger.info("Selected node: %s", node_name)
        else:
            logger.info("No node selected - processing complete")

        return node_name


    def _parse_tool_call(self, response: LLMResponse) -> str | None:
        """Parse tool call from LLM response.

        Args:
            response: LLMResponse with tool_calls

        Returns:
            Node name, or None if complete/no work needed
        """
        if not response.tool_calls:
            logger.warning("No tool calls in LLM response")
            return None

        try:
            tool_call = response.tool_calls[0]

            if tool_call.function_name == "select_node":
                node_name = tool_call.arguments.get("node_name")

                if node_name == "complete":
                    logger.debug("LLM indicated completion")
                    return None

                logger.debug("Parsed tool call with node: %s", node_name)
                return node_name
            
            logger.warning("Unexpected tool call: %s", tool_call.function_name)
            return None

        except (KeyError, TypeError):
            logger.exception("Failed to parse tool call")
            return None

    def _build_selection_prompt(
        self,
        message: str,
        registry: NodeOptionsRegistry,
    ) -> str:
        """Build prompt for LLM-based node selection.

        Args:
            message: The user's message
            broker: Knowledge broker with current context
            registry: Node options registry for getting node descriptions

        Returns:
            Formatted prompt for LLM
        """
        nodes_menu = registry.get_option_system_prompt_menu()

        return dedent(f"""\
            You are a decision engine for an AI companion system. Decide which single
            processing node should run next based on the user's message and context.

            Available Nodes:
            {nodes_menu}

            Special Values:
            - "complete": Use when no more processing is needed

            User Message: {message}

            Select exactly one node to execute next, or "complete" if finished.
            Consider what analysis or processing is still needed.
        """)
