from src.llm.base import FunctionDefinition, FunctionParameters, ToolDefinition
from src.nodes.orchestration.node_options_registry import NodeOptionsRegistry


NO_ACTIONS_NEEDED_NODE_RESPONSE: str = "complete"

def build_select_nodes_tool(registry: NodeOptionsRegistry) -> ToolDefinition:
    """Build tools schema for function calling.
    Args:
        available_nodes: Set of registered node names
    Returns:
        Tool definition following OpenAI format
    """
    node_names = registry.get_option_names()
    node_names.add(NO_ACTIONS_NEEDED_NODE_RESPONSE)

    return ToolDefinition(
        type="function",
        function=FunctionDefinition(
            name="select_node",
            description="Select the next processing node to execute, or indicate completion",
            parameters=FunctionParameters(
                type="object",
                properties={
                    "node_name": {
                        "type": "string",
                        "enum": node_names,
                        "description": "Name of the node to execute next, or 'complete' if no more work needed",
                    }
                },
                required=["node_name"],
            ),
        ),
    )
