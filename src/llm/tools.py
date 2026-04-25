from src.llm.base import FunctionDefinition, FunctionParameters, ToolDefinition

NO_ACTIONS_NEEDED_NODE_RESPONSE: str = "complete"


def build_select_nodes_tool(registry: "NodeRegistry") -> ToolDefinition:  # noqa: F821
    node_names = sorted(registry.get_names()) + [NO_ACTIONS_NEEDED_NODE_RESPONSE]

    return ToolDefinition(
        type="function",
        function=FunctionDefinition(
            name="select_nodes",
            description=(
                "Select one or more processing nodes to execute in parallel, or indicate completion. "
                "Nodes listed together must be genuinely independent — they must not read broker fields "
                "that another node in the same batch writes. Use 'complete' alone when no more work "
                "is needed this turn."
            ),
            parameters=FunctionParameters(
                type="object",
                properties={
                    "node_names": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": node_names,
                        },
                        "description": (
                            "List of node names to execute in parallel, or ['complete'] if finished. "
                            "Only group nodes that have no data dependency on each other within this batch."
                        ),
                        "minItems": 1,
                    }
                },
                required=["node_names"],
            ),
        ),
    )
