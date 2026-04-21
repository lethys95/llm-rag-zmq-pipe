from src.nodes.core.base_node import BaseNode
from src.nodes.orchestration.node_options_registry import NodeOptionsRegistry


def register_node(cls: type[BaseNode]) -> type[BaseNode]:
    if not issubclass(cls, BaseNode):
        raise TypeError(f"{cls.__name__} must inherit from BaseNode")
    registry = NodeOptionsRegistry.get_instance()
    registry._registered_nodes[cls.__name__] = cls
    return cls
