"""Core node infrastructure and types.

This module contains the fundamental building blocks for the node system:
- BaseNode: Abstract base class for all nodes
- NodeResult: Standard result format from node execution
- NodeStatus: Enum for node execution status
- NodeProtocol: Protocol definition for node interface
"""

from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.core.base import BaseNode
from src.nodes.core.node_protocol import NodeProtocol

__all__ = [
    "NodeResult",
    "NodeStatus",
    "BaseNode",
    "NodeProtocol",
]
