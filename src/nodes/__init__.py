"""Node-based execution system for flexible AI processing.

This package provides a flexible node-based execution framework that replaces
the rigid sequential pipeline with a dynamic, queue-based system where:
- Nodes are discrete processing units
- Knowledge Broker accumulates context from all nodes
- Decision Engine selects which nodes to run
- Queue Manager handles priority-based execution
- Registry manages node lifecycle

Directory Structure:
- core/: Base classes, protocols, and types
- orchestration/: Central coordination components
- processing/: Content analysis and response nodes
- storage_nodes/: Persistence and storage nodes
- communication_nodes/: Network communication nodes
- algo_nodes/: Algorithm-specific nodes

Key Components:
- BaseNode: Abstract base class for all nodes
- NodeResult: Standard result format from node execution
- KnowledgeBroker: Central knowledge pool
- TaskQueueManager: Priority-based execution engine
- DecisionEngine: LLM-driven node selection
- NodeRegistry: Plugin-style node management
"""

from src.nodes.core import NodeResult, NodeStatus, BaseNode, NodeProtocol
from src.nodes.orchestration import KnowledgeBroker, TaskQueueManager, DecisionEngine, NodeRegistry

__all__ = [
    "NodeResult",
    "NodeStatus",
    "BaseNode",
    "NodeProtocol",
    "KnowledgeBroker",
    "TaskQueueManager",
    "DecisionEngine",
    "NodeRegistry",
]
