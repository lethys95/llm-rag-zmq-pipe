"""Node-based execution system for flexible AI processing.

This package provides a flexible node-based execution framework that replaces
the rigid sequential pipeline with a dynamic, queue-based system where:
- Nodes are discrete processing units
- Knowledge Broker accumulates context from all nodes
- Decision Engine selects which nodes to run
- Queue Manager handles priority-based execution
- Registry manages node lifecycle

Key Components:
- BaseNode: Abstract base class for all nodes
- NodeResult: Standard result format from node execution
- KnowledgeBroker: Central knowledge pool
- TaskQueueManager: Priority-based execution engine
- DecisionEngine: LLM-driven node selection
- NodeRegistry: Plugin-style node management
"""

from .result import NodeResult, NodeStatus
from .base import BaseNode
from .knowledge_broker import KnowledgeBroker
from .queue_manager import TaskQueueManager
from .decision_engine import DecisionEngine
from .registry import NodeRegistry

__all__ = [
    "NodeResult",
    "NodeStatus",
    "BaseNode",
    "KnowledgeBroker",
    "TaskQueueManager",
    "DecisionEngine",
    "NodeRegistry",
]
