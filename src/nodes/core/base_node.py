"""Base node class for execution nodes."""

import logging
from abc import ABC, abstractmethod

from src.nodes.core.result import NodeResult
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
import re

logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """Abstract base class for all execution nodes.

    All nodes must inherit from this class and implement the execute() method.
    Nodes can optionally override should_run() for conditional execution.

    Attributes:
        name: Unique identifier for this node
        priority: Execution priority (0 = highest, higher numbers = lower priority)
        dependencies: List of node names that must complete successfully before this node
        queue_type: Either "immediate" (blocks user response) or "background" (async)
        timeout: Optional timeout in seconds for node execution
    """

    def __init__(self, name: str | None, timeout: float | None = None):
        """Initialize the base node.

        Args:
            name: Unique identifier for this node
            priority: Execution priority (0 = highest, default: 5)
            dependencies: List of node names that must complete first
            queue_type: "immediate" or "background" (default: "immediate")
            timeout: Optional timeout in seconds (default: None)
        """
        self.name = name
        self.timeout = timeout
        if not name:
            name = self._make_simple_name()

        logger.debug(f"Node '{self.name}' initialized: ")

    @abstractmethod
    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        """Execute the node's primary logic.

        This method must be implemented by all concrete nodes. It should:
        1. Read any needed context from the broker
        2. Perform its processing
        3. Write results back to the broker via NodeResult.data
        4. Return a NodeResult with status and optional next_nodes

        Args:
            broker: Knowledge broker for reading/writing context

        Returns:
            NodeResult with status, data, and optional next nodes to enqueue
        """
        pass

    def should_run(self, broker: KnowledgeBroker) -> bool:
        """Determine if this node should run based on current context.

        Override this method to implement conditional execution logic.
        Default implementation always returns True.

        Args:
            broker: Knowledge broker to check conditions against

        Returns:
            True if node should execute, False to skip
        """
        ...

    def _make_simple_name(self) -> str:
        """
        Create snake_case name from class
        """
        some_name = self.__class__.__name__.replace("Node", "")
        return re.sub(r"(?<!^)(?=[A-Z])", "_", some_name).lower()

    @abstractmethod
    def get_description(self) -> str: ...

    # def validate_dependencies(self, completed_nodes: set[str]) -> bool:
    #     """Check if all dependencies have been completed.

    #     Args:
    #         completed_nodes: Set of node names that have completed successfully

    #     Returns:
    #         True if all dependencies are met, False otherwise
    #     """
    #     if not self.dependencies:
    #         return True

    #     missing = set(self.dependencies) - completed_nodes

    #     if missing:
    #         logger.debug(
    #             f"Node '{self.name}' waiting for dependencies: {missing}"
    #         )
    #         return False

    #     return True

    # def __lt__(self, other: Self) -> bool:
    #     """Compare nodes by priority for queue ordering.

    #     Lower priority number = higher precedence (executes first).

    #     Args:
    #         other: Another BaseNode to compare against

    #     Returns:
    #         True if this node has higher priority (lower number)
    #     """
    #     return self.priority < other.priority

    # def __repr__(self) -> str:
    #    """String representation of the node."""
    #    return (
    #        f"<{self.__class__.__name__} "
    #        f"name='{self.name}' "
    #        f"priority={self.priority} "
    #        f"queue={self.queue_type}>"
    #    )
