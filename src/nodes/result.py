"""Node execution result types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeStatus(str, Enum):
    """Status of node execution."""
    
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


@dataclass
class NodeResult:
    """Result from node execution.
    
    Attributes:
        status: Execution status (SUCCESS, FAILED, SKIPPED, PARTIAL)
        data: Data produced by the node (stored in broker)
        error: Error message if failed, None otherwise
        next_nodes: List of node instances to enqueue based on this result
        metadata: Additional metadata about execution
    """
    
    status: NodeStatus
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    next_nodes: list[Any] = field(default_factory=list)  # List of BaseNode instances
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == NodeStatus.SUCCESS
    
    def is_failed(self) -> bool:
        """Check if execution failed."""
        return self.status == NodeStatus.FAILED
    
    def is_skipped(self) -> bool:
        """Check if execution was skipped."""
        return self.status == NodeStatus.SKIPPED
