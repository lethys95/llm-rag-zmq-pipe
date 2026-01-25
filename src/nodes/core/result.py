"""Node execution result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.nodes.core.node_protocol import NodeProtocol


class NodeStatus(str, Enum):
    """Status of node execution."""
    
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


@dataclass
class NodeResult:
    """Result from node execution."""
    
    status: NodeStatus
    data: dict = field(default_factory=dict)
    error: str | None = None
    next_nodes: list[NodeProtocol] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == NodeStatus.SUCCESS
    
    def is_failed(self) -> bool:
        """Check if execution failed."""
        return self.status == NodeStatus.FAILED
    
    def is_skipped(self) -> bool:
        """Check if execution was skipped."""
        return self.status == NodeStatus.SKIPPED
