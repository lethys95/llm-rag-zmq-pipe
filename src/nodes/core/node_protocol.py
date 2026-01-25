"""Protocol for node interface to avoid circular imports."""

from __future__ import annotations

from typing import Protocol


class NodeProtocol(Protocol):
    """Protocol defining the interface for execution nodes.
    
    This allows result.py to type next_nodes without importing BaseNode,
    avoiding circular imports while maintaining type safety.
    
    We don't import KnowledgeBroker or NodeResult here to keep this
    file free of dependencies.
    """
    
    name: str
    priority: int
    dependencies: list[str]
    queue_type: str
    timeout: float | None
