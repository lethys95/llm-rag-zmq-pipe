"""Knowledge broker for accumulating context across nodes."""

import logging
from typing import Any
from datetime import datetime

logger = logging.getLogger(__name__)


class KnowledgeBroker:
    """Central knowledge pool that accumulates context from all nodes.
    
    The knowledge broker acts as a shared context store where nodes can:
    - Read data produced by previously executed nodes
    - Write their own results for downstream nodes
    - Access accumulated knowledge for decision making
    
    This decouples nodes from each other - they only need to know what
    keys to read/write, not about other nodes' implementations.
    
    Attributes:
        context_pool: Dictionary storing all accumulated knowledge
        node_execution_order: List tracking the order nodes executed
        execution_metadata: Metadata about the overall execution
    """
    
    def __init__(self):
        """Initialize an empty knowledge broker."""
        self.context_pool: dict[str, Any] = {}
        self.node_execution_order: list[str] = []
        self.execution_metadata: dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "total_nodes_executed": 0,
            "failed_nodes": [],
            "skipped_nodes": [],
        }
        
        logger.debug("Knowledge broker initialized")
    
    def add_knowledge(self, key: str, data: Any, node_name: str | None = None) -> None:
        """Add knowledge to the broker.
        
        Args:
            key: The key to store data under
            data: The data to store (any type)
            node_name: Optional name of the node adding this knowledge
        """
        self.context_pool[key] = data
        
        if node_name:
            logger.debug(
                f"Node '{node_name}' added knowledge: '{key}' "
                f"(type: {type(data).__name__})"
            )
        else:
            logger.debug(f"Added knowledge: '{key}' (type: {type(data).__name__})")
    
    def get_knowledge(self, key: str, default: Any = None) -> Any:
        """Get knowledge from the broker.
        
        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The stored value or default if not found
        """
        value = self.context_pool.get(key, default)
        
        if value is None and default is None:
            logger.debug(f"Knowledge key '{key}' not found, returning None")
        
        return value
    
    def has_knowledge(self, key: str) -> bool:
        """Check if a key exists in the broker.
        
        Args:
            key: The key to check
            
        Returns:
            True if key exists, False otherwise
        """
        return key in self.context_pool
    
    def get_full_context(self) -> dict[str, Any]:
        """Get all accumulated knowledge.
        
        Returns:
            Complete context pool as a dictionary
        """
        return self.context_pool.copy()
    
    def record_node_execution(
        self,
        node_name: str,
        status: str,
        duration: float | None = None
    ) -> None:
        """Record that a node was executed.
        
        Args:
            node_name: Name of the executed node
            status: Execution status (success, failed, skipped)
            duration: Optional execution duration in seconds
        """
        self.node_execution_order.append(node_name)
        self.execution_metadata["total_nodes_executed"] += 1
        
        if status == "failed":
            self.execution_metadata["failed_nodes"].append(node_name)
        elif status == "skipped":
            self.execution_metadata["skipped_nodes"].append(node_name)
        
        if duration is not None:
            key = f"_duration_{node_name}"
            self.context_pool[key] = duration
        
        logger.debug(
            f"Recorded execution: '{node_name}' ({status})"
            + (f" in {duration:.3f}s" if duration else "")
        )
    
    def get_execution_summary(self) -> dict[str, Any]:
        """Get summary of execution statistics.
        
        Returns:
            Dictionary with execution metadata and statistics
        """
        return {
            "metadata": self.execution_metadata.copy(),
            "execution_order": self.node_execution_order.copy(),
            "total_knowledge_keys": len(self.context_pool),
        }
    
    def clear(self) -> None:
        """Clear all knowledge and reset the broker.
        
        Use this between requests to start fresh.
        """
        self.context_pool.clear()
        self.node_execution_order.clear()
        self.execution_metadata = {
            "created_at": datetime.now().isoformat(),
            "total_nodes_executed": 0,
            "failed_nodes": [],
            "skipped_nodes": [],
        }
        
        logger.debug("Knowledge broker cleared")
    
    def __repr__(self) -> str:
        """String representation of the broker."""
        return (
            f"<KnowledgeBroker "
            f"keys={len(self.context_pool)} "
            f"nodes_executed={len(self.node_execution_order)}>"
        )
