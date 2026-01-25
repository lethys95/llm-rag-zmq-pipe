"""Node registry for managing available execution nodes."""

import logging
from dataclasses import dataclass
from src.nodes.core.base import BaseNode

@dataclass
class NodeInfo:
    name: str
    class_name: str
    module: str
    docstring: str | None

logger = logging.getLogger(__name__)


class NodeRegistry:
    """Central registry of all available execution nodes.
    
    The registry manages node classes and provides factory methods for
    creating node instances. It acts as a plugin system where nodes
    can be registered and instantiated dynamically.
    
    Future enhancements:
    - Auto-discovery via decorators
    - Config-based enable/disable
    - Per-user node customization
    
    Attributes:
        _nodes: Dictionary mapping node names to node classes
        _instance: Singleton instance
    """
    
    _instance = None
    
    def __init__(self):
        """Initialize the node registry."""
        self._nodes: dict[str, type["BaseNode"]] = {}
        logger.debug("Node registry initialized")
    
    @classmethod
    def get_instance(cls) -> "NodeRegistry":
        """Get singleton instance of the registry.
        
        Returns:
            NodeRegistry singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
            logger.debug("Created new NodeRegistry singleton instance")
        return cls._instance
    
    def register(self, node_class: type["BaseNode"], name: str | None = None) -> None:
        """Register a node class.
        
        Args:
            node_class: The node class to register (must inherit from BaseNode)
            name: Optional custom name (defaults to class name)
        """
        node_name = name or node_class.__name__
        
        if node_name in self._nodes:
            logger.warning(f"Overwriting existing node registration: '{node_name}'")
        
        self._nodes[node_name] = node_class
        logger.info(f"Registered node: '{node_name}' ({node_class.__name__})")
    
    def unregister(self, name: str) -> bool:
        """Unregister a node class.
        
        Args:
            name: Name of the node to unregister
            
        Returns:
            True if node was unregistered, False if not found
        """
        if name in self._nodes:
            del self._nodes[name]
            logger.info(f"Unregistered node: '{name}'")
            return True
        
        logger.warning(f"Cannot unregister node '{name}': not found")
        return False
    
    def create(self, name: str, **kwargs) -> "BaseNode":
        """Create a node instance by name.
        
        Args:
            name: Name of the registered node
            **kwargs: Arguments to pass to the node constructor
            
        Returns:
            Instance of the requested node
            
        Raises:
            KeyError: If node name is not registered
        """
        if name not in self._nodes:
            available = ', '.join(self._nodes.keys())
            raise KeyError(
                f"Node '{name}' not registered. Available nodes: {available}"
            )
        
        node_class = self._nodes[name]
        node_instance = node_class(**kwargs)
        
        logger.debug(f"Created node instance: '{name}'")
        return node_instance
    
    def is_registered(self, name: str) -> bool:
        """Check if a node is registered.
        
        Args:
            name: Name of the node to check
            
        Returns:
            True if registered, False otherwise
        """
        return name in self._nodes
    
    def list_available(self) -> list[str]:
        """Get list of all registered node names.
        
        Returns:
            List of registered node names
        """
        return list(self._nodes.keys())
    
    def get_node_info(self, name: str) -> NodeInfo:
        """Get metadata about a registered node.
        
        Args:
            name: Name of the node
            
        Returns:
            NodeInfo dataclass with node metadata
            
        Raises:
            KeyError: If node name is not registered
        """
        if name not in self._nodes:
            raise KeyError(f"Node '{name}' not registered")
        
        node_class = self._nodes[name]
        
        return NodeInfo(
            name=name,
            class_name=node_class.__name__,
            module=node_class.__module__,
            docstring=node_class.__doc__,
        )
    
    def clear(self) -> None:
        """Clear all registered nodes.
        
        Use with caution - this will remove all node registrations.
        """
        count = len(self._nodes)
        self._nodes.clear()
        logger.warning(f"Cleared all {count} registered nodes")
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"<NodeRegistry nodes={len(self._nodes)}>"
