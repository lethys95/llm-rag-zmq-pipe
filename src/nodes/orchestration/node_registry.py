"""Node registry — holds ready-to-run node instances built at startup."""

import inspect
import logging
import time

from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry_decorator import get_registered_classes

logger = logging.getLogger(__name__)


class NodeRegistry:
    """Holds ready-to-run node instances, built once at startup with deps injected.

    Usage:
        registry = NodeRegistry.build(
            zmq_handler=zmq,
            llm=llm_provider,
            rag=rag_provider,
            embedding_service=embeddings,
            conversation_store=store,
            primary_response_handler=handler,
        )
        result = await registry.execute("PrimaryResponseNode", broker)
    """

    def __init__(self) -> None:
        self._nodes: dict[str, BaseNode] = {}

    @classmethod
    def build(cls, **deps: object) -> "NodeRegistry":
        """Instantiate all registered node classes with their required deps.

        Each node's __init__ parameters are inspected by name. Any parameter
        whose name matches a key in deps is injected automatically. Parameters
        with no match are left to their default values (or raise TypeError if
        they have no default — which means a required dep is missing).
        """
        registry = cls()
        for node_cls in get_registered_classes():
            try:
                instance = cls._instantiate(node_cls, deps)
                registry._nodes[node_cls.__name__] = instance
                logger.debug("Registered node '%s'", node_cls.__name__)
            except Exception:
                logger.exception("Failed to instantiate node '%s'", node_cls.__name__)
        logger.info("NodeRegistry built with %d nodes", len(registry))
        return registry

    @staticmethod
    def _instantiate(node_cls: type[BaseNode], deps: dict[str, object]) -> BaseNode:
        sig = inspect.signature(node_cls.__init__)
        kwargs = {
            name: deps[name]
            for name in sig.parameters
            if name != "self" and name in deps
        }
        return node_cls(**kwargs)

    def get(self, name: str) -> BaseNode | None:
        return self._nodes.get(name)

    async def execute(self, name: str, broker: KnowledgeBroker) -> NodeResult | None:
        node = self.get(name)
        if not node:
            logger.error("Node '%s' not found in registry", name)
            return None
        t0 = time.monotonic()
        result = await node.execute(broker)
        duration_ms = (time.monotonic() - t0) * 1000
        if result is not None:
            result.metadata["duration_ms"] = duration_ms
            logger.debug("[node] %s → %s  %.0fms", name, result.status.value, duration_ms)
        return result

    def get_names(self) -> set[str]:
        return set(self._nodes.keys())

    def get_menu(self) -> str:
        """Build a human-readable menu of available nodes for the coordinator prompt."""
        lines = []
        for name, node in self._nodes.items():
            cls = node.__class__
            deps = cls.dependencies
            dep_str = f"  requires: {', '.join(deps)}" if deps else "  requires: none"
            crit = cls.min_criticality
            crit_str = f"  min_criticality: {crit:.1f} ({self._criticality_label(crit)})"
            lines.append(f"{name}\n{dep_str}\n{crit_str}\n  {node.get_description()}")
        return "\n\n".join(lines)

    @staticmethod
    def _criticality_label(value: float) -> str:
        if value == 0.0:
            return "always run"
        if value <= 0.2:
            return "skip only for completely trivial turns"
        if value <= 0.4:
            return "skip for casual turns with low emotional content"
        if value <= 0.6:
            return "skip unless moderate distress or complexity is present"
        return "skip unless significant distress or urgency is present"

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return f"<NodeRegistry nodes={list(self._nodes.keys())}>"
