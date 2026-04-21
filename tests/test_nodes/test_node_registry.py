"""Tests for NodeRegistry DI mechanism."""

import pytest
from unittest.mock import patch, MagicMock

from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry import NodeRegistry


# ---------------------------------------------------------------------------
# Test nodes — NOT decorated with @register_node to avoid polluting global state

class _NoDepsNode(BaseNode):
    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        return NodeResult(status=NodeStatus.SUCCESS)


class _OneDepsNode(BaseNode):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        return NodeResult(status=NodeStatus.SUCCESS)


class _TwoDepsNode(BaseNode):
    def __init__(self, llm, rag):
        super().__init__()
        self.llm = llm
        self.rag = rag

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        return NodeResult(status=NodeStatus.SUCCESS)


class _RequiredMissingNode(BaseNode):
    def __init__(self, nonexistent_dep):
        super().__init__()
        self.dep = nonexistent_dep

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        return NodeResult(status=NodeStatus.SUCCESS)


# ---------------------------------------------------------------------------

@pytest.fixture
def deps():
    return {"llm": MagicMock(), "rag": MagicMock()}


def test_instantiate_no_deps():
    node = NodeRegistry._instantiate(_NoDepsNode, {})
    assert isinstance(node, _NoDepsNode)


def test_instantiate_injects_matching_dep(deps):
    node = NodeRegistry._instantiate(_OneDepsNode, deps)
    assert node.llm is deps["llm"]


def test_instantiate_injects_all_matching_deps(deps):
    node = NodeRegistry._instantiate(_TwoDepsNode, deps)
    assert node.llm is deps["llm"]
    assert node.rag is deps["rag"]


def test_instantiate_ignores_extra_deps(deps):
    """Extra deps that don't match any parameter are silently ignored."""
    extra = {**deps, "unused_dep": MagicMock()}
    node = NodeRegistry._instantiate(_OneDepsNode, extra)
    assert node.llm is deps["llm"]


def test_instantiate_raises_on_missing_required_dep():
    with pytest.raises(TypeError):
        NodeRegistry._instantiate(_RequiredMissingNode, {})


def test_build_skips_node_with_missing_dep_and_continues(deps):
    """A node that can't be instantiated is logged and skipped; others still register."""
    test_classes = frozenset({_OneDepsNode, _RequiredMissingNode})
    with patch(
        "src.nodes.orchestration.node_registry.get_registered_classes",
        return_value=test_classes,
    ):
        registry = NodeRegistry.build(**deps)

    assert "_OneDepsNode" in registry.get_names()
    assert "_RequiredMissingNode" not in registry.get_names()


def test_build_populates_registry(deps):
    test_classes = frozenset({_NoDepsNode, _OneDepsNode})
    with patch(
        "src.nodes.orchestration.node_registry.get_registered_classes",
        return_value=test_classes,
    ):
        registry = NodeRegistry.build(**deps)

    assert registry.get_names() == {"_NoDepsNode", "_OneDepsNode"}


def test_get_returns_none_for_unknown_node(deps):
    with patch(
        "src.nodes.orchestration.node_registry.get_registered_classes",
        return_value=frozenset(),
    ):
        registry = NodeRegistry.build(**deps)

    assert registry.get("NonExistentNode") is None


def test_get_menu_contains_node_names(deps):
    test_classes = frozenset({_NoDepsNode, _OneDepsNode})
    with patch(
        "src.nodes.orchestration.node_registry.get_registered_classes",
        return_value=test_classes,
    ):
        registry = NodeRegistry.build(**deps)

    menu = registry.get_menu()
    assert "_NoDepsNode" in menu
    assert "_OneDepsNode" in menu


def test_len(deps):
    test_classes = frozenset({_NoDepsNode, _OneDepsNode, _TwoDepsNode})
    with patch(
        "src.nodes.orchestration.node_registry.get_registered_classes",
        return_value=test_classes,
    ):
        registry = NodeRegistry.build(**deps)

    assert len(registry) == 3
