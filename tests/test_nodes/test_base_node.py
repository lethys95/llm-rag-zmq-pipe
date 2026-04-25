"""Tests for BaseNode fixes."""

import pytest
from src.nodes.core.base_node import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker


class _NamedNode(BaseNode):
    def __init__(self):
        super().__init__(name="my_node")

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        return NodeResult(status=NodeStatus.SUCCESS)


class _UnnamedNode(BaseNode):
    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        return NodeResult(status=NodeStatus.SUCCESS)


class _CustomDescriptionNode(BaseNode):
    def get_description(self) -> str:
        return "does something special"

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        return NodeResult(status=NodeStatus.SUCCESS)


def test_explicit_name_is_used():
    assert _NamedNode().name == "my_node"


def test_name_defaults_to_snake_case_class_name():
    node = _UnnamedNode()
    assert node.name == "__unnamed"


def test_get_description_defaults_to_name():
    node = _NamedNode()
    assert node.get_description() == node.name


def test_get_description_can_be_overridden():
    node = _CustomDescriptionNode()
    assert node.get_description() == "does something special"
