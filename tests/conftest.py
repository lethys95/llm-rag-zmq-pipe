"""Shared fixtures."""

import pytest
from src.models.sentiment import DialogueInput
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker


@pytest.fixture
def dialogue_input():
    return DialogueInput(content="I feel really alone today.", speaker="user")


@pytest.fixture
def broker(dialogue_input):
    return KnowledgeBroker(dialogue_input=dialogue_input)
