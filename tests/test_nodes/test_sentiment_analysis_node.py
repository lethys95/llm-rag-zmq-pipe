"""Tests for SentimentAnalysisNode broker contract."""

import pytest
from unittest.mock import MagicMock

from src.nodes.algo_nodes.sentiment_analysis_node import SentimentAnalysisNode
from src.nodes.core.result import NodeStatus
from src.models.sentiment import SentimentAnalysis


@pytest.fixture
def sentiment():
    return SentimentAnalysis(sentiment="negative", confidence=0.9, memory_owner="user")


@pytest.fixture
def handler(sentiment):
    m = MagicMock()
    m.analyze.return_value = sentiment
    return m


@pytest.fixture
def node(handler):
    return SentimentAnalysisNode(sentiment_analysis_handler=handler)


@pytest.mark.asyncio
async def test_success_writes_to_broker(node, broker, sentiment):
    result = await node.execute(broker)
    assert result.status == NodeStatus.SUCCESS
    assert broker.sentiment_analysis is sentiment


@pytest.mark.asyncio
async def test_handler_called_with_correct_args(node, handler, broker):
    await node.execute(broker)
    handler.analyze.assert_called_once_with(
        message=broker.dialogue_input.content,
        speaker=broker.dialogue_input.speaker,
    )


@pytest.mark.asyncio
async def test_handler_returning_none_fails(node, handler, broker):
    handler.analyze.return_value = None
    result = await node.execute(broker)
    assert result.status == NodeStatus.FAILED
    assert broker.sentiment_analysis is None


@pytest.mark.asyncio
async def test_missing_dialogue_input_fails(node):
    from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
    empty_broker = KnowledgeBroker()
    result = await node.execute(empty_broker)
    assert result.status == NodeStatus.FAILED
