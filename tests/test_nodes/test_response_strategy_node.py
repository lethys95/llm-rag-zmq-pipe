import pytest
from unittest.mock import MagicMock

from src.nodes.algo_nodes.response_strategy_node import ResponseStrategyNode
from src.nodes.core.result import NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.models.response_strategy import ResponseStrategy


@pytest.fixture
def strategy():
    return ResponseStrategy(
        approach="reflective_listening",
        tone="empathetic_warm",
        needs_focus=["belonging"],
        system_prompt_addition="Validate their feelings. Don't problem-solve.",
        reasoning="High belonging need.",
    )


@pytest.fixture
def handler(strategy):
    m = MagicMock()
    m.select.return_value = strategy
    return m


@pytest.fixture
def node(handler):
    return ResponseStrategyNode(response_strategy_handler=handler)


@pytest.mark.asyncio
async def test_success_writes_to_broker(node, broker, strategy):
    result = await node.execute(broker)
    assert result.status == NodeStatus.SUCCESS
    assert broker.response_strategy is strategy


@pytest.mark.asyncio
async def test_handler_called_with_broker_fields(node, handler, broker):
    await node.execute(broker)
    handler.select.assert_called_once_with(
        needs_analysis=broker.needs_analysis,
        emotional_state=broker.emotional_state,
    )


@pytest.mark.asyncio
async def test_metadata_contains_approach_and_tone(node, broker, strategy):
    result = await node.execute(broker)
    assert result.metadata["approach"] == strategy.approach
    assert result.metadata["tone"] == strategy.tone


@pytest.mark.asyncio
async def test_handler_returning_none_fails(node, handler, broker):
    handler.select.return_value = None
    result = await node.execute(broker)
    assert result.status == NodeStatus.FAILED
    assert broker.response_strategy is None


@pytest.mark.asyncio
async def test_missing_dialogue_input_fails(node):
    empty_broker = KnowledgeBroker()
    result = await node.execute(empty_broker)
    assert result.status == NodeStatus.FAILED
