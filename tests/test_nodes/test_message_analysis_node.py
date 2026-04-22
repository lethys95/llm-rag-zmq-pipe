import pytest
from unittest.mock import MagicMock

from src.nodes.algo_nodes.message_analysis_node import MessageAnalysisNode
from src.nodes.core.result import NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.models.emotional_state import EmotionalState
from src.models.user_fact import UserFact


@pytest.fixture
def emotional_state():
    return EmotionalState(
        sadness=0.7,
        valence=-0.6,
        arousal=0.4,
        dominance=0.3,
        confidence=0.8,
    )


@pytest.fixture
def user_facts():
    return [
        UserFact(
            claim="user likes pizza",
            sentiment="positive",
            confidence=0.9,
            chrono_relevance=0.8,
            subject="food",
            memory_owner="user",
        )
    ]


@pytest.fixture
def emotional_handler(emotional_state):
    m = MagicMock()
    m.analyze.return_value = emotional_state
    return m


@pytest.fixture
def fact_handler(user_facts):
    m = MagicMock()
    m.extract.return_value = user_facts
    return m


@pytest.fixture
def node(emotional_handler, fact_handler):
    return MessageAnalysisNode(
        emotional_state_handler=emotional_handler,
        user_fact_extraction_handler=fact_handler,
    )


@pytest.mark.asyncio
async def test_success_writes_both_to_broker(node, broker, emotional_state, user_facts):
    result = await node.execute(broker)
    assert result.status == NodeStatus.SUCCESS
    assert broker.emotional_state is emotional_state
    assert broker.user_facts is user_facts


@pytest.mark.asyncio
async def test_handlers_called_with_correct_args(node, emotional_handler, fact_handler, broker):
    await node.execute(broker)
    emotional_handler.analyze.assert_called_once_with(broker.dialogue_input.content)
    fact_handler.extract.assert_called_once_with(
        broker.dialogue_input.content,
        broker.dialogue_input.speaker,
    )


@pytest.mark.asyncio
async def test_partial_when_emotional_state_fails(node, emotional_handler, broker):
    emotional_handler.analyze.return_value = None
    result = await node.execute(broker)
    assert result.status == NodeStatus.PARTIAL
    assert broker.emotional_state is None


@pytest.mark.asyncio
async def test_partial_when_no_facts_extracted(node, fact_handler, broker):
    fact_handler.extract.return_value = []
    result = await node.execute(broker)
    assert result.status == NodeStatus.PARTIAL


@pytest.mark.asyncio
async def test_failed_when_both_handlers_return_nothing(node, emotional_handler, fact_handler, broker):
    emotional_handler.analyze.return_value = None
    fact_handler.extract.return_value = []
    result = await node.execute(broker)
    assert result.status == NodeStatus.FAILED


@pytest.mark.asyncio
async def test_missing_dialogue_input_fails(node):
    empty_broker = KnowledgeBroker()
    result = await node.execute(empty_broker)
    assert result.status == NodeStatus.FAILED
