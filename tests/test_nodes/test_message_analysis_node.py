import pytest
from unittest.mock import MagicMock

from src.nodes.algo_nodes.message_analysis_node import MessageAnalysisNode
from src.nodes.core.result import NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.models.user_fact import UserFact


@pytest.fixture
def user_facts():
    return [
        UserFact(
            claim="user likes pizza",
            chrono_relevance=0.8,
            subject="food",
            memory_owner="user",
        )
    ]


@pytest.fixture
def fact_handler(user_facts):
    m = MagicMock()
    m.extract.return_value = user_facts
    return m


@pytest.fixture
def node(fact_handler):
    return MessageAnalysisNode(user_fact_extraction_handler=fact_handler)


@pytest.mark.asyncio
async def test_success_writes_facts_to_broker(node, broker, user_facts):
    result = await node.execute(broker)
    assert result.status == NodeStatus.SUCCESS
    assert broker.user_facts is user_facts


@pytest.mark.asyncio
async def test_handler_called_with_correct_args(node, fact_handler, broker):
    await node.execute(broker)
    fact_handler.extract.assert_called_once_with(
        broker.dialogue_input.content,
        broker.dialogue_input.speaker,
        emotional_state=broker.emotional_state,
    )


@pytest.mark.asyncio
async def test_partial_when_no_facts_extracted(node, fact_handler, broker):
    fact_handler.extract.return_value = []
    result = await node.execute(broker)
    assert result.status == NodeStatus.PARTIAL


@pytest.mark.asyncio
async def test_missing_dialogue_input_fails(node):
    empty_broker = KnowledgeBroker()
    result = await node.execute(empty_broker)
    assert result.status == NodeStatus.FAILED
