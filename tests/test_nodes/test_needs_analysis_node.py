import pytest
from unittest.mock import MagicMock

from src.nodes.algo_nodes.needs_analysis_node import NeedsAnalysisNode
from src.nodes.core.result import NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.models.analysis import NeedsAnalysis


@pytest.fixture
def needs_analysis():
    return NeedsAnalysis(
        belonging=0.7, meaning=0.8,
        primary_needs=["meaning", "belonging"],
        unmet_needs=["meaning", "belonging"],
        need_urgency=0.5,
        need_persistence=0.8,
        context_summary="Struggling with purpose and connection.",
    )


@pytest.fixture
def handler(needs_analysis):
    m = MagicMock()
    m.analyze.return_value = needs_analysis
    return m


@pytest.fixture
def node(handler):
    return NeedsAnalysisNode(needs_analysis_handler=handler)


@pytest.mark.asyncio
async def test_success_writes_to_broker(node, broker, needs_analysis):
    result = await node.execute(broker)
    assert result.status == NodeStatus.SUCCESS
    assert broker.needs_analysis is needs_analysis


@pytest.mark.asyncio
async def test_handler_called_with_broker_fields(node, handler, broker):
    await node.execute(broker)
    handler.analyze.assert_called_once_with(
        message=broker.dialogue_input.content,
        speaker=broker.dialogue_input.speaker,
        emotional_state=None,
        retrieved_documents=broker.retrieved_documents,
    )


@pytest.mark.asyncio
async def test_metadata_contains_primary_needs_and_urgency(node, broker, needs_analysis):
    result = await node.execute(broker)
    assert result.metadata["primary_needs"] == needs_analysis.primary_needs
    assert result.metadata["urgency"] == pytest.approx(needs_analysis.need_urgency)


@pytest.mark.asyncio
async def test_handler_returning_none_fails(node, handler, broker):
    handler.analyze.return_value = None
    result = await node.execute(broker)
    assert result.status == NodeStatus.FAILED
    assert broker.needs_analysis is None


@pytest.mark.asyncio
async def test_missing_dialogue_input_fails(node):
    empty_broker = KnowledgeBroker()
    result = await node.execute(empty_broker)
    assert result.status == NodeStatus.FAILED
