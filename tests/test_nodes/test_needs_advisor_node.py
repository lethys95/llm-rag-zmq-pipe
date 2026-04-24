import pytest
from unittest.mock import MagicMock

from src.nodes.algo_nodes.needs_advisor_node import NeedsAdvisorNode
from src.nodes.core.result import NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.models.advisor import AdvisorOutput


@pytest.fixture
def advisor_output():
    return AdvisorOutput(advisor="needs", advice="They need connection right now.", potency=0.7)


@pytest.fixture
def handler(advisor_output):
    m = MagicMock()
    m.advise.return_value = advisor_output
    return m


@pytest.fixture
def node(handler):
    return NeedsAdvisorNode(needs_advisor_handler=handler)


@pytest.mark.asyncio
async def test_appends_to_advisor_outputs(node, broker, advisor_output):
    result = await node.execute(broker)
    assert result.status == NodeStatus.SUCCESS
    assert advisor_output in broker.advisor_outputs


@pytest.mark.asyncio
async def test_handler_called_with_broker_needs_analysis(node, handler, broker):
    await node.execute(broker)
    handler.advise.assert_called_once_with(needs_analysis=broker.needs_analysis)


@pytest.mark.asyncio
async def test_works_without_needs_analysis(node, broker):
    broker.needs_analysis = None
    result = await node.execute(broker)
    assert result.status == NodeStatus.SUCCESS


@pytest.mark.asyncio
async def test_works_without_dialogue_input(node, handler, advisor_output):
    empty_broker = KnowledgeBroker()
    result = await node.execute(empty_broker)
    assert result.status == NodeStatus.SUCCESS
    assert advisor_output in empty_broker.advisor_outputs


@pytest.mark.asyncio
async def test_metadata_contains_potency(node, broker, advisor_output):
    result = await node.execute(broker)
    assert result.metadata["potency"] == advisor_output.potency
    assert result.metadata["advisor"] == "needs"


@pytest.mark.asyncio
async def test_multiple_advisors_accumulate(node, broker):
    existing = AdvisorOutput(advisor="memory", advice="something", potency=0.8)
    broker.advisor_outputs.append(existing)
    await node.execute(broker)
    assert len(broker.advisor_outputs) == 2
