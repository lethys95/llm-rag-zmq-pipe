import pytest
from unittest.mock import MagicMock

from src.nodes.algo_nodes.strategy_advisor_node import StrategyAdvisorNode
from src.nodes.core.result import NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.models.advisor import AdvisorOutput
from src.models.analysis import NeedsAnalysis


@pytest.fixture
def advisor_output():
    return AdvisorOutput(advisor="strategy", advice="Stay warm and don't push.", potency=0.6)


@pytest.fixture
def handler(advisor_output):
    m = MagicMock()
    m.advise.return_value = advisor_output
    return m


@pytest.fixture
def node(handler):
    return StrategyAdvisorNode(strategy_advisor_handler=handler)


@pytest.mark.asyncio
async def test_appends_to_advisor_outputs(node, broker, advisor_output):
    result = await node.execute(broker)
    assert result.status == NodeStatus.SUCCESS
    assert advisor_output in broker.advisor_outputs


@pytest.mark.asyncio
async def test_passes_response_strategy_from_broker(node, handler, broker):
    broker.response_strategy = MagicMock()
    await node.execute(broker)
    _, kwargs = handler.advise.call_args
    assert kwargs["response_strategy"] is broker.response_strategy


@pytest.mark.asyncio
async def test_passes_urgency_from_needs_analysis(node, handler, broker):
    broker.needs_analysis = MagicMock(need_urgency=0.6)
    await node.execute(broker)
    _, kwargs = handler.advise.call_args
    assert kwargs["need_urgency"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_zero_urgency_when_no_needs_analysis(node, handler, broker):
    broker.needs_analysis = None
    await node.execute(broker)
    _, kwargs = handler.advise.call_args
    assert kwargs["need_urgency"] == 0.0


@pytest.mark.asyncio
async def test_metadata_contains_advisor_name(node, broker, advisor_output):
    result = await node.execute(broker)
    assert result.metadata["advisor"] == "strategy"
    assert result.metadata["potency"] == advisor_output.potency


@pytest.mark.asyncio
async def test_multiple_advisors_accumulate(node, broker):
    existing = AdvisorOutput(advisor="memory", advice="something", potency=0.8)
    broker.advisor_outputs.append(existing)
    await node.execute(broker)
    assert len(broker.advisor_outputs) == 2


@pytest.mark.asyncio
async def test_works_without_dialogue_input(node, handler, advisor_output):
    empty_broker = KnowledgeBroker()
    result = await node.execute(empty_broker)
    assert result.status == NodeStatus.SUCCESS
