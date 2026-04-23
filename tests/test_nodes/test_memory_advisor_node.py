import pytest
from unittest.mock import MagicMock

from src.nodes.algo_nodes.memory_advisor_node import MemoryAdvisorNode
from src.nodes.core.result import NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.models.advisor import AdvisorOutput
from src.models.analysis import MemoryEvaluation
from src.rag.selector import RAGDocument


@pytest.fixture
def advisor_output():
    return AdvisorOutput(
        advisor="memory",
        advice="This person has been grieving since their mother died.",
        potency=0.85,
    )


@pytest.fixture
def handler(advisor_output):
    m = MagicMock()
    m.advise.return_value = advisor_output
    return m


@pytest.fixture
def node(handler):
    return MemoryAdvisorNode(memory_advisor_handler=handler)


@pytest.mark.asyncio
async def test_appends_to_advisor_outputs(node, broker, advisor_output):
    result = await node.execute(broker)
    assert result.status == NodeStatus.SUCCESS
    assert advisor_output in broker.advisor_outputs


@pytest.mark.asyncio
async def test_handler_called_with_broker_fields(node, handler, broker):
    await node.execute(broker)
    handler.advise.assert_called_once_with(
        message=broker.dialogue_input.content,
        evaluated_memories=broker.evaluated_memories,
    )


@pytest.mark.asyncio
async def test_metadata_contains_potency(node, broker, advisor_output):
    result = await node.execute(broker)
    assert result.metadata["potency"] == advisor_output.potency
    assert result.metadata["advisor"] == "memory"


@pytest.mark.asyncio
async def test_multiple_advisors_accumulate(node, broker):
    existing = AdvisorOutput(advisor="other", advice="something", potency=0.5)
    broker.advisor_outputs.append(existing)
    await node.execute(broker)
    assert len(broker.advisor_outputs) == 2


@pytest.mark.asyncio
async def test_missing_dialogue_input_fails(node):
    empty_broker = KnowledgeBroker()
    result = await node.execute(empty_broker)
    assert result.status == NodeStatus.FAILED
