import pytest
from unittest.mock import MagicMock

from src.nodes.algo_nodes.memory_evaluation_node import MemoryEvaluationNode
from src.nodes.core.result import NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.models.analysis import MemoryEvaluation
from src.rag.selector import RAGDocument


@pytest.fixture
def docs():
    return [
        RAGDocument(content="user lost their mother last year", score=0.9, metadata={}),
        RAGDocument(content="user enjoys hiking", score=0.6, metadata={}),
    ]


@pytest.fixture
def evaluations(docs):
    return [
        (docs[0], MemoryEvaluation(relevance=0.9, chrono_relevance=0.95, reasoning="Grief context.")),
        (docs[1], MemoryEvaluation(relevance=0.1, chrono_relevance=0.3, reasoning="Unrelated.")),
    ]


@pytest.fixture
def handler(evaluations):
    m = MagicMock()
    m.evaluate.return_value = evaluations
    return m


@pytest.fixture
def node(handler):
    return MemoryEvaluationNode(memory_evaluation_handler=handler)


@pytest.mark.asyncio
async def test_success_writes_to_broker(node, broker, docs, evaluations):
    broker.retrieved_documents = docs
    result = await node.execute(broker)
    assert result.status == NodeStatus.SUCCESS
    assert broker.evaluated_memories is evaluations


@pytest.mark.asyncio
async def test_handler_called_with_broker_fields(node, handler, broker, docs):
    broker.retrieved_documents = docs
    await node.execute(broker)
    handler.evaluate.assert_called_once_with(
        message=broker.dialogue_input.content,
        documents=docs,
        emotional_state=None,
    )


@pytest.mark.asyncio
async def test_metadata_contains_count(node, broker, docs, evaluations):
    broker.retrieved_documents = docs
    result = await node.execute(broker)
    assert result.metadata["evaluated"] == len(evaluations)


@pytest.mark.asyncio
async def test_empty_retrieved_documents_skips_handler(node, handler, broker):
    broker.retrieved_documents = []
    result = await node.execute(broker)
    assert result.status == NodeStatus.SUCCESS
    assert broker.evaluated_memories == []
    handler.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_missing_dialogue_input_fails(node):
    empty_broker = KnowledgeBroker()
    result = await node.execute(empty_broker)
    assert result.status == NodeStatus.FAILED
