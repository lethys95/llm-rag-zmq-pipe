import pytest
from unittest.mock import MagicMock

from src.nodes.algo_nodes.memory_retrieval_node import MemoryRetrievalNode
from src.nodes.core.result import NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.rag.selector import RAGDocument


@pytest.fixture
def docs():
    return [
        RAGDocument(content="user likes pizza", score=0.9, metadata={}),
        RAGDocument(content="user felt lonely last week", score=0.7, metadata={}),
    ]


@pytest.fixture
def handler(docs):
    m = MagicMock()
    m.retrieve.return_value = docs
    return m


@pytest.fixture
def node(handler):
    return MemoryRetrievalNode(memory_retrieval_handler=handler)


@pytest.mark.asyncio
async def test_success_writes_to_broker(node, broker, docs):
    result = await node.execute(broker)
    assert result.status == NodeStatus.SUCCESS
    assert broker.retrieved_documents == docs


@pytest.mark.asyncio
async def test_handler_called_with_correct_args(node, handler, broker):
    await node.execute(broker)
    handler.retrieve.assert_called_once_with(
        query=broker.dialogue_input.content,
        memory_owner=broker.dialogue_input.speaker,
    )


@pytest.mark.asyncio
async def test_empty_result_still_succeeds(node, handler, broker):
    handler.retrieve.return_value = []
    result = await node.execute(broker)
    assert result.status == NodeStatus.SUCCESS
    assert broker.retrieved_documents == []


@pytest.mark.asyncio
async def test_result_metadata_contains_count(node, broker, docs):
    result = await node.execute(broker)
    assert result.metadata["retrieved"] == len(docs)


@pytest.mark.asyncio
async def test_missing_dialogue_input_fails(node):
    empty_broker = KnowledgeBroker()
    result = await node.execute(empty_broker)
    assert result.status == NodeStatus.FAILED
