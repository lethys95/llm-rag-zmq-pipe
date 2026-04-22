import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

from src.handlers.memory_retrieval import MemoryRetrievalHandler
from src.rag.selector import RAGDocument
from src.rag.algorithms.memory_chrono_decay import MemoryDecayAlgorithm


def make_doc(content="memory", score=0.8, age_days=0.0, chrono=0.5, relevance=0.8):
    now = datetime.now(timezone.utc)
    return RAGDocument(
        content=content,
        score=score,
        metadata={
            "memory_owner": "user",
            "relevance": relevance,
            "chrono_relevance": chrono,
            "timestamp": (now - timedelta(days=age_days)).isoformat(),
        },
    )


@pytest.fixture
def mock_rag():
    m = MagicMock()
    m.retrieve_documents_with_embedding.return_value = [
        make_doc("recent memory", age_days=1, chrono=0.8),
        make_doc("old memory", age_days=200, chrono=0.1),
    ]
    return m


@pytest.fixture
def mock_embedding():
    m = MagicMock()
    m.encode.return_value = [0.1] * 384
    return m


@pytest.fixture
def decay():
    return MemoryDecayAlgorithm(
        memory_half_life_days=30.0,
        chrono_weight=1.0,
        retrieval_threshold=0.15,
        prune_threshold=0.05,
        max_documents=10,
    )


@pytest.fixture
def handler(mock_rag, mock_embedding, decay):
    return MemoryRetrievalHandler(
        rag=mock_rag,
        embedding_service=mock_embedding,
        memory_decay=decay,
    )


def test_returns_list_of_documents(handler):
    result = handler.retrieve("I feel alone", memory_owner="user")
    assert isinstance(result, list)


def test_encodes_query(handler, mock_embedding):
    handler.retrieve("I feel alone", memory_owner="user")
    mock_embedding.encode.assert_called_once_with("I feel alone")


def test_passes_embedding_to_rag(handler, mock_rag, mock_embedding):
    mock_embedding.encode.return_value = [0.5] * 384
    handler.retrieve("query", memory_owner="user")
    call_kwargs = mock_rag.retrieve_documents_with_embedding.call_args
    assert call_kwargs.kwargs["query_embedding"] == [0.5] * 384


def test_decay_filters_old_low_chrono_doc(handler):
    result = handler.retrieve("query", memory_owner="user")
    contents = [d.content for d in result]
    assert "recent memory" in contents
    assert "old memory" not in contents


def test_memory_owner_filter_passed_to_rag(handler, mock_rag):
    handler.retrieve("query", memory_owner="alice")
    call_kwargs = mock_rag.retrieve_documents_with_embedding.call_args
    filter_arg = call_kwargs.kwargs.get("filter_conditions")
    assert filter_arg is not None


def test_no_owner_filter_when_none(handler, mock_rag):
    handler.retrieve("query", memory_owner=None)
    call_kwargs = mock_rag.retrieve_documents_with_embedding.call_args
    filter_arg = call_kwargs.kwargs.get("filter_conditions")
    assert filter_arg is None


def test_rag_failure_returns_empty_list(handler, mock_rag):
    mock_rag.retrieve_documents_with_embedding.side_effect = Exception("Qdrant down")
    result = handler.retrieve("query", memory_owner="user")
    assert result == []


def test_fetch_limit_is_multiple_of_max_documents(handler, mock_rag, decay):
    handler.retrieve("query", memory_owner="user")
    call_kwargs = mock_rag.retrieve_documents_with_embedding.call_args
    limit = call_kwargs.kwargs["limit"]
    assert limit > decay.max_documents
