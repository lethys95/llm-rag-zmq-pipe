import json
import pytest
from unittest.mock import MagicMock

from src.handlers.memory_evaluation import MemoryEvaluationHandler
from src.models.analysis import MemoryEvaluation
from src.rag.selector import RAGDocument


def make_doc(content: str) -> RAGDocument:
    return RAGDocument(content=content, score=0.8, metadata={})


def make_response(*entries) -> str:
    return json.dumps(list(entries))


DOCS = [
    make_doc("user lost their mother last year"),
    make_doc("user enjoys hiking on weekends"),
]

VALID_RESPONSE = make_response(
    {"index": 0, "relevance": 0.9, "chrono_relevance": 0.95, "reasoning": "Grief is directly relevant to their current loneliness."},
    {"index": 1, "relevance": 0.1, "chrono_relevance": 0.3, "reasoning": "Unrelated to the current emotional context."},
)


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def handler(mock_llm):
    return MemoryEvaluationHandler(llm_provider=mock_llm, max_retries=3, retry_delay=0.0)


def test_returns_tuple_per_document(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.evaluate("I feel so alone", DOCS)
    assert len(result) == 2
    assert all(isinstance(doc, RAGDocument) and isinstance(ev, MemoryEvaluation) for doc, ev in result)


def test_documents_paired_correctly(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.evaluate("I feel so alone", DOCS)
    doc0, ev0 = result[0]
    assert doc0.content == "user lost their mother last year"
    assert ev0.relevance == pytest.approx(0.9)
    assert ev0.chrono_relevance == pytest.approx(0.95)
    assert "grief" in ev0.reasoning.lower()


def test_reasoning_populated(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.evaluate("I feel so alone", DOCS)
    for _, ev in result:
        assert len(ev.reasoning) > 0


def test_json_mode_passed_to_llm(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    handler.evaluate("I feel so alone", DOCS)
    _, kwargs = mock_llm.generate.call_args
    assert kwargs.get("json_mode") is True


def test_empty_documents_returns_empty(handler, mock_llm):
    result = handler.evaluate("I feel so alone", [])
    assert result == []
    mock_llm.generate.assert_not_called()


def test_invalid_json_returns_empty(handler, mock_llm):
    mock_llm.generate.return_value = "not json"
    result = handler.evaluate("I feel so alone", DOCS)
    assert result == []


def test_retries_on_failure_then_succeeds(handler, mock_llm):
    mock_llm.generate.side_effect = ["not json", "not json", VALID_RESPONSE]
    result = handler.evaluate("I feel so alone", DOCS)
    assert len(result) == 2
    assert mock_llm.generate.call_count == 3


def test_exhausted_retries_returns_empty(handler, mock_llm):
    mock_llm.generate.return_value = "not json"
    result = handler.evaluate("I feel so alone", DOCS)
    assert result == []
    assert mock_llm.generate.call_count == 3


def test_out_of_range_index_skipped(handler, mock_llm):
    bad_response = make_response(
        {"index": 99, "relevance": 0.8, "chrono_relevance": 0.5, "reasoning": "oob"},
        {"index": 0, "relevance": 0.7, "chrono_relevance": 0.6, "reasoning": "valid"},
    )
    mock_llm.generate.return_value = bad_response
    result = handler.evaluate("test", DOCS)
    assert len(result) == 1
    assert result[0][0].content == DOCS[0].content


def test_message_and_memories_in_prompt(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    handler.evaluate("I feel so alone", DOCS)
    prompt = mock_llm.generate.call_args[0][0]
    assert "I feel so alone" in prompt
    assert "user lost their mother" in prompt
    assert "user enjoys hiking" in prompt
