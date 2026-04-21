"""Tests for SentimentAnalysisHandler structured output contract."""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.handlers.sentiment_analysis import SentimentAnalysisHandler
from src.models.sentiment import SentimentAnalysis


VALID_RESPONSE = json.dumps({
    "sentiment": "negative",
    "confidence": 0.95,
    "emotional_tone": "grieving",
    "relevance": 1.0,
    "chrono_relevance": 0.95,
    "context_summary": "User's mother died",
    "key_topics": ["grief", "family", "death"],
})

MINIMAL_VALID_RESPONSE = json.dumps({
    "sentiment": "neutral",
    "confidence": 0.6,
})


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.store.return_value = "point-id-123"
    return rag


@pytest.fixture
def handler(mock_llm, mock_rag):
    embedding_service = MagicMock()
    embedding_service.encode.return_value = [0.1] * 384
    return SentimentAnalysisHandler(
        llm_provider=mock_llm,
        rag_provider=mock_rag,
        max_retries=3,
        retry_delay=0.0,
        embedding_service=embedding_service,
    )


def test_valid_response_returns_sentiment_analysis(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.analyze("My mother died yesterday", speaker="user")
    assert isinstance(result, SentimentAnalysis)


def test_correct_sentiment_value(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.analyze("My mother died yesterday", speaker="user")
    assert result.sentiment == "negative"


def test_memory_owner_set_from_speaker(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.analyze("My mother died yesterday", speaker="alice")
    assert result.memory_owner == "alice"


def test_optional_fields_populated(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.analyze("My mother died yesterday", speaker="user")
    assert result.emotional_tone == "grieving"
    assert result.chrono_relevance == 0.95
    assert "grief" in result.key_topics


def test_minimal_valid_response_succeeds(handler, mock_llm):
    mock_llm.generate.return_value = MINIMAL_VALID_RESPONSE
    result = handler.analyze("okay", speaker="user")
    assert isinstance(result, SentimentAnalysis)
    assert result.sentiment == "neutral"


def test_invalid_json_returns_none(handler, mock_llm):
    mock_llm.generate.return_value = "this is not json at all"
    result = handler.analyze("something", speaker="user")
    assert result is None


def test_wrong_sentiment_value_returns_none(handler, mock_llm):
    bad = json.dumps({"sentiment": "confused", "confidence": 0.8})
    mock_llm.generate.return_value = bad
    result = handler.analyze("something", speaker="user")
    assert result is None


def test_confidence_out_of_range_returns_none(handler, mock_llm):
    bad = json.dumps({"sentiment": "positive", "confidence": 1.5})
    mock_llm.generate.return_value = bad
    result = handler.analyze("something", speaker="user")
    assert result is None


def test_retries_on_failure_then_succeeds(handler, mock_llm):
    mock_llm.generate.side_effect = [
        "not json",
        "not json",
        VALID_RESPONSE,
    ]
    result = handler.analyze("something", speaker="user")
    assert result is not None
    assert mock_llm.generate.call_count == 3


def test_exhausted_retries_returns_none(handler, mock_llm):
    mock_llm.generate.return_value = "not json"
    result = handler.analyze("something", speaker="user")
    assert result is None
    assert mock_llm.generate.call_count == 3


def test_rag_store_called_on_success(handler, mock_llm, mock_rag):
    mock_llm.generate.return_value = VALID_RESPONSE
    handler.analyze("My mother died yesterday", speaker="user")
    mock_rag.store.assert_called_once()


def test_rag_store_not_called_on_failure(handler, mock_llm, mock_rag):
    mock_llm.generate.return_value = "not json"
    handler.analyze("something", speaker="user")
    mock_rag.store.assert_not_called()
