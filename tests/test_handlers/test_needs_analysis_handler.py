import json
import pytest
from unittest.mock import MagicMock

from src.handlers.needs_analysis import NeedsAnalysisHandler
from src.models.analysis import NeedsAnalysis
from src.models.emotional_state import EmotionalState
from src.rag.selector import RAGDocument


def _make_response(**overrides) -> str:
    base = {
        "physiological": 0.0, "safety": 0.0, "belonging": 0.7,
        "esteem": 0.2, "autonomy": 0.0, "meaning": 0.8, "growth": 0.1,
        "primary_needs": ["meaning", "belonging"],
        "unmet_needs": ["meaning", "belonging"],
        "need_urgency": 0.5,
        "need_persistence": 0.8,
        "context_summary": "Struggling with purpose and connection after loss.",
    }
    base.update(overrides)
    return json.dumps(base)


VALID_RESPONSE = _make_response()


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def handler(mock_llm):
    return NeedsAnalysisHandler(llm_provider=mock_llm, max_retries=3, retry_delay=0.0)


@pytest.fixture
def emotional_state():
    return EmotionalState(
        sadness=0.5, loneliness=0.6, confusion=0.4,
        valence=-0.5, arousal=0.3, dominance=0.2,
        confidence=0.8, summary="Sad and disconnected.",
    )


@pytest.fixture
def memories():
    return [
        RAGDocument(
            content="user lost their mother three months ago",
            score=0.9,
            metadata={"subject": "family", "sentiment": "negative"},
        ),
        RAGDocument(
            content="user mentioned feeling purposeless",
            score=0.7,
            metadata={"subject": "wellbeing", "sentiment": "negative"},
        ),
    ]


def test_valid_response_returns_needs_analysis(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.analyze("I don't know what to do with myself", speaker="user")
    assert isinstance(result, NeedsAnalysis)


def test_memory_owner_set_from_speaker(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.analyze("message", speaker="alice")
    assert result.memory_owner == "alice"


def test_need_scores_parsed(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.analyze("message", speaker="user")
    assert result.belonging == pytest.approx(0.7)
    assert result.meaning == pytest.approx(0.8)
    assert result.physiological == pytest.approx(0.0)


def test_primary_needs_parsed(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.analyze("message", speaker="user")
    assert "meaning" in result.primary_needs
    assert "belonging" in result.primary_needs


def test_urgency_and_persistence_parsed(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.analyze("message", speaker="user")
    assert result.need_urgency == pytest.approx(0.5)
    assert result.need_persistence == pytest.approx(0.8)


def test_context_summary_populated(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.analyze("message", speaker="user")
    assert len(result.context_summary) > 0


def test_emotional_state_included_in_prompt(handler, mock_llm, emotional_state):
    mock_llm.generate.return_value = VALID_RESPONSE
    handler.analyze("message", speaker="user", emotional_state=emotional_state)
    prompt = mock_llm.generate.call_args[0][0]
    assert "loneliness" in prompt
    assert "valence" in prompt.lower()


def test_memories_included_in_prompt(handler, mock_llm, memories):
    mock_llm.generate.return_value = VALID_RESPONSE
    handler.analyze("message", speaker="user", retrieved_documents=memories)
    prompt = mock_llm.generate.call_args[0][0]
    assert "mother" in prompt


def test_json_mode_passed_to_llm(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    handler.analyze("message", speaker="user")
    _, kwargs = mock_llm.generate.call_args
    assert kwargs.get("json_mode") is True


def test_invalid_json_returns_none(handler, mock_llm):
    mock_llm.generate.return_value = "not json"
    result = handler.analyze("message", speaker="user")
    assert result is None


def test_unknown_need_name_returns_none(handler, mock_llm):
    bad = _make_response(primary_needs=["nonexistent_need"])
    mock_llm.generate.return_value = bad
    result = handler.analyze("message", speaker="user")
    assert result is None


def test_retries_on_failure(handler, mock_llm):
    mock_llm.generate.side_effect = ["not json", "not json", VALID_RESPONSE]
    result = handler.analyze("message", speaker="user")
    assert result is not None
    assert mock_llm.generate.call_count == 3


def test_exhausted_retries_returns_none(handler, mock_llm):
    mock_llm.generate.return_value = "not json"
    result = handler.analyze("message", speaker="user")
    assert result is None
