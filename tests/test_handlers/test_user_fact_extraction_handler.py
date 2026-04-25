import json
import pytest
from unittest.mock import MagicMock

from src.handlers.user_fact_extraction import UserFactExtractionHandler
from src.models.user_fact import UserFact


VALID_RESPONSE = json.dumps({"facts": [
    {
        "claim": "user likes pepperoni pizza",
        "chrono_relevance": 0.85,
        "subject": "food preferences",
    },
    {
        "claim": "user dislikes pineapple on pizza",
        "chrono_relevance": 0.8,
        "subject": "food preferences",
    },
]})

EMPTY_RESPONSE = json.dumps({"facts": []})


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def handler(mock_llm):
    return UserFactExtractionHandler(
        llm_provider=mock_llm,
        max_retries=3,
        retry_delay=0.0,
    )


def test_valid_response_returns_user_facts(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.extract("I love pepperoni, hate pineapple", speaker="user")
    assert len(result) == 2
    assert all(isinstance(f, UserFact) for f in result)


def test_memory_owner_set_from_speaker(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.extract("message", speaker="alice")
    assert all(f.memory_owner == "alice" for f in result)


def test_fact_fields_correct(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.extract("message", speaker="user")
    first = result[0]
    assert first.claim == "user likes pepperoni pizza"
    assert first.chrono_relevance == pytest.approx(0.85)
    assert first.subject == "food preferences"
    assert first.valence is None
    assert first.arousal is None
    assert first.dominance is None


def test_vad_stamped_from_emotional_state(handler, mock_llm):
    from src.models.emotional_state import EmotionalState
    mock_llm.generate.return_value = VALID_RESPONSE
    state = EmotionalState(valence=-0.6, arousal=0.7, dominance=0.3, confidence=0.8)
    result = handler.extract("message", speaker="user", emotional_state=state)
    assert result[0].valence == pytest.approx(-0.6)
    assert result[0].arousal == pytest.approx(0.7)
    assert result[0].dominance == pytest.approx(0.3)


def test_empty_array_returns_empty_list(handler, mock_llm):
    mock_llm.generate.return_value = EMPTY_RESPONSE
    result = handler.extract("neutral message", speaker="user")
    assert result == []


def test_invalid_json_returns_empty_list(handler, mock_llm):
    mock_llm.generate.return_value = "not json"
    result = handler.extract("message", speaker="user")
    assert result == []


def test_missing_required_fields_returns_empty_list(handler, mock_llm):
    bad = json.dumps({"facts": [{
        "claim": "user likes something",
        # missing chrono_relevance and subject
    }]})
    mock_llm.generate.return_value = bad
    result = handler.extract("message", speaker="user")
    assert result == []


def test_retries_on_failure_then_succeeds(handler, mock_llm):
    mock_llm.generate.side_effect = ["not json", "not json", VALID_RESPONSE]
    result = handler.extract("message", speaker="user")
    assert len(result) == 2
    assert mock_llm.generate.call_count == 3


def test_exhausted_retries_returns_empty_list(handler, mock_llm):
    mock_llm.generate.return_value = "not json"
    result = handler.extract("message", speaker="user")
    assert result == []
    assert mock_llm.generate.call_count == 3
