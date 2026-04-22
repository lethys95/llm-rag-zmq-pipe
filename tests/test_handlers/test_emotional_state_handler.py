import json
import pytest
from unittest.mock import MagicMock

from src.handlers.emotional_state import EmotionalStateHandler
from src.models.emotional_state import EmotionalState


def _make_response(**overrides) -> str:
    base = {
        "joy": 0.0, "sadness": 0.0, "grief": 0.0, "anger": 0.0,
        "frustration": 0.0, "fear": 0.0, "anxiety": 0.0, "disgust": 0.0,
        "guilt": 0.0, "shame": 0.0, "loneliness": 0.0, "overwhelm": 0.0,
        "contentment": 0.0, "confusion": 0.0,
        "valence": 0.0, "arousal": 0.0, "dominance": 0.5,
        "confidence": 0.8, "summary": "Neutral state.",
    }
    base.update(overrides)
    return json.dumps(base)


GRIEF_RESPONSE = _make_response(
    sadness=0.6, grief=0.85, loneliness=0.5, anxiety=0.2, guilt=0.1,
    valence=-0.7, arousal=0.3, dominance=0.2, confidence=0.85,
    summary="Ongoing grief after losing their mother.",
)


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def handler(mock_llm):
    return EmotionalStateHandler(llm_provider=mock_llm, max_retries=3, retry_delay=0.0)


def test_valid_response_returns_emotional_state(handler, mock_llm):
    mock_llm.generate.return_value = GRIEF_RESPONSE
    result = handler.analyze("I can't stop thinking about my mum.")
    assert isinstance(result, EmotionalState)


def test_all_emotion_fields_present(handler, mock_llm):
    mock_llm.generate.return_value = GRIEF_RESPONSE
    result = handler.analyze("something")
    for field in ["joy", "sadness", "grief", "anger", "frustration", "fear",
                  "anxiety", "disgust", "guilt", "shame", "loneliness",
                  "overwhelm", "contentment", "confusion"]:
        assert hasattr(result, field)


def test_vad_values_correct(handler, mock_llm):
    mock_llm.generate.return_value = GRIEF_RESPONSE
    result = handler.analyze("something")
    assert result.valence == pytest.approx(-0.7)
    assert result.arousal == pytest.approx(0.3)
    assert result.dominance == pytest.approx(0.2)


def test_confidence_populated(handler, mock_llm):
    mock_llm.generate.return_value = GRIEF_RESPONSE
    result = handler.analyze("something")
    assert result.confidence == pytest.approx(0.85)


def test_emotion_scores_correct(handler, mock_llm):
    mock_llm.generate.return_value = GRIEF_RESPONSE
    result = handler.analyze("something")
    assert result.grief == pytest.approx(0.85)
    assert result.loneliness == pytest.approx(0.5)
    assert result.joy == pytest.approx(0.0)


def test_summary_populated(handler, mock_llm):
    mock_llm.generate.return_value = GRIEF_RESPONSE
    result = handler.analyze("something")
    assert "grief" in result.summary.lower()


def test_json_mode_passed_to_llm(handler, mock_llm):
    mock_llm.generate.return_value = GRIEF_RESPONSE
    handler.analyze("something")
    mock_llm.generate.assert_called_once()
    _, kwargs = mock_llm.generate.call_args
    assert kwargs.get("json_mode") is True


def test_missing_required_vad_field_returns_none(handler, mock_llm):
    bad = json.dumps({
        "joy": 0.0, "sadness": 0.5, "grief": 0.0, "anger": 0.0,
        "frustration": 0.0, "fear": 0.0, "anxiety": 0.0, "disgust": 0.0,
        "guilt": 0.0, "shame": 0.0, "loneliness": 0.0, "overwhelm": 0.0,
        "contentment": 0.0, "confusion": 0.0,
        # valence missing
        "arousal": 0.3, "dominance": 0.5, "confidence": 0.7,
    })
    mock_llm.generate.return_value = bad
    result = handler.analyze("something")
    assert result is None


def test_out_of_range_valence_returns_none(handler, mock_llm):
    mock_llm.generate.return_value = _make_response(valence=2.0)
    result = handler.analyze("something")
    assert result is None


def test_out_of_range_emotion_score_returns_none(handler, mock_llm):
    mock_llm.generate.return_value = _make_response(grief=1.5)
    result = handler.analyze("something")
    assert result is None


def test_invalid_json_returns_none(handler, mock_llm):
    mock_llm.generate.return_value = "not json"
    result = handler.analyze("something")
    assert result is None


def test_retries_on_failure_then_succeeds(handler, mock_llm):
    mock_llm.generate.side_effect = ["not json", "not json", GRIEF_RESPONSE]
    result = handler.analyze("something")
    assert result is not None
    assert mock_llm.generate.call_count == 3


def test_exhausted_retries_returns_none(handler, mock_llm):
    mock_llm.generate.return_value = "not json"
    result = handler.analyze("something")
    assert result is None
    assert mock_llm.generate.call_count == 3
