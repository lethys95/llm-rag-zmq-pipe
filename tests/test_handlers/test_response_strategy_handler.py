import json
import pytest
from unittest.mock import MagicMock

from src.handlers.response_strategy import ResponseStrategyHandler
from src.models.response_strategy import ResponseStrategy
from src.models.analysis import NeedsAnalysis
from src.models.emotional_state import EmotionalState


def _make_response(**overrides) -> str:
    base = {
        "approach": "reflective_listening",
        "tone": "empathetic_warm",
        "needs_focus": ["belonging", "meaning"],
        "system_prompt_addition": "Validate their grief. Don't suggest fixes. Ask what they miss most.",
        "reasoning": "High belonging and meaning needs with elevated persistence suggests chronic grief pattern.",
    }
    base.update(overrides)
    return json.dumps(base)


VALID_RESPONSE = _make_response()


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def handler(mock_llm, worker_call):
    return ResponseStrategyHandler(worker_llm=mock_llm, worker_call=worker_call)


@pytest.fixture
def needs_analysis():
    return NeedsAnalysis(
        belonging=0.7, meaning=0.8,
        primary_needs=["meaning", "belonging"],
        unmet_needs=["meaning", "belonging"],
        need_urgency=0.5, need_persistence=0.8,
        context_summary="Struggling with purpose and connection.",
    )


@pytest.fixture
def emotional_state():
    return EmotionalState(
        sadness=0.6, grief=0.7,
        valence=-0.6, arousal=0.3, dominance=0.2,
        confidence=0.85,
    )


def test_valid_response_returns_strategy(handler, mock_llm, needs_analysis):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.select(needs_analysis=needs_analysis, emotional_state=None)
    assert isinstance(result, ResponseStrategy)


def test_approach_parsed(handler, mock_llm, needs_analysis):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.select(needs_analysis=needs_analysis, emotional_state=None)
    assert result.approach == "reflective_listening"


def test_system_prompt_addition_populated(handler, mock_llm, needs_analysis):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.select(needs_analysis=needs_analysis, emotional_state=None)
    assert len(result.system_prompt_addition) > 0


def test_reasoning_populated(handler, mock_llm, needs_analysis):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.select(needs_analysis=needs_analysis, emotional_state=None)
    assert len(result.reasoning) > 0


def test_needs_included_in_prompt(handler, mock_llm, needs_analysis):
    mock_llm.generate.return_value = VALID_RESPONSE
    handler.select(needs_analysis=needs_analysis, emotional_state=None)
    prompt = mock_llm.generate.call_args[0][0]
    assert "meaning" in prompt
    assert "belonging" in prompt


def test_emotional_state_included_in_prompt(handler, mock_llm, needs_analysis, emotional_state):
    mock_llm.generate.return_value = VALID_RESPONSE
    handler.select(needs_analysis=needs_analysis, emotional_state=emotional_state)
    prompt = mock_llm.generate.call_args[0][0]
    assert "grief" in prompt


def test_no_inputs_returns_none(handler, mock_llm):
    result = handler.select(needs_analysis=None, emotional_state=None)
    assert result is None
    mock_llm.generate.assert_not_called()


def test_invalid_approach_returns_none(handler, mock_llm, needs_analysis):
    mock_llm.generate.return_value = _make_response(approach="made_up_therapy")
    result = handler.select(needs_analysis=needs_analysis, emotional_state=None)
    assert result is None


def test_invalid_tone_returns_none(handler, mock_llm, needs_analysis):
    mock_llm.generate.return_value = _make_response(tone="aggressive")
    result = handler.select(needs_analysis=needs_analysis, emotional_state=None)
    assert result is None


def test_json_mode_passed_to_llm(handler, mock_llm, needs_analysis):
    mock_llm.generate.return_value = VALID_RESPONSE
    handler.select(needs_analysis=needs_analysis, emotional_state=None)
    _, kwargs = mock_llm.generate.call_args
    assert kwargs.get("json_mode") is True


def test_retries_on_failure(handler, mock_llm, needs_analysis):
    mock_llm.generate.side_effect = ["not json", "not json", VALID_RESPONSE]
    result = handler.select(needs_analysis=needs_analysis, emotional_state=None)
    assert result is not None
    assert mock_llm.generate.call_count == 3


def test_exhausted_retries_returns_none(handler, mock_llm, needs_analysis):
    mock_llm.generate.return_value = "not json"
    result = handler.select(needs_analysis=needs_analysis, emotional_state=None)
    assert result is None
