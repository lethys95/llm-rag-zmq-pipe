import pytest

from src.handlers.strategy_advisor import StrategyAdvisorHandler, _POTENCY_FLOOR
from src.models.advisor import AdvisorOutput
from src.models.response_strategy import ResponseStrategy


def make_strategy(
    addition="Validate their grief before anything else. Don't suggest silver linings.",
    approach="reflective_listening",
    tone="empathetic_warm",
):
    return ResponseStrategy(
        approach=approach,
        tone=tone,
        needs_focus=["belonging"],
        system_prompt_addition=addition,
        reasoning="User is in acute grief.",
    )


@pytest.fixture
def handler():
    return StrategyAdvisorHandler()


def test_returns_advisor_output(handler):
    result = handler.advise(make_strategy())
    assert isinstance(result, AdvisorOutput)
    assert result.advisor == "strategy"


def test_advice_is_system_prompt_addition(handler):
    addition = "Keep it warm. Don't push for solutions."
    result = handler.advise(make_strategy(addition=addition))
    assert result.advice == addition


def test_potency_floor_when_low_urgency(handler):
    result = handler.advise(make_strategy(), need_urgency=0.0)
    assert result.potency == pytest.approx(_POTENCY_FLOOR)


def test_potency_floor_when_urgency_below_floor(handler):
    result = handler.advise(make_strategy(), need_urgency=0.1)
    assert result.potency == pytest.approx(_POTENCY_FLOOR)


def test_potency_uses_urgency_when_above_floor(handler):
    result = handler.advise(make_strategy(), need_urgency=0.8)
    assert result.potency == pytest.approx(0.8)


def test_potency_capped_at_one(handler):
    result = handler.advise(make_strategy(), need_urgency=1.0)
    assert result.potency == pytest.approx(1.0)


def test_no_strategy_returns_zero_potency(handler):
    result = handler.advise(None)
    assert result.potency == 0.0
    assert result.advisor == "strategy"


def test_no_llm_dependency():
    # StrategyAdvisorHandler has no LLM — verify it runs without one
    handler = StrategyAdvisorHandler()
    result = handler.advise(make_strategy(), need_urgency=0.5)
    assert result.potency == pytest.approx(0.5)
