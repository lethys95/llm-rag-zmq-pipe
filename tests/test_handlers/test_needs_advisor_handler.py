import pytest

from src.handlers.needs_advisor import NeedsAdvisorHandler
from src.models.advisor import AdvisorOutput
from src.models.analysis import NeedsAnalysis


def make_needs(urgency=0.7, summary="They are struggling with loneliness and lack of purpose."):
    return NeedsAnalysis(
        belonging=0.8,
        meaning=0.3,
        need_urgency=urgency,
        need_persistence=0.6,
        context_summary=summary,
        primary_needs=["belonging"],
        unmet_needs=["belonging"],
    )


@pytest.fixture
def handler():
    return NeedsAdvisorHandler()


def test_returns_advisor_output(handler):
    result = handler.advise(make_needs())
    assert isinstance(result, AdvisorOutput)
    assert result.advisor == "needs"


def test_advice_is_context_summary(handler):
    summary = "They need connection and validation right now."
    result = handler.advise(make_needs(summary=summary))
    assert result.advice == summary


def test_potency_equals_urgency(handler):
    result = handler.advise(make_needs(urgency=0.65))
    assert result.potency == pytest.approx(0.65)


def test_high_urgency_high_potency(handler):
    result = handler.advise(make_needs(urgency=0.9))
    assert result.potency == pytest.approx(0.9)


def test_zero_urgency_zero_potency(handler):
    result = handler.advise(make_needs(urgency=0.0))
    assert result.potency == pytest.approx(0.0)


def test_no_needs_analysis_returns_zero_potency(handler):
    result = handler.advise(None)
    assert result.potency == 0.0
    assert result.advisor == "needs"


def test_no_llm_dependency():
    # NeedsAdvisorHandler has no LLM — verify it runs without one
    handler = NeedsAdvisorHandler()
    result = handler.advise(make_needs(urgency=0.5))
    assert result.potency == pytest.approx(0.5)
