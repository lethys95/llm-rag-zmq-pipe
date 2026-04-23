import json
import pytest
from unittest.mock import MagicMock

from src.handlers.memory_advisor import MemoryAdvisorHandler
from src.models.advisor import AdvisorOutput
from src.models.analysis import MemoryEvaluation
from src.rag.selector import RAGDocument


def make_evaluated(content="user lost their mother", relevance=0.8, reasoning="directly related to grief"):
    doc = RAGDocument(content=content, score=0.9, metadata={})
    evaluation = MemoryEvaluation(
        relevance=relevance,
        chrono_relevance=0.9,
        reasoning=reasoning,
    )
    return (doc, evaluation)


VALID_RESPONSE = json.dumps({
    "advice": "This person has been grieving since their mother died last year. Loneliness has been a consistent theme. They tend to shut down when offered practical advice too early.",
    "potency": 0.85,
})


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def handler(mock_llm):
    return MemoryAdvisorHandler(llm_provider=mock_llm, max_retries=3, retry_delay=0.0)


def test_returns_advisor_output(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.advise("I feel so alone", [make_evaluated()])
    assert isinstance(result, AdvisorOutput)
    assert result.advisor == "memory"


def test_advice_populated(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.advise("I feel so alone", [make_evaluated()])
    assert len(result.advice) > 0
    assert "griev" in result.advice.lower()


def test_potency_populated(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    result = handler.advise("I feel so alone", [make_evaluated()])
    assert result.potency == pytest.approx(0.85)


def test_empty_memories_returns_zero_potency(handler, mock_llm):
    result = handler.advise("something", [])
    assert result.potency == 0.0
    mock_llm.generate.assert_not_called()


def test_memories_included_in_prompt(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    handler.advise("message", [make_evaluated("user lost their mother", reasoning="grief context")])
    prompt = mock_llm.generate.call_args[0][0]
    assert "mother" in prompt
    assert "grief context" in prompt


def test_json_mode_passed_to_llm(handler, mock_llm):
    mock_llm.generate.return_value = VALID_RESPONSE
    handler.advise("message", [make_evaluated()])
    _, kwargs = mock_llm.generate.call_args
    assert kwargs.get("json_mode") is True


def test_invalid_json_returns_fallback(handler, mock_llm):
    mock_llm.generate.return_value = "not json"
    result = handler.advise("message", [make_evaluated()])
    assert result.advisor == "memory"
    assert result.potency == 0.0


def test_retries_on_failure(handler, mock_llm):
    mock_llm.generate.side_effect = ["not json", "not json", VALID_RESPONSE]
    result = handler.advise("message", [make_evaluated()])
    assert result.potency == pytest.approx(0.85)
    assert mock_llm.generate.call_count == 3
