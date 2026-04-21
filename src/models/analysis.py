"""Analysis models for AI-driven evaluation and assessment."""

from dataclasses import dataclass


@dataclass
class MemoryEvaluation:
    """Result of AI memory evaluation."""

    relevance: float
    chrono_relevance: float
    reasoning: str
    should_boost: bool
    boost_factor: float


@dataclass
class NeedsAnalysis:
    """Analysis of user's psychological needs using Maslow's hierarchy.

    Memory Owner: user
    """

    memory_owner: str

    # Maslow's hierarchy scores (0.0-1.0)
    physiological: float
    safety: float
    belonging: float
    esteem: float
    autonomy: float
    meaning: float
    growth: float

    # Derived fields
    primary_needs: list[str]
    unmet_needs: list[str]
    need_urgency: float
    need_persistence: float
    context_summary: str
    suggested_approach: str
