"""Tests for MemoryDecayAlgorithm — pure math, no mocks needed."""

import pytest
from datetime import datetime, timedelta, timezone
from math import isclose

from src.rag.algorithms.memory_chrono_decay import MemoryDecayAlgorithm, calculate_time_decay
from src.rag.selector import RAGDocument


HALF_LIFE = 30.0
NOW = datetime(2026, 4, 21, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def algo():
    return MemoryDecayAlgorithm(
        memory_half_life_days=HALF_LIFE,
        chrono_weight=1.0,
        retrieval_threshold=0.15,
        prune_threshold=0.05,
        max_documents=10,
    )


def make_doc(relevance=1.0, chrono_relevance=0.5, age_days=0.0, score=0.8):
    timestamp = NOW - timedelta(days=age_days)
    return RAGDocument(
        content="test memory",
        score=score,
        metadata={
            "relevance": relevance,
            "chrono_relevance": chrono_relevance,
            "timestamp": timestamp.isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# calculate_time_decay (standalone function)

def test_fresh_memory_has_decay_near_one():
    decay = calculate_time_decay(NOW, NOW, half_life_days=HALF_LIFE)
    assert isclose(decay, 1.0, rel_tol=1e-6)


def test_decay_at_half_life_is_half():
    old = NOW - timedelta(days=HALF_LIFE)
    decay = calculate_time_decay(old, NOW, half_life_days=HALF_LIFE)
    assert isclose(decay, 0.5, rel_tol=1e-6)


def test_future_memory_returns_one():
    future = NOW + timedelta(days=10)
    decay = calculate_time_decay(future, NOW, half_life_days=HALF_LIFE)
    assert decay == 1.0


def test_older_memory_decays_more():
    young = NOW - timedelta(days=10)
    old = NOW - timedelta(days=60)
    assert calculate_time_decay(young, NOW, HALF_LIFE) > calculate_time_decay(old, NOW, HALF_LIFE)


# ---------------------------------------------------------------------------
# MemoryDecayAlgorithm.calculate_memory_score

def test_fresh_high_relevance_scores_near_relevance(algo):
    score = algo.calculate_memory_score(
        relevance=0.8,
        chrono_relevance=0.5,
        timestamp=NOW,
        current_time=NOW,
    )
    assert isclose(score, 0.8, rel_tol=1e-6)


def test_high_chrono_relevance_slows_decay(algo):
    old = NOW - timedelta(days=60)
    high_chrono = algo.calculate_memory_score(1.0, chrono_relevance=0.9, timestamp=old, current_time=NOW)
    low_chrono = algo.calculate_memory_score(1.0, chrono_relevance=0.1, timestamp=old, current_time=NOW)
    assert high_chrono > low_chrono


def test_max_chrono_relevance_prevents_decay(algo):
    # chrono_relevance=1.0 → decay_rate=0.0 → no decay regardless of age
    old = NOW - timedelta(days=365)
    score = algo.calculate_memory_score(0.7, chrono_relevance=1.0, timestamp=old, current_time=NOW)
    assert isclose(score, 0.7, rel_tol=1e-6)


def test_zero_chrono_relevance_decays_at_full_rate(algo):
    old = NOW - timedelta(days=HALF_LIFE)
    score = algo.calculate_memory_score(1.0, chrono_relevance=0.0, timestamp=old, current_time=NOW)
    assert isclose(score, 0.5, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# filter_and_rank

def test_filter_removes_below_threshold(algo):
    docs = [
        make_doc(relevance=0.9, chrono_relevance=0.0, age_days=200),  # very decayed, below threshold
        make_doc(relevance=1.0, chrono_relevance=1.0, age_days=0),    # fresh, above threshold
    ]
    result = algo.filter_and_rank(docs, current_time=NOW)
    assert len(result) == 1
    assert result[0].metadata["chrono_relevance"] == 1.0


def test_filter_returns_sorted_by_score_descending(algo):
    docs = [
        make_doc(relevance=0.3, chrono_relevance=1.0, age_days=0),
        make_doc(relevance=0.9, chrono_relevance=1.0, age_days=0),
        make_doc(relevance=0.6, chrono_relevance=1.0, age_days=0),
    ]
    result = algo.filter_and_rank(docs, current_time=NOW)
    scores = [d.metadata["relevance"] for d in result]
    assert scores == sorted(scores, reverse=True)


def test_filter_respects_max_docs(algo):
    docs = [make_doc(relevance=1.0, chrono_relevance=1.0, age_days=0) for _ in range(20)]
    result = algo.filter_and_rank(docs, max_docs=5, current_time=NOW)
    assert len(result) <= 5


def test_filter_empty_list_returns_empty(algo):
    assert algo.filter_and_rank([], current_time=NOW) == []


# ---------------------------------------------------------------------------
# score_document — metadata fallback

def test_missing_metadata_returns_raw_score(algo):
    doc = RAGDocument(content="no metadata", score=0.42, metadata={})
    score = algo.score_document(doc, current_time=NOW)
    assert score == 0.42


# ---------------------------------------------------------------------------
# identify_prunable

def test_identify_prunable_returns_ids_below_threshold(algo):
    docs = [
        RAGDocument(
            content="very old",
            score=0.1,
            metadata={
                "relevance": 0.9,
                "chrono_relevance": 0.0,
                "timestamp": (NOW - timedelta(days=500)).isoformat(),
                "point_id": "abc123",
            },
        ),
        RAGDocument(
            content="fresh",
            score=0.9,
            metadata={
                "relevance": 1.0,
                "chrono_relevance": 1.0,
                "timestamp": NOW.isoformat(),
                "point_id": "def456",
            },
        ),
    ]
    prunable = algo.identify_prunable(docs, current_time=NOW)
    assert "abc123" in prunable
    assert "def456" not in prunable
