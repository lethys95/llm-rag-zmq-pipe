"""Unit tests for RAG memory decay algorithms."""

from datetime import datetime, timedelta

import pytest

from src.rag.algorithms import (
    calculate_time_decay,
    calculate_access_boost,
    calculate_memory_score,
)


@pytest.mark.unit
class TestTimeDecay:
    """Tests for time-based decay calculation."""
    
    def test_no_decay_at_creation(self):
        """Test that memory has full score at creation time."""
        created_at = datetime.now()
        current_time = created_at
        half_life_days = 30.0
        
        decay = calculate_time_decay(created_at, current_time, half_life_days)
        
        assert decay == 1.0
        
    def test_half_decay_at_half_life(self):
        """Test that memory has 0.5 score after one half-life."""
        created_at = datetime.now()
        current_time = created_at + timedelta(days=30)
        half_life_days = 30.0
        
        decay = calculate_time_decay(created_at, current_time, half_life_days)
        
        assert abs(decay - 0.5) < 0.01
        
    def test_quarter_decay_at_two_half_lives(self):
        """Test that memory has 0.25 score after two half-lives."""
        created_at = datetime.now()
        current_time = created_at + timedelta(days=60)
        half_life_days = 30.0
        
        decay = calculate_time_decay(created_at, current_time, half_life_days)
        
        assert abs(decay - 0.25) < 0.01
        
    def test_very_old_memory_approaches_zero(self):
        """Test that very old memories have near-zero score."""
        created_at = datetime.now()
        current_time = created_at + timedelta(days=300)
        half_life_days = 30.0
        
        decay = calculate_time_decay(created_at, current_time, half_life_days)
        
        assert decay < 0.001
        
    def test_different_half_life(self):
        """Test decay calculation with different half-life value."""
        created_at = datetime.now()
        current_time = created_at + timedelta(days=60)
        half_life_days = 60.0
        
        decay = calculate_time_decay(created_at, current_time, half_life_days)
        
        # After one half-life (60 days), should be 0.5
        assert abs(decay - 0.5) < 0.01
        
    def test_negative_time_delta(self):
        """Test that negative time delta (future creation) returns 1.0."""
        created_at = datetime.now() + timedelta(days=10)
        current_time = datetime.now()
        half_life_days = 30.0
        
        decay = calculate_time_decay(created_at, current_time, half_life_days)
        
        # Future memories should have full score
        assert decay == 1.0


@pytest.mark.unit
class TestAccessBoost:
    """Tests for access-based boost calculation."""
    
    def test_no_boost_for_unaccessed_memory(self):
        """Test that memories never accessed get no boost."""
        boost = calculate_access_boost(access_count=0, retrieval_count=0)
        
        assert boost == 0.0
        
    def test_boost_increases_with_access_count(self):
        """Test that boost increases with more accesses."""
        boost_1 = calculate_access_boost(access_count=1, retrieval_count=10)
        boost_5 = calculate_access_boost(access_count=5, retrieval_count=10)
        boost_10 = calculate_access_boost(access_count=10, retrieval_count=10)
        
        assert boost_1 < boost_5 < boost_10
        
    def test_boost_increases_with_retrieval_count(self):
        """Test that boost increases with more retrievals."""
        boost_low = calculate_access_boost(access_count=5, retrieval_count=5)
        boost_high = calculate_access_boost(access_count=5, retrieval_count=20)
        
        assert boost_low < boost_high
        
    def test_boost_saturates(self):
        """Test that boost approaches but doesn't exceed limit."""
        boost = calculate_access_boost(access_count=100, retrieval_count=100)
        
        # Boost should be high but not exceed reasonable bounds
        assert 0.0 < boost < 10.0
        
    def test_boost_with_zero_retrievals(self):
        """Test boost calculation with zero retrievals (edge case)."""
        boost = calculate_access_boost(access_count=5, retrieval_count=0)
        
        # Should still have some boost based on access count
        assert boost > 0.0


@pytest.mark.unit
class TestMemoryScore:
    """Tests for combined memory score calculation."""
    
    def test_fresh_unaccessed_memory(self):
        """Test score for a fresh, never-accessed memory."""
        created_at = datetime.now()
        current_time = created_at
        half_life_days = 30.0
        access_count = 0
        retrieval_count = 0
        chrono_weight = 0.7
        
        score = calculate_memory_score(
            created_at=created_at,
            current_time=current_time,
            half_life_days=half_life_days,
            access_count=access_count,
            retrieval_count=retrieval_count,
            chrono_weight=chrono_weight,
        )
        
        # Fresh memory with no accesses: only chronological component
        assert 0.0 < score <= 1.0
        
    def test_old_frequently_accessed_memory(self):
        """Test that frequent access can offset age decay."""
        created_at = datetime.now() - timedelta(days=60)
        current_time = datetime.now()
        half_life_days = 30.0
        access_count = 20
        retrieval_count = 50
        chrono_weight = 0.7
        
        score = calculate_memory_score(
            created_at=created_at,
            current_time=current_time,
            half_life_days=half_life_days,
            access_count=access_count,
            retrieval_count=retrieval_count,
            chrono_weight=chrono_weight,
        )
        
        # Despite being old, frequent access should boost score
        assert score > 0.1
        
    def test_chrono_weight_effect(self):
        """Test that chrono_weight affects score calculation."""
        created_at = datetime.now() - timedelta(days=30)
        current_time = datetime.now()
        half_life_days = 30.0
        access_count = 5
        retrieval_count = 10
        
        score_high_chrono = calculate_memory_score(
            created_at=created_at,
            current_time=current_time,
            half_life_days=half_life_days,
            access_count=access_count,
            retrieval_count=retrieval_count,
            chrono_weight=0.9,
        )
        
        score_low_chrono = calculate_memory_score(
            created_at=created_at,
            current_time=current_time,
            half_life_days=half_life_days,
            access_count=access_count,
            retrieval_count=retrieval_count,
            chrono_weight=0.3,
        )
        
        # Higher chrono weight means access boost matters less
        # Since time decay is 0.5 and access helps, low chrono should be higher
        assert score_low_chrono > score_high_chrono
        
    def test_score_bounds(self):
        """Test that memory score is always within reasonable bounds."""
        created_at = datetime.now() - timedelta(days=15)
        current_time = datetime.now()
        half_life_days = 30.0
        access_count = 10
        retrieval_count = 20
        chrono_weight = 0.7
        
        score = calculate_memory_score(
            created_at=created_at,
            current_time=current_time,
            half_life_days=half_life_days,
            access_count=access_count,
            retrieval_count=retrieval_count,
            chrono_weight=chrono_weight,
        )
        
        assert 0.0 <= score <= 10.0  # Reasonable upper bound
        
    def test_equal_weight_distribution(self):
        """Test score calculation with equal chrono/access weights."""
        created_at = datetime.now() - timedelta(days=30)
        current_time = datetime.now()
        half_life_days = 30.0
        access_count = 5
        retrieval_count = 10
        chrono_weight = 0.5
        
        score = calculate_memory_score(
            created_at=created_at,
            current_time=current_time,
            half_life_days=half_life_days,
            access_count=access_count,
            retrieval_count=retrieval_count,
            chrono_weight=chrono_weight,
        )
        
        # Both components should contribute equally
        assert score > 0.0
