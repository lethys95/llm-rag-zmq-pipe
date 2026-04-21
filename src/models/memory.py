"""Data models for memory management."""

from dataclasses import dataclass, field
from datetime import datetime

#  This entire file needs to be severely re-evaluated.

@dataclass
class MemoryMetadata:
    """Metadata for a stored memory."""

    timestamp: datetime
    memory_owner: str
    sentiment: str
    confidence: float
    emotional_tone: str | None = None
    relevance: float = 0.5
    chrono_relevance: float = 0.5
    context_summary: str | None = None
    key_topics: list[str] = field(default_factory=list[str])
    access_count: int = 0
    last_accessed: datetime | None = None
    memory_score: float = 0.0
    is_consolidated: bool = False
    consolidated_with: list[str] = field(default_factory=list[str])


@dataclass
class ConversationState:
    """Current conversation state for memory evaluation."""

    message_count: int = 0
    recent_topics: list[str] = field(default_factory=list[str])
    emotional_tone: str | None = None
    trust_score: float = 0.0


@dataclass
class TrustAnalysis:
    """Result of trust analysis."""

    score: float
    relationship_age_days: int
    interaction_count: int
    positive_interactions: int
    negative_interactions: int
    consistency_score: float
    reasoning: str


@dataclass
class TrustRecord:
    """Record of trust-related interaction."""

    timestamp: datetime
    user_id: str
    interaction_type: str
    depth_score: float
    consistency_score: float
