# Implementation Guide: Memory Decay & Detox Protocol Integration

## Overview

This document provides a step-by-step guide for completing the integration of the memory decay algorithm and detox protocol into the main pipeline. Use this as a reference for future implementation sessions.

---

## Current Status

### ✅ Completed

1. **Analysis Document**: [`docs/ANALYSIS_MEMORY_DECAY_AND_DETOX.md`](docs/ANALYSIS_MEMORY_DECAY_AND_DETOX.md)
   - Comprehensive analysis of memory decay concept validity
   - Implementation correctness review
   - Proposed system design

2. **Core Algorithm Files**:
   - [`src/rag/algorithms/memory_chrono_decay.py`](src/rag/algorithms/memory_chrono_decay.py) - Memory decay algorithm (existing, needs integration)
   - [`src/rag/algorithms/nudging_algorithm.py`](src/rag/algorithms/nudging_algorithm.py) - Nudging algorithm (NEW, styleguide compliant)

3. **Algo Nodes** (in [`src/nodes/algo_nodes/`](src/nodes/algo_nodes/)):
   - [`memory_evaluator_node.py`](src/nodes/algo_nodes/memory_evaluator_node.py) - AI-driven memory re-evaluation
   - [`memory_consolidation_node.py`](src/nodes/algo_nodes/memory_consolidation_node.py) - Memory merging and consolidation
   - [`detox_scheduler.py`](src/nodes/algo_nodes/detox_scheduler.py) - Detox session scheduling and orchestration

### ⚠️ Needs Attention

1. **Styleguide Violations**: Some files may still have minor violations - review and fix
2. **Integration**: Nodes are created but not integrated into main pipeline
3. **Trust Analysis**: Not implemented yet (critical dependency)
4. **Testing**: No tests written for new components

---

## Implementation Roadmap

### Phase 1: Fix Styleguide Violations & Core Infrastructure

#### 1.1 Review and Fix All New Files

**Files to review**:
- [`src/nodes/algo_nodes/memory_evaluator_node.py`](src/nodes/algo_nodes/memory_evaluator_node.py)
- [`src/nodes/algo_nodes/memory_consolidation_node.py`](src/nodes/algo_nodes/memory_consolidation_node.py)
- [`src/nodes/algo_nodes/detox_scheduler.py`](src/nodes/algo_nodes/detox_scheduler.py)

**Styleguide checklist**:
- [ ] No `dict[str, Any]` - replace with dataclasses
- [ ] No `Any` type annotations
- [ ] No imports inside functions (unless optional dependency)
- [ ] No string type annotations
- [ ] Use absolute imports
- [ ] Use union types (`str | None`) not `Optional[str]`

#### 1.2 Create Missing Dataclasses

Create [`src/models/memory.py`](src/models/memory.py) for memory-related models:

```python
"""Data models for memory management."""

from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field


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
    key_topics: list[str] = field(default_factory=list)
    access_count: int = 0
    last_accessed: datetime | None = None
    memory_score: float = 0.0
    is_consolidated: bool = False
    consolidated_with: list[str] = field(default_factory=list)


@dataclass
class ConversationState:
    """Current conversation state for memory evaluation."""
    
    message_count: int = 0
    recent_topics: list[str] = field(default_factory=list)
    emotional_tone: str | None = None
    trust_score: float = 0.0
```

#### 1.3 Create Trust Analysis System

Create [`src/nodes/algo_nodes/trust_analysis_node.py`](src/nodes/algo_nodes/trust_analysis_node.py):

```python
"""Trust analysis node for relationship maturity tracking."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.nodes.core.base import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker

logger = logging.getLogger(__name__)


@dataclass
class TrustAnalysis:
    """Result of trust analysis."""
    
    score: float  # 0.0-1.0
    relationship_age_days: int
    interaction_count: int
    positive_interactions: int
    negative_interactions: int
    consistency_score: float
    reasoning: str


class TrustAnalysisNode(BaseNode):
    """Analyzes and tracks relationship trust over time.
    
    Trust score is calculated based on:
    - Relationship age (time since first interaction)
    - Interaction frequency and count
    - Positive vs negative interaction ratio
    - Consistency of interactions
    - Depth of shared information
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="trust_analysis",
            priority=1,
            queue_type="immediate",
            **kwargs
        )
    
    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        """Calculate trust score for current user."""
        # Implementation here
        pass
```

---

### Phase 2: Integrate Memory Decay into Pipeline

#### 2.1 Enhance SentimentAnalysisNode

**File**: [`src/nodes/processing/sentiment_analysis_node.py`](src/nodes/processing/sentiment_analysis_node.py)

**Current state**: Only analyzes sentiment and stores in RAG

**Needed changes**:
1. After sentiment analysis, retrieve relevant memories
2. Apply memory decay filtering
3. Track access counts for retrieved documents
4. Pass filtered memories to broker

**Implementation**:

```python
async def execute(self, broker: KnowledgeBroker) -> NodeResult:
    dialogue_input = broker.dialogue_input
    if not dialogue_input:
        return NodeResult(status=NodeStatus.FAILED, error="No dialogue_input")
    
    # 1. Analyze sentiment
    sentiment = self.handler.analyze(
        dialogue_input.content,
        dialogue_input.speaker
    )
    
    if not sentiment:
        return NodeResult(status=NodeStatus.SKIPPED)
    
    # 2. Retrieve relevant memories
    from src.rag.embeddings import EmbeddingService
    from src.rag.algorithms import MemoryDecayAlgorithm
    
    embedding_service = EmbeddingService.get_instance()
    query_embedding = embedding_service.encode(dialogue_input.content)
    
    raw_docs = self.rag.retrieve_documents(
        query_embedding=query_embedding,
        limit=100
    )
    
    # 3. Apply memory decay filtering
    memory_algo = MemoryDecayAlgorithm(
        memory_half_life_days=self.config.memory_half_life_days,
        chrono_weight=self.config.chrono_weight
    )
    
    filtered_docs = memory_algo.filter_and_rank(
        documents=raw_docs,
        threshold=self.config.memory_retrieval_threshold,
        max_docs=self.config.max_context_documents
    )
    
    # 4. Update access counts (TODO: implement in RAG)
    await self._update_access_counts(filtered_docs)
    
    # 5. Store results in broker
    broker.sentiment_analysis = sentiment
    broker.retrieved_documents = filtered_docs
    
    return NodeResult(status=NodeStatus.SUCCESS)
```

#### 2.2 Add Access Tracking to RAG

**File**: [`src/rag/qdrant_connector.py`](src/rag/qdrant_connector.py)

**Add method**:

```python
def update_access_count(self, point_id: str) -> None:
    """Increment access count for a document.
    
    Args:
        point_id: The point ID to update
    """
    # Get current metadata
    point = self.client.retrieve(
        collection_name=self.collection_name,
        ids=[point_id]
    )[0]
    
    # Update access count
    metadata = point.payload
    metadata["access_count"] = metadata.get("access_count", 0) + 1
    metadata["last_accessed"] = datetime.now().isoformat()
    
    # Update in Qdrant
    self.client.set_payload(
        collection_name=self.collection_name,
        payload=metadata,
        points=[point_id]
    )
```

---

### Phase 3: Integrate Detox Protocol

#### 3.1 Create Detox Orchestration

**File**: [`src/orchestrator.py`](src/orchestrator.py) (modify existing)

**Add detox scheduler**:

```python
class Orchestrator:
    def __init__(self, ...):
        # ... existing init ...
        
        # Initialize detox scheduler
        from src.nodes.algo_nodes import DetoxScheduler
        self.detox_scheduler = DetoxScheduler(
            idle_trigger_minutes=60,
            min_session_interval_minutes=120
        )
    
    async def process_message(self, message):
        # Update activity
        self.detox_scheduler.update_activity()
        
        # ... existing processing ...
    
    async def run_background_tasks(self):
        """Run background tasks like detox sessions."""
        while True:
            if self.detox_scheduler.should_run_detox():
                await self._run_detox_session()
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _run_detox_session(self):
        """Run a complete detox session."""
        self.detox_scheduler.start_detox_session()
        
        try:
            # Create broker for detox
            broker = KnowledgeBroker()
            
            # Run detox session node
            detox_node = DetoxSessionNode(
                nudging_algorithm=self.nudging_algorithm,
                memory_consolidation_node=self.memory_consolidation_node
            )
            
            result = await detox_node.execute(broker)
            
            logger.info(f"Detox session completed: {result.data}")
            
        finally:
            self.detox_scheduler.end_detox_session()
```

#### 3.2 Initialize Nudging Algorithm

**File**: [`src/orchestrator.py`](src/orchestrator.py)

**Add to init**:

```python
from src.rag.algorithms import NudgingAlgorithm, CompanionPersonality, NudgingWeights

self.nudging_algorithm = NudgingAlgorithm(
    companion=CompanionPersonality(),  # Load from config or RAG
    weights=NudgingWeights()  # Load from config
)
```

---

### Phase 4: Implement Trust Analysis

#### 4.1 Create Trust Tracking Storage

**File**: [`src/storage/trust_store.py`](src/storage/trust_store.py) (NEW)

```python
"""Storage for trust analysis data."""

import logging
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TrustRecord:
    """Record of trust-related interaction."""
    
    timestamp: datetime
    user_id: str
    interaction_type: str  # "positive", "negative", "neutral"
    depth_score: float  # How deep/personal was the interaction
    consistency_score: float  # How consistent with past behavior


class TrustStore:
    """Stores and retrieves trust analysis data."""
    
    def __init__(self):
        self.records: dict[str, list[TrustRecord]] = {}
    
    def add_record(self, record: TrustRecord) -> None:
        """Add a trust record."""
        if record.user_id not in self.records:
            self.records[record.user_id] = []
        
        self.records[record.user_id].append(record)
    
    def get_records(self, user_id: str) -> list[TrustRecord]:
        """Get all records for a user."""
        return self.records.get(user_id, [])
    
    def calculate_trust_score(self, user_id: str) -> float:
        """Calculate current trust score for a user."""
        records = self.get_records(user_id)
        
        if not records:
            return 0.0
        
        # Calculate based on:
        # - Relationship age
        # - Interaction count
        # - Positive/negative ratio
        # - Consistency
        # - Depth of sharing
        
        # Placeholder implementation
        return min(1.0, len(records) / 100.0)
```

#### 4.2 Integrate Trust Analysis into Pipeline

**File**: [`src/nodes/processing/sentiment_analysis_node.py`](src/nodes/processing/sentiment_analysis_node.py)

**Add after sentiment analysis**:

```python
# Calculate trust score
from src.storage.trust_store import TrustStore

trust_store = TrustStore()  # Should be singleton or injected
trust_score = trust_store.calculate_trust_score(dialogue_input.speaker)

broker.trust_score = trust_score
```

---

### Phase 5: Connect Memory Evaluator

#### 5.1 Add Memory Evaluator to Pipeline

**File**: [`src/orchestrator.py`](src/orchestrator.py) or create new pipeline

**After sentiment analysis and memory retrieval**:

```python
# After SentimentAnalysisNode has run and populated broker.retrieved_documents

from src.nodes.algo_nodes import MemoryEvaluatorNode

memory_evaluator = MemoryEvaluatorNode(
    llm_provider=self.llm_provider,
    max_retries=3
)

result = await memory_evaluator.execute(broker)

# Now broker.evaluated_memories contains (document, evaluation) tuples
```

#### 5.2 Use Evaluated Memories in Context Interpreter

**File**: [`src/handlers/context_interpreter.py`](src/handlers/context_interpreter.py)

**Modify to use evaluated memories**:

```python
def interpret(self, query: str, evaluated_memories: list[tuple[RAGDocument, MemoryEvaluation]]):
    """Interpret context using AI-evaluated memories."""
    
    # Apply AI evaluations to adjust scores
    adjusted_docs = []
    for doc, evaluation in evaluated_memories:
        # Update document metadata with AI evaluation
        doc.metadata["ai_relevance"] = evaluation.relevance
        doc.metadata["ai_chrono_relevance"] = evaluation.chrono_relevance
        
        # Apply boost if recommended
        if evaluation.should_boost:
            doc.score *= (1.0 + evaluation.boost_factor)
        
        adjusted_docs.append(doc)
    
    # Sort by adjusted score
    adjusted_docs.sort(key=lambda d: d.score, reverse=True)
    
    # Use top N for context
    context_docs = adjusted_docs[:25]
    
    # ... rest of interpretation logic ...
```

---

## Critical Implementation Notes

### 1. Personality System Must Be Dynamic

**Current Issue**: [`CompanionPersonality`](src/rag/algorithms/nudging_algorithm.py:20) has hardcoded topics:

```python
base_personality: dict[str, float] = field(default_factory=lambda: {
    "gender_relations": 0.1,
    "politics": 0.0,
    "general_outlook": 0.1
})
```

**Solution**: Make it fully dynamic:

```python
@dataclass
class CompanionPersonality:
    """Companion personality configuration - fully dynamic."""
    
    base_personality: dict[str, float] = field(default_factory=dict)
    current_positions: dict[str, float] = field(default_factory=dict)
    personality_weight: float = 0.3
    
    def get_position(self, topic: str, default: float = 0.0) -> float:
        """Get position for any topic, creating if needed."""
        if topic not in self.current_positions:
            self.current_positions[topic] = self.base_personality.get(topic, default)
        return self.current_positions[topic]
    
    def set_position(self, topic: str, position: float) -> None:
        """Set position for any topic."""
        self.current_positions[topic] = position
        
        # Also update base if this is first time
        if topic not in self.base_personality:
            self.base_personality[topic] = position
```

**Where to store**: RAG system with special collection `companion_personality`

### 2. Memory Evaluator Needs Conversation Context

**Current Issue**: [`MemoryEvaluatorNode._get_conversation_state()`](src/nodes/algo_nodes/memory_evaluator_node.py:169) assumes `broker.conversation_history` exists

**Solution**: Ensure `KnowledgeBroker` has conversation history:

```python
# In KnowledgeBroker
@dataclass
class KnowledgeBroker:
    # ... existing fields ...
    conversation_history: list[SentimentAnalysis] = field(default_factory=list)
    trust_score: float = 0.0
    retrieved_documents: list[RAGDocument] = field(default_factory=list)
    evaluated_memories: list[tuple[RAGDocument, MemoryEvaluation]] = field(default_factory=list)
```

### 3. Detox Session Needs Conversation History

**Current Issue**: [`DetoxSessionNode.execute()`](src/nodes/algo_nodes/detox_scheduler.py:185) expects conversation history

**Solution**: Store conversation history in RAG or separate storage:

```python
# In ConversationStore or similar
class ConversationStore:
    def get_recent_history(self, user_id: str, limit: int = 50) -> list[SentimentAnalysis]:
        """Get recent conversation history for user."""
        # Query RAG for recent messages
        # Return as list of SentimentAnalysis objects
        pass
```

---

## Data Flow Diagrams

### Current Flow (Incomplete)

```
User Message
    ↓
SentimentAnalysisNode
    ↓
SentimentAnalysisHandler (LLM)
    ↓
Store in RAG with metadata
    ↓
[MISSING: Memory retrieval with decay]
    ↓
[MISSING: Memory evaluation]
    ↓
Context Interpreter
    ↓
Primary Response
```

### Target Flow (Complete)

```
User Message
    ↓
SentimentAnalysisNode
    ├─→ Analyze sentiment (LLM)
    ├─→ Store in RAG
    ├─→ Retrieve memories (RAG query)
    ├─→ Apply memory decay filter
    └─→ Update access counts
    ↓
TrustAnalysisNode
    └─→ Calculate trust score
    ↓
MemoryEvaluatorNode
    ├─→ Re-evaluate each memory (LLM)
    └─→ Apply AI adjustments
    ↓
Context Interpreter
    └─→ Use evaluated memories
    ↓
Primary Response
    ↓
[Background: Detox Scheduler]
    ├─→ Check if idle
    └─→ If yes: Run DetoxSessionNode
        ├─→ Extract topics
        ├─→ Run nudging algorithm
        ├─→ Run memory consolidation
        └─→ Generate guidance
```

---

## Testing Strategy

### Unit Tests

**File**: [`tests/test_nudging_algorithm.py`](tests/test_nudging_algorithm.py) (NEW)

```python
"""Tests for nudging algorithm."""

import pytest
from src.rag.algorithms import NudgingAlgorithm, CompanionPersonality, NudgingWeights


class TestNudgingAlgorithm:
    def test_calculate_neutral(self):
        """Test neutral position calculation."""
        algo = NudgingAlgorithm()
        neutral = algo.calculate_neutral("test_topic")
        assert -1.0 <= neutral <= 1.0
    
    def test_calculate_nudge_basic(self):
        """Test basic nudge calculation."""
        algo = NudgingAlgorithm()
        result = algo.calculate_nudge(
            topic="test",
            user_position=-0.7,
            trust_score=0.0
        )
        
        assert result is not None
        assert result.topic == "test"
        assert -1.0 <= result.companion_after <= 1.0
    
    def test_trust_score_affects_influence(self):
        """Test that higher trust increases companion influence."""
        algo = NudgingAlgorithm()
        
        # Low trust
        result_low = algo.calculate_nudge("test", -0.7, trust_score=0.1)
        
        # High trust
        result_high = algo.calculate_nudge("test", -0.7, trust_score=0.9)
        
        # Companion influence should be higher with high trust
        assert result_high.companion_influence > result_low.companion_influence
```

**File**: [`tests/test_memory_evaluator.py`](tests/test_memory_evaluator.py) (NEW)

```python
"""Tests for memory evaluator node."""

import pytest
from src.nodes.algo_nodes import MemoryEvaluatorNode
from src.rag.selector import RAGDocument


@pytest.mark.asyncio
async def test_memory_evaluator_basic(mock_llm):
    """Test basic memory evaluation."""
    node = MemoryEvaluatorNode(llm_provider=mock_llm)
    
    # Create mock broker with documents
    broker = MockBroker()
    broker.retrieved_documents = [
        RAGDocument(
            content="Test memory",
            score=0.8,
            metadata={"timestamp": "2024-01-01T00:00:00Z"}
        )
    ]
    
    result = await node.execute(broker)
    
    assert result.status == NodeStatus.SUCCESS
    assert hasattr(broker, "evaluated_memories")
```

### Integration Tests

**File**: [`tests/test_memory_decay_integration.py`](tests/test_memory_decay_integration.py) (NEW)

```python
"""Integration tests for memory decay with AI evaluation."""

@pytest.mark.asyncio
async def test_full_memory_pipeline():
    """Test complete memory pipeline from storage to retrieval."""
    # 1. Store memory with sentiment
    # 2. Wait (or mock time passage)
    # 3. Retrieve with memory decay
    # 4. Evaluate with AI
    # 5. Verify scores are adjusted correctly
    pass
```

---

## Configuration Updates

### Add to [`src/config/defaults.py`](src/config/defaults.py)

```python
DEFAULT_CONFIG = {
    # ... existing config ...
    
    # Memory Decay
    "memory_half_life_days": 30.0,
    "chrono_weight": 1.0,
    "memory_retrieval_threshold": 0.15,
    "memory_prune_threshold": 0.05,
    "max_context_documents": 25,
    
    # Detox Protocol
    "detox_idle_trigger_minutes": 60,
    "detox_min_interval_minutes": 120,
    "detox_max_duration_minutes": 30,
    
    # Nudging Algorithm
    "nudge_strength": 0.15,
    "max_companion_drift": 0.3,
    "base_user_influence": 0.3,
    "base_companion_influence": 0.7,
    "max_trust_boost": 0.3,
    
    # Memory Consolidation
    "consolidation_threshold": 0.7,
    "max_memories_per_batch": 10,
    
    # Trust Analysis
    "trust_calculation_method": "interaction_based",  # or "time_based", "hybrid"
}
```

---

## Database Schema Updates

### Qdrant Collections

#### 1. Main Memories Collection

**Collection**: `memories`

**Metadata schema**:

```python
{
    # Core fields
    "timestamp": "2024-01-15T10:30:00Z",
    "memory_owner": "user",
    "point_id": "uuid-string",
    
    # Sentiment analysis
    "sentiment": "negative",
    "confidence": 0.95,
    "emotional_tone": "grieving",
    "relevance": 0.9,
    "chrono_relevance": 0.95,
    "context_summary": "User's mother passed away",
    "key_topics": ["family", "death", "grief"],
    
    # Memory tracking
    "access_count": 5,
    "last_accessed": "2024-01-20T14:22:00Z",
    "memory_score": 0.85,
    
    # AI evaluations (list of historical evaluations)
    "ai_evaluations": [
        {
            "timestamp": "2024-01-20T00:00:00Z",
            "relevance": 0.92,
            "chrono_relevance": 0.93,
            "reasoning": "Still emotionally significant"
        }
    ],
    
    # Consolidation
    "is_consolidated": False,
    "consolidated_with": [],
    "consolidation_reasoning": None
}
```

#### 2. Companion Personality Collection

**Collection**: `companion_personality`

**Document structure**:

```python
{
    "topic": "gender_relations",
    "base_position": 0.1,
    "current_position": 0.12,
    "last_updated": "2024-01-20T00:00:00Z",
    "evolution_history": [
        {
            "timestamp": "2024-01-15T00:00:00Z",
            "position": 0.1,
            "reason": "Initial baseline"
        },
        {
            "timestamp": "2024-01-20T00:00:00Z",
            "position": 0.12,
            "reason": "Nudged during detox session"
        }
    ]
}
```

#### 3. Trust Records Collection

**Collection**: `trust_records`

**Document structure**:

```python
{
    "user_id": "user",
    "timestamp": "2024-01-20T14:22:00Z",
    "interaction_type": "positive",
    "depth_score": 0.8,
    "consistency_score": 0.9,
    "notes": "User shared personal information about family"
}
```

---

## Implementation Checklist

### Immediate (Session 1)

- [x] Create analysis document
- [x] Create MemoryEvaluatorNode
- [x] Create MemoryConsolidationNode
- [x] Create DetoxScheduler
- [x] Create NudgingAlgorithm
- [ ] Fix all styleguide violations
- [ ] Create memory.py models file
- [ ] Update __init__.py files

### Short-term (Session 2)

- [ ] Implement TrustAnalysisNode
- [ ] Create TrustStore
- [ ] Integrate memory decay into SentimentAnalysisNode
- [ ] Add access tracking to RAG
- [ ] Update KnowledgeBroker with new fields

### Medium-term (Session 3)

- [ ] Integrate MemoryEvaluatorNode into pipeline
- [ ] Integrate DetoxScheduler into Orchestrator
- [ ] Create background task runner for detox sessions
- [ ] Implement companion personality storage/retrieval

### Long-term (Session 4+)

- [ ] Write comprehensive tests
- [ ] Implement source position retrieval (RAG queries)
- [ ] Add memory consolidation to detox sessions
- [ ] Create monitoring/statistics dashboard
- [ ] Implement adaptive parameters per user

---

## Common Pitfalls & Solutions

### Pitfall 1: Circular Imports

**Problem**: Nodes import from algorithms, algorithms import from nodes

**Solution**: Keep algorithms pure (no node dependencies), nodes can import algorithms

### Pitfall 2: Missing Metadata

**Problem**: Memory decay fails if `relevance`, `chrono_relevance`, or `timestamp` missing

**Solution**: Always validate and provide defaults:

```python
relevance = doc.metadata.get("relevance", 0.5)
chrono_relevance = doc.metadata.get("chrono_relevance", 0.5)
timestamp = doc.metadata.get("timestamp", datetime.now().isoformat())
```

### Pitfall 3: Async/Sync Mismatch

**Problem**: Some methods are async, some are sync

**Solution**: Keep algorithms sync (pure functions), nodes async (I/O operations)

### Pitfall 4: Broker State Management

**Problem**: Broker fields not initialized, causing AttributeError

**Solution**: Use `getattr(broker, "field_name", default_value)` or ensure fields exist

---

## Performance Considerations

### 1. Memory Evaluation is Expensive

**Problem**: Evaluating every retrieved document with LLM is slow

**Solutions**:
- Only evaluate top N documents (e.g., top 50 by similarity)
- Cache evaluations for recently evaluated documents
- Use smaller/faster LLM for evaluation
- Batch evaluate multiple documents in single LLM call

### 2. Detox Sessions Can Be Long

**Problem**: Consolidating 1000+ memories takes time

**Solutions**:
- Run in background thread/process
- Process in batches
- Set max duration and pause/resume if needed
- Only consolidate topics with 3+ memories

### 3. RAG Queries Can Be Slow

**Problem**: Retrieving all memories for consolidation is slow

**Solutions**:
- Use pagination
- Filter by date range (only consolidate recent memories)
- Use Qdrant scroll API for large result sets
- Cache results between detox sessions

---

## Future Enhancements

### 1. Multi-Dimensional Memory Scoring

Add dimensions beyond relevance and chrono_relevance:

```python
@dataclass
class MemoryDimensions:
    """Multi-dimensional memory scoring."""
    
    relevance: float  # General importance
    chrono_relevance: float  # Temporal persistence
    emotional_intensity: float  # How emotionally charged
    social_context: float  # Relationship relevance
    actionability: float  # Can something be done
    narrative_arc: float  # Part of ongoing story
```

### 2. Adaptive Parameters

Learn optimal parameters per user:

```python
class AdaptiveMemoryDecay(MemoryDecayAlgorithm):
    """Memory decay with user-specific parameters."""
    
    def __init__(self, user_id: str):
        # Load user-specific parameters from RAG
        user_params = self._load_user_params(user_id)
        
        super().__init__(
            memory_half_life_days=user_params.half_life,
            chrono_weight=user_params.chrono_weight
        )
```

### 3. Memory Hierarchies

Organize memories into hierarchies:

```
Life Chapter: "College Years"
    ├─ Theme: "Friendships"
    │   ├─ Event: "Met best friend"
    │   └─ Event: "Graduation party"
    └─ Theme: "Academic Struggles"
        ├─ Event: "Failed exam"
        └─ Event: "Changed major"
```

### 4. Memory Visualization

Create UI to visualize memory landscape:
- Timeline view of important events
- Topic clusters
- Decay curves over time
- Trust score progression

---

## Debugging Tips

### 1. Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Check Memory Scores

```python
from src.rag.algorithms import MemoryDecayAlgorithm

algo = MemoryDecayAlgorithm()
stats = algo.get_decay_stats(documents)
print(stats)
```

### 3. Inspect Nudging State

```python
from src.rag.algorithms import NudgingAlgorithm

algo = NudgingAlgorithm()
state = algo.get_algorithm_state()
print(state)
```

### 4. Monitor Detox Sessions

```python
# Add to DetoxSessionNode
logger.info(f"Detox session results: {broker.detox_results}")
```

---

## References

- **Analysis**: [`docs/ANALYSIS_MEMORY_DECAY_AND_DETOX.md`](docs/ANALYSIS_MEMORY_DECAY_AND_DETOX.md)
- **Memory Decay Docs**: [`docs/MEMORY_DECAY_ALGORITHM.md`](docs/MEMORY_DECAY_ALGORITHM.md)
- **Detox Protocol Docs**: [`docs/algorithms/DETOX_PROTOCOL.md`](docs/algorithms/DETOX_PROTOCOL.md)
- **Brainstorming**: [`docs/BRAINSTORMING.md`](docs/BRAINSTORMING.md)
- **Styleguide**: [`.clinerules/1-styleguide.md`](.clinerules/1-styleguide.md)

---

## Quick Start for Next Session

1. **Fix styleguide violations**: Remove all `dict[str, Any]`, replace with dataclasses
2. **Create `src/models/memory.py`**: Centralize memory-related models
3. **Implement TrustAnalysisNode**: Critical dependency for nudging
4. **Integrate into pipeline**: Connect all nodes in orchestrator
5. **Write tests**: Ensure everything works correctly

---

## Notes

- The personality system should be **completely dynamic** - topics are discovered and added as needed, not predefined
- Memory evaluation is **expensive** - optimize by evaluating only top candidates
- Detox sessions should run in **background** - don't block user interactions
- Trust analysis is **critical** - without it, nudging algorithm can't work properly
- All new code must follow **styleguide** - no exceptions
