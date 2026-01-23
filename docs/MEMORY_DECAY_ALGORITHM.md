# Memory Decay Algorithm Documentation

## Overview

The Memory Decay Algorithm mimics human memory by applying time-based scoring to RAG documents. Important, long-lasting memories (high chrono_relevance) persist much longer than mundane, temporary information (low chrono_relevance).

## Core Concept

**Human Memory Analogy:**
- **Family loss (chrono=0.95)**: Stays in memory for years → high chrono_relevance
- **Breakfast menu (chrono=0.1)**: Forgotten in days → low chrono_relevance
- **Exam tomorrow (chrono=0.2)**: Relevant until exam, then fades → low chrono_relevance

## Mathematical Formula

### Exponential Decay

```python
decay_rate = 1.0 - (chrono_relevance * chrono_weight)
lambda_decay = ln(2) / memory_half_life_days
decay_multiplier = exp(-decay_rate * lambda_decay * age_days)

memory_score = relevance * decay_multiplier
```

### Parameters

1. **`relevance`** (0.0-1.0): Base importance/impact
   - How important the event is RIGHT NOW
   - Example: Family loss = 0.9, Lunch thought = 0.3

2. **`chrono_relevance`** (0.0-1.0): Temporal persistence
   - How long the memory should last
   - High value = persists longer
   - Low value = fades quickly

3. **`memory_half_life_days`** (configurable): Time scale
   - Days for 50% decay at mid chrono_relevance (0.5)
   - Default: 30 days
   - Increase to make ALL memories last longer
   - Decrease to make ALL memories fade faster

4. **`chrono_weight`** (0.0-2.0): Sensitivity multiplier
   - Amplifies or dampens chrono_relevance effect
   - 1.0 = normal behavior
   - >1.0 = exaggerate differences (high chrono lasts MUCH longer)
   - <1.0 = dampen differences (more uniform decay)

## Examples

### Default Configuration
(`half_life=30 days`, `chrono_weight=1.0`)

**Family Loss:**
```python
relevance=0.9, chrono=0.95, half_life=30
Age 7 days → score ≈ 0.88 (barely decayed)
Age 30 days → score ≈ 0.85 (still strong)
Age 365 days → score ≈ 0.66 (persists a year later)
```

**Breakfast Menu:**
```python
relevance=0.3, chrono=0.1, half_life=30
Age 7 days → score ≈ 0.14 (fading)
Age 14 days → score ≈ 0.07 (mostly gone)
Age 30 days → score ≈ 0.02 (forgotten)
```

**Exam Tomorrow:**
```python
relevance=0.7, chrono=0.2, half_life=30
Age 1 day → score ≈ 0.63 (still relevant)
Age 2 days → score ≈ 0.57 (fading after exam)
Age 7 days → score ≈ 0.39 (no longer important)
```

### Longer Memory (`half_life=60 days`)

All memories last 2x longer:
```python
Family Loss (chrono=0.95):
  Age 60 days → score ≈ 0.85 (same as 30 days before)
  
Breakfast (chrono=0.1):
  Age 28 days → score ≈ 0.07 (takes 2x longer to forget)
```

### Amplified Chrono Effect (`chrono_weight=1.5`)

Differences become more extreme:
```python
Family Loss (chrono=0.95 → effective 1.0):
  Age 30 days → score ≈ 0.90 (almost no decay)
  
Breakfast (chrono=0.1 → effective 0.15):
  Age 14 days → score ≈ 0.05 (decays even faster)
```

## Usage

### Basic Retrieval with Memory Scoring

```python
from src.rag.algorithms import MemoryDecayAlgorithm
from src.rag.qdrant_connector import QdrantRAG

# Initialize algorithm
memory_algo = MemoryDecayAlgorithm(
    memory_half_life_days=30.0,
    chrono_weight=1.0,
    retrieval_threshold=0.15,
    max_documents=25
)

# Retrieve from RAG
rag = QdrantRAG(collection_name="memories")
raw_documents = rag.retrieve_documents(
    query_embedding=query_vector,
    limit=100  # Get more, we'll filter
)

# Apply memory decay filtering
filtered_docs = memory_algo.filter_and_rank(
    documents=raw_documents,
    threshold=0.15,  # Min score to keep
    max_docs=25      # Top 25 after scoring
)

# Use filtered_docs for context interpretation
```

### Storing Documents with Metadata

```python
from datetime import datetime

# When storing a document, include sentiment metadata
metadata = {
    "timestamp": datetime.now().isoformat(),
    "relevance": 0.9,           # From sentiment analysis
    "chrono_relevance": 0.95,   # From sentiment analysis
    "context_summary": "User's mother passed away",
    "sentiment": "negative",
    "emotional_tone": "grief"
}

rag.store(
    text="Important conversation about family",
    embedding=embedding_vector,
    metadata=metadata
)
```

### Manual Score Calculation

```python
from datetime import datetime, timedelta

# Calculate score for a specific memory
score = memory_algo.calculate_memory_score(
    relevance=0.9,
    chrono_relevance=0.95,
    timestamp=datetime.now() - timedelta(days=30),
    current_time=datetime.now()
)

print(f"Memory score after 30 days: {score:.3f}")
```

### Database Pruning

```python
# Periodic cleanup of old, faded memories
from src.rag.algorithms import MemoryDecayAlgorithm
from src.rag.qdrant_connector import QdrantRAG

memory_algo = MemoryDecayAlgorithm(
    prune_threshold=0.05  # Delete if score < thresh
)

rag = QdrantRAG(collection_name="memories")

# Get all documents (or batch process)
all_docs = rag.retrieve_documents(
    query_embedding=[0] * 384,  # Dummy query
    limit=10000,
    score_threshold=0.0  # Get everything
)

# Identify prunable documents
prunable_ids = memory_algo.identify_prunable(
    documents=all_docs,
    prune_threshold=0.05
)

# Delete from database
if prunable_ids:
    rag.delete(prunable_ids)
    print(f"Pruned {len(prunable_ids)} faded memories")
```

### Statistics and Monitoring

```python
# Get statistics about memory decay
stats = memory_algo.get_decay_stats(documents)

print(f"Total documents: {stats['total_documents']}")
print(f"Average memory score: {stats['avg_memory_score']:.3f}")
print(f"Above retrieval threshold: {stats['above_retrieval_threshold']}")
print(f"Below prune threshold: {stats['below_prune_threshold']}")
```

## Configuration

### defaults.py

```python
DEFAULT_CONFIG = {
    # Memory Decay Algorithm Configuration
    "memory_half_life_days": 30.0,      # Time scale for decay
    "chrono_weight": 1.0,                # Sensitivity multiplier
    "memory_retrieval_threshold": 0.15,  # Min score for retrieval
    "memory_prune_threshold": 0.05,      # Min score to keep in DB
    "max_context_documents": 25,         # Max docs to interpreter
}
```

### config.json

```json
{
  "memory_half_life_days": 30.0,
  "chrono_weight": 1.0,
  "memory_retrieval_threshold": 0.15,
  "memory_prune_threshold": 0.05,
  "max_context_documents": 25
}
```

## Tuning Guide

### Scenario: Long-term Personal Assistant

**Goal**: Memories should last months, important events persist years

```python
memory_half_life_days = 90.0    # 3 month baseline
chrono_weight = 1.2              # Amplify chrono effect
retrieval_threshold = 0.12       # Lower threshold (more memories)
prune_threshold = 0.03           # Aggressive pruning
```

**Effect**: Family events stay relevant for years, mundane things fade in weeks

### Scenario: Short-term Chat Bot

**Goal**: Memories fade quickly, focus on recent context

```python
memory_half_life_days = 7.0     # 1 week baseline
chrono_weight = 0.5              # Dampen chrono effect (more uniform)
retrieval_threshold = 0.20       # Higher threshold (fewer memories)
prune_threshold = 0.08           # Conservative pruning
```

**Effect**: Even important events fade within weeks, keeps database clean

### Scenario: Testing/Development

**Goal**: Exaggerated behavior for easy observation

```python
memory_half_life_days = 1.0     # 1 day baseline
chrono_weight = 2.0              # Maximum amplification
retrieval_threshold = 0.10       # Low threshold (see more)
prune_threshold = 0.02           # Aggressive pruning
```

**Effect**: See memory decay in hours instead of days

## Integration with Pipeline

### Complete Flow

```
1. User Message
   ↓
2. Sentiment Analysis → relevance, chrono_relevance
   ↓
3. Store with timestamp metadata
   ↓
4. Later: RAG Retrieval
   ↓
5. MemoryDecayAlgorithm.filter_and_rank()
   ↓
6. Context Interpreter (top 25 docs)
   ↓
7. Primary Response
```

### Example Pipeline Code

```python
# In pipeline or custom retrieval logic
memory_algo = MemoryDecayAlgorithm(
    memory_half_life_days=settings.memory_half_life_days,
    chrono_weight=settings.chrono_weight,
    retrieval_threshold=settings.memory_retrieval_threshold,
    max_documents=settings.max_context_documents
)

# Retrieve and filter
raw_docs = rag.retrieve_documents(query_embedding, limit=100)
filtered_docs = memory_algo.filter_and_rank(raw_docs)

# Interpret context
context = interpreter.interpret(query, filtered_docs)

# Generate response
response = primary_handler.generate_response(query, context=context)
```

## Decay Curves Visualization

### High Chrono-Relevance (0.9)
```
Score
1.0 |████████████████████████████████████████
0.9 |███████████████████████████████████▓▓▓▓░
0.8 |██████████████████████████████████▓▓▓░░░
0.7 |█████████████████████████████████▓▓▓░░░░
    +----------------------------------------
     0d   30d   60d   90d  120d  150d  180d
```

### Medium Chrono-Relevance (0.5)
```
Score
1.0 |████████████▓░░░░░░░░░░░░░░░░░░░░░░░░░░
0.9 |███████████▓░░░░░░░░░░░░░░░░░░░░░░░░░░░
0.8 |██████████▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░
0.7 |█████████▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
0.6 |████████▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
0.5 |███████▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    +----------------------------------------
     0d   30d   60d   90d  120d  150d  180d
```

### Low Chrono-Relevance (0.1)
```
Score
1.0 |██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
0.9 |█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
0.8 |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
0.7 |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
0.6 |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
0.5 |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    +----------------------------------------
     0d   30d   60d   90d  120d  150d  180d
```

## Best Practices

1. **Set realistic half_life**: 30 days is good default for conversational AI
2. **Use chrono_weight carefully**: 1.0 is usually best, adjust only if needed
3. **Monitor statistics**: Check `get_decay_stats()` to tune parameters
4. **Prune regularly**: Run pruning weekly or monthly to keep DB clean
5. **Test with real data**: Tune parameters based on actual usage patterns
6. **Document your settings**: Different use cases need different configs

## Troubleshooting

### Problem: Too many old memories in retrieval

**Solution**: Increase `retrieval_threshold` or decrease `memory_half_life_days`

### Problem: Important memories disappearing too fast

**Solution**: Increase `memory_half_life_days` or `chrono_weight`

### Problem: Database growing too large

**Solution**: Decrease `prune_threshold` or run pruning more frequently

### Problem: All memories decaying uniformly

**Solution**: Verify sentiment analysis is setting proper `chrono_relevance` values

## API Reference

See `src/rag/algorithms.py` for complete API documentation.

### Key Methods

- `calculate_memory_score()`: Score single memory
- `score_documents()`: Batch score documents
- `filter_and_rank()`: Main retrieval filter
- `identify_prunable()`: Find documents to delete
- `get_decay_stats()`: Get statistics

## Future Enhancements

Potential additions:
- Multiple decay curves (linear, power law, sigmoid)
- Per-user memory parameters
- Automatic parameter tuning based on usage
- Memory consolidation (merge similar old memories)
- Boosting for recently accessed memories
