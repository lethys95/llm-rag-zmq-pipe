# Session Handoff — 2026-04-21

## What was done this session

### Code
- Fixed `BaseNode.__init__` (name bug + `should_run` returning Ellipsis)
- Rewrote `NodeOptionsRegistry` → `NodeRegistry` with instance-based DI via constructor inspection
- Simplified `node_registry_decorator.py` — now just a module-level set, no singleton side effects
- Updated `DecisionEngine.select_node` to take broker + registry (no more global singleton lookup), prompt now includes execution history
- Fixed all communication nodes — `ZMQHandler` injected via `__init__`, not class variable
- Fixed `StoreConversationNode` — removed invalid kwargs to `super().__init__()`
- Fixed `PrimaryResponseNode` — renamed param to `primary_response_handler`
- Deleted `node_options_registry.py`, updated all imports
- Wired `src/orchestrator.py` — builds all deps at startup, runs async node loop per request via `asyncio.to_thread` for ZMQ polling
- Implemented `SentimentAnalysisNode` in `src/nodes/algo_nodes/`
- 54 passing tests covering: NodeRegistry DI, BaseNode, KnowledgeBroker, SentimentAnalysisHandler, SentimentAnalysisNode, MemoryDecayAlgorithm

### Decisions made
- Decorator pattern kept — 20-30 nodes expected, co-location is worth it
- Handler + Node split kept — nodes are orchestration units, handlers are capabilities. A node may eventually call multiple handlers
- No base class for "system instruction nodes" yet — premature
- Access boost in memory decay dropped — feedback loop risk, complexity not worth it
- Pruning deferred — not blocking, can wait until pipeline works end-to-end

---

## Critical design decisions made this session (READ THIS)

### The data model needs to be redesigned before MemoryRetrievalNode

The current `SentimentAnalysis` model is trying to do two different jobs at once and should be split into two:

**1. `EmotionalState`** — session-scoped, drives response strategy, does NOT need long-term storage
```python
emotions: dict[str, float]   # {"sadness": 0.8, "anger": 0.1, ...}
valence: float                # -1.0 to 1.0
arousal: float                # 0.0 to 1.0
dominance: float              # 0.0 to 1.0
```
VAD (Valence-Arousal-Dominance) model. Arousal = intensity. Dominance = feeling in control vs powerless. Useful for response strategy and crisis detection. NOT useful as metadata on pizza preference memories.

**2. `List[UserFact]`** — stored long-term in Qdrant, one vector point per fact
```python
class UserFact(BaseModel):
    claim: str           # "user likes pepperoni pizza"
    sentiment: str       # positive / negative / neutral (basic valence per claim)
    confidence: float
    chrono_relevance: float
    subject: str         # "food preferences", "relationships", etc.
```
This replaces raw message storage. Instead of storing "I'm going to make pizza tonight..." as one blob, you extract 3-5 atomic facts and store each as a separate vector point. Better retrieval precision. Chrono_relevance applies per fact — "user likes pepperoni" is stable, "user is making pizza tonight" is ephemeral.

**Constraint:** Extraction should be limited to directly stated or strongly implied facts. No loose inference. Hallucinated preferences that get stored and retrieved will feel like the companion misremembers the user, which is worse than not remembering.

### What this means for the next implementation steps

The current `SentimentAnalysis` model and `SentimentAnalysisHandler` need to be replaced/redesigned before `MemoryRetrievalNode` makes sense. The retrieval node needs to know what it's retrieving — raw messages (current) or atomic facts (new design).

Suggested order:
1. Design and implement `EmotionalState` model
2. Design and implement `UserFact` model  
3. Rewrite `SentimentAnalysisHandler` → split into `EmotionalStateHandler` + `UserFactExtractionHandler`
4. Update `SentimentAnalysisNode` to use the new handlers and write both to broker
5. Update `StoreConversationNode` to store `UserFact` list to Qdrant (one point per fact)
6. THEN implement `MemoryRetrievalNode` — it now retrieves `UserFact` points, applies decay, writes to `broker.retrieved_documents`

### KnowledgeBroker fields that need adding/changing
- `broker.sentiment_analysis` → rename to `broker.emotional_state: EmotionalState`
- Add `broker.user_facts: list[UserFact]`
- `broker.retrieved_documents` stays but now contains `UserFact`-originated points

---

## Current codebase state

### Working and solid
See previous NEXT_SESSION notes — all of that still applies. In addition:
- `NodeRegistry` — fully working, DI by constructor inspection
- `SentimentAnalysisNode` — implemented, tested
- `Orchestrator` — wired, async, builds all deps at startup

### What's not wired yet
- `src/orchestrator.py` — wired but needs `algo_nodes` import added (already done for SentimentAnalysisNode)
- `MemoryRetrievalNode` — blocked on data model redesign above
- `NeedsAnalysisNode` — not yet implemented
- Response strategy node — not yet implemented
- Detox/recalibration background task — not yet implemented

### Known issues
- `OpenRouterLLM.generate()` hardcodes `settings.primary_llm` — all nodes use the same model profile regardless of which LLM config was intended. Multi-model support (fast worker vs primary) needs to be fixed in the LLM provider layer. Not blocking for now.
- `rag_enabled` field referenced in `src/rag/factory.py` but doesn't exist in `Settings`. Factory bypassed in orchestrator by instantiating `QdrantRAG` directly. Factory should be fixed or removed.
- `ReceiveMessageNode` is registered but the orchestrator also handles ZMQ receive directly in `_main_loop`. These are redundant — decide whether receive is a node or orchestrator responsibility.

---

## Key files to read first
1. This file
2. `docs/PROJECT_VISION.md`
3. `src/orchestrator.py`
4. `src/nodes/orchestration/knowledge_broker.py`
5. `src/nodes/orchestration/node_registry.py`
