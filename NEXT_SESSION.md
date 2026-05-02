# Session Handoff — 2026-04-25

## Current state

167 passing tests. Pipeline tested end-to-end against real LLMs (Cerebras/gpt-oss-120b for workers,
GLM-4.7 for primary). Parallel coordinator is live. Node descriptions are rich and complete.
Dependency and criticality metadata is on every node.

Read these docs before touching anything:
- `docs/ADVISOR_PATTERN_CONCEPT.md` — critical architecture, read first
- `docs/ORCHESTRATOR.md` — event model and coordinator gap
- `docs/PSYCHOLOGY.md` — OCEAN, VAD, tonic/phasic, detox framework

---

## What was built this session (2026-04-25)

**Bug fixes from first live test-run:**
- `asyncio.to_thread` wrapping for coordinator LLM call and Qdrant retrieval (were blocking event loop)
- ZMQ `_extract_frames` returns `None` on bad frame count or unknown topic instead of raising
- `_create_dialogue_input` uses `DialogueInput.model_validate()` instead of stringify hack
- `forward_response` calls `flush_queue()` after queuing
- `MemoryRetrievalNode` returns `NodeStatus.PARTIAL` on empty results (not `SUCCESS`)
- `OpenRouterLLM._extract_response` guards against `content: null`
- Self-referential memory: `UserFactExtractionHandler` no longer writes to Qdrant inline.
  Facts are now written by `ConversationStorage` after the response is sent.
  `MessageAnalysisNode` description updated to reflect this.

**Parallel coordinator:**
- `build_select_nodes_tool` now takes `node_names: list[str]` instead of single `node_name`
- `Coordinator.select_node` → `select_nodes`, returns `list[str] | None`
- `_run_node_loop` uses `asyncio.gather` to execute each batch concurrently
- Coordinator prompt explains parallelism contract (only batch independent nodes)
- Typical turn now executes in 5 coordinator rounds instead of 11 sequential calls

**Dependency system:**
- `BaseNode.dependencies: list[str] = []` class attribute
- Each node declares prerequisites by class name
- `NodeRegistry.get_menu()` renders `requires: ...` per node

**Criticality guidance:**
- `BaseNode.min_criticality: float = 0.0` class attribute
- Each node declares its threshold against `need_urgency`
- `NodeRegistry.get_menu()` renders `min_criticality: X.X (label)` per node
- Nodes that always run: EmotionalState, MemoryRetrieval, MessageAnalysis, NeedsAnalysis,
  FormatAdvisor, PrimaryResponse (0.0)
- Nodes that skip for casual turns: MemoryEvaluation (0.2), MemoryAdvisor (0.2),
  ResponseStrategy (0.3), NeedsAdvisor (0.3), StrategyAdvisor (0.3)

**Rich node descriptions:**
- Every `get_description()` rewritten as a full decision guide: what the node writes,
  what the output contains, what broker fields it reads, what depends on it, skip conditions,
  pairing recommendations

**Timing/observability:**
- Log timestamps now include milliseconds (`%(msecs)03d`)
- `NodeRegistry.execute` measures per-node wall time, stamps `result.metadata["duration_ms"]`
- `_run_node_loop` logs `[coordinator] round N selection: Xms` and `[batch N] Xms total | ...`
- `_handle_request` logs turn total with per-node breakdown on completion

---

## Known design debt

**NeedsAdvisor and StrategyAdvisor are pass-throughs.** `NeedsAnalysisNode` produces
`context_summary` (natural language, LLM-generated). `NeedsAdvisorNode` just wraps it.
Same for `StrategyAdvisorNode` forwarding `system_prompt_addition`. The classifier nodes
are doing the advisor's job. The clean fix: classifier nodes produce structured outputs only
(scores, labels), advisor nodes own the LLM synthesis. Left as-is intentionally — it saves
two LLM calls per turn and the quality is acceptable for now.

**`emotional_state` in Qdrant is intentionally suspended — the design is unresolved.**
`ConversationStorage` outcomments it (`emotional_state = None`) because storing a per-turn
VAD snapshot alongside conversation entries doesn't carry enough useful signal. The VAD of a
single message tells you how the user felt when they said something, not how they responded
to what the companion said. Whether a strategy is working is a trajectory signal — it emerges
across multiple turns, not within one.

If emotional information belongs in long-term memory, the correct form is synthesised
observations ("user has been progressively more open over the last four conversations"),
not raw VAD scores. That form belongs to the character_state / observation layer, which
is not yet designed. Do not wire `emotional_state` through to `ConversationStorage` until
the reflection design is settled — the right structure will become clear from what the
reflection process needs to read.

**System prompts in all handlers are working stubs.** Quality is good enough for dev/testing
but none have been systematically tuned from real outputs.

**Coordinator overhead is ~900ms per call on Cerebras/gpt-oss-120b.** 6 rounds = ~5s of the
~12s total turn time. The prompt is large (full menu + history + state summary). Acceptable
for now but worth tracking as the menu grows.

**MemoryRetrievalNode cold-loads `all-MiniLM-L6-v2` on first call (~2.3s on CUDA).** Warm
calls are <100ms. Only affects the first turn after startup.

---

## What needs building — priority order

### 1. Observation/character_state layer

`RelationshipObservation` as a stored object in a dedicated `character_state` Qdrant collection:
- Natural language observation written by advisors during the turn
- Provenance: advisor name, timestamp, event context
- No session_id — just timestamp
- Retrieved at turn start to give the coordinator and advisors continuity across turns

Design the reflection process that reads accumulated observations and synthesises calibration
notes (what worked, what didn't, what patterns have emerged). Reflection runs as a scheduled
internal event, not during the turn.

### 2. Strategy transition logic informed by MI

The strategy roster needs transition logic, not just selection logic:
- Failure signatures per strategy (what withdrawal/resistance looks like)
- Natural transition paths (what to do when a strategy isn't landing)
- Resistance → roll back to reflective listening (hardcoded rule from MI)
- Premature problem-solving detection (righting reflex guard)
- Change talk vs. sustain talk detection in the turn

### 3. Crisis node (high-criticality friend response)

Runs when coordinator assesses high urgency. Not a script, not a hotline link — friend
response. Slow down, acknowledge, ask directly. The companion stays a friend through the
entire spectrum. High end of the criticality dial, not a mode switch.

### 4. Background reflection / detox process

Scheduled internal event. Reads accumulated observations since last run. Detects OCEAN drift.
Synthesises calibration notes. Stores in character_state.

### 5. Third-party book RAG (clinical + humanistic resources)

Two separate Qdrant collections (distinct from conversation memory):
- `clinical_resources` — MI manuals, CBT/DBT/ACT literature, psychology papers
- `humanistic_resources` — accessible psychology writing, self-help, identity/voice material

Ingestion pipeline: chunk source material (semantic/paragraph-level, not fixed-size),
embed with the same `EmbeddingService`, store with provenance metadata.

Query construction: queries into these collections should be derived from classifier outputs
(needs analysis + emotional state + strategy), NOT the raw user message. "MI techniques for
high belonging need, moderate urgency, sustain talk present" retrieves more precisely than
the user's words.

Copyright: needs careful handling — use open-access papers, licensed private copies,
or permissively licensed material.

Advisor nodes (MemoryAdvisor, future StrategyAdvisor with real synthesis) are the consumers.

### 6. External event sources

Infrastructure for injecting non-user events (news feeds, Reddit, scheduled check-ins) into
the event stream. The companion having a world beyond the user's messages is what makes it
feel like a person. Most external events produce no user-facing output — the coordinator
routes accordingly.

---

## Working components

See tests/ — 167 passing.

Coordinator-selectable nodes (@register_node, live in registry):
- EmotionalStateNode, MessageAnalysisNode — classifiers
- MemoryRetrievalNode, MemoryEvaluationNode — memory
- NeedsAnalysisNode, ResponseStrategyNode — analysis
- MemoryAdvisorNode, NeedsAdvisorNode, StrategyAdvisorNode, FormatAdvisorNode — advisors
- PrimaryResponseNode — generation

Infrastructure (orchestrator-owned, never coordinator-selectable):
- ConversationStorage — background persistence after response sent
- ZMQ send/receive — inline in orchestrator and ZMQHandler
