# Orchestrator & Coordinator

This document describes the orchestration system — what it is, what it's supposed to do, and the critical gap between what it currently does and what it needs to do.

Read this before touching `src/orchestrator.py`, `src/nodes/orchestration/coordinator.py`, or `src/nodes/orchestration/knowledge_broker.py`.

---

## The Event Model

The orchestrator is not a request-response handler. It is an **event processor**.

A user message is one type of event, but it is not the only type. Events include:

- A user sending a message (the common case)
- A scheduled check-in triggered by the chrono system
- An idle-time reflection completing (detox, OCEAN recalibration)
- The companion deciding to initiate contact proactively
- An internal state update with no user-facing output

The appropriate response to an event is not always a primary response to the user. Sometimes the right outcome is an internal state change only. Sometimes it is a proactive message the companion initiates. **Sometimes the right choice is to do nothing.**

This "choice not to act" is deliberate and important. A companion that always responds is a chatbot. A companion that exercises judgement about when to speak and when to stay quiet is something closer to a person. The coordinator is responsible for making that judgement.

"Complete" does not mean "a response was sent." It means "this event has been handled appropriately" — which may or may not have involved sending anything.

This has implications for how nodes are designed: not every node produces user-facing output, and the workflow should not assume it always ends at a primary response.

---

## Two Distinct Things

**Orchestrator** (`src/orchestrator.py`) is the outer shell. It owns the ZMQ connection, runs the main loop, receives events, creates a fresh `KnowledgeBroker` per event, and drives the node execution loop. It is infrastructure.

**Coordinator** (`src/nodes/orchestration/coordinator.py`) is an LLM — specifically the worker LLM — that decides which node to run next at each step of an event. It is called repeatedly by the orchestrator until it signals completion. It is the intelligence behind node selection and the judgement behind inaction.

The distinction matters: the orchestrator is a runtime loop, the coordinator is a decision-maker. Problems with *what gets run* are coordinator problems. Problems with *how the loop runs* are orchestrator problems.

---

## What the Coordinator Is Supposed to Do

The coordinator's job is not to follow a fixed pipeline. It is to make **adaptive, context-sensitive decisions** about which node should run next given everything known so far.

This is why an LLM is used instead of a hardcoded sequence. A fixed pipeline like:

```
MessageAnalysis → MemoryRetrieval → NeedsAnalysis → ResponseStrategy → PrimaryResponse → Store → Forward
```

runs the same nodes in the same order regardless of what the user said. An LLM coordinator can reason:

- "The user's message is a greeting with no emotional content — skip NeedsAnalysis and go straight to PrimaryResponse"
- "EmotionalStateHandler returned crisis-level distress — the needs analysis should run before strategy, and the strategy node needs to know this is urgent"
- "MemoryRetrievalNode returned no results — skip MemoryEvaluationNode, there's nothing to evaluate"
- "The user mentioned something new about their family — UserFactExtraction should run before storing"

The coordinator can also route differently based on what has already run and what it found. This is the entire point of LLM-based orchestration.

---

## Coordinator Visibility into Broker State — Implemented

The coordinator's prompt includes three things at each decision round:

- The original event message
- The nodes already run this turn
- A structured summary of everything the broker currently holds (`broker.get_state_summary()`)

The state summary covers: emotional state (VAD + summary), user facts extracted, retrieved and evaluated memories, needs analysis (primary needs, urgency, context summary), response strategy, advisor outputs, and whether the primary response has been generated.

This lets the coordinator reason about intermediate results — if `MemoryRetrievalNode` returned nothing, the summary says so, and the coordinator can skip `MemoryEvaluationNode`. If `NeedsAnalysisNode` returned urgency=0.9, the summary reflects that, and the coordinator can activate the full advisor chain.

`KnowledgeBroker.get_state_summary()` produces this output. `Coordinator._build_prompt()` injects it. Both are in the current codebase.

---

## How the Broker Relates to the Coordinator

The `KnowledgeBroker` is the shared typed context that accumulates state across nodes. Every node reads from it and writes to it. The coordinator receives a distilled snapshot of it at every decision round via `broker.get_state_summary()`.

`get_state_summary()` returns a concise multi-line string describing what each broker field holds — populated fields include their signal values (e.g. urgency score, response character count), unpopulated fields are noted as "not yet run". It is designed to be scanned quickly and contains enough signal for routing decisions without dumping raw data.

`get_execution_summary()` is the parallel method for metadata — which nodes ran, which failed, timings.

---

## Node Selection Logic (What the Coordinator Should Reason About)

These are the kinds of decisions the coordinator should be able to make with full broker visibility:

**Skip decisions:**
- No retrieved memories → skip memory advisor
- No emotional state → skip needs analysis (insufficient input)
- No needs analysis → skip response strategy (insufficient input)
- Primary response already generated → skip to forward/store

**Escalation decisions:**
- High urgency + low dominance → flag for different strategy path
- Missing critical nodes that downstream nodes depend on → run them first

**Ordering decisions:**
- MessageAnalysis should nearly always run first
- MemoryRetrieval should run before MemoryAdvisor (obvious dependency)
- NeedsAnalysis should run before ResponseStrategy (obvious dependency)
- PrimaryResponse should run after all advisors

**Termination decisions:**
- All necessary nodes have run and primary response is generated → "complete"
- A node failed in a way that makes continuation meaningless → "complete" with error logged

The coordinator should understand these dependencies, not have them hardcoded. The node descriptions (from `get_description()`) and the broker state summary together should be enough for it to reason correctly.

---

## Criticality and Proportional Activation

The companion is a friend first. Most interactions are casual — a cat picture, a quick check-in, a light observation. The psychology machinery exists for when it is needed, not as a constant overhead applied to every message.

The criticality dial runs from "send something light and connective" to "full advisory chain, this person needs careful support." The same system handles both ends. Knowing when NOT to deploy complexity is itself a form of intelligence. "Hi. What's up?" can be the correct output of a full advisory chain.

The coordinator is responsible for reading the turn and activating accordingly. After `NeedsAnalysisNode` returns urgency=0.05 with no primary needs, the coordinator should skip ResponseStrategyNode, MemoryEvaluationNode, and the advisor chain — not because they are disabled, but because they add nothing. The node descriptions and `min_criticality` field on each node communicate when they are warranted.

---

## Crisis Response Is a Gradient, Not a Mode Switch

Current AI chatbots implement a binary: detect distress → drop warmth → output a hotline link. This has the opposite of the intended effect. When the companion becomes cold and clinical at the moment the user is most vulnerable, the user learns not to be honest with it.

There is no "crisis mode." There is a single continuous dial. At the high end, the companion responds with more presence, slower pace, more directness — but the character never disappears. The friend doesn't become a pamphlet. Warmth is not a feature that gets suspended when things get serious. Only intensity changes.

The crisis node (not yet built) is the high end of that dial, not a separate mode.

---

## The Event Model and Session Continuity

### session_id Is Removed

`session_id` was removed from the data model. The concept does not hold. The companion's relationship with the user is a single continuous stream from first interaction onward. There is no meaningful session boundary.

The concept breaks entirely once external event sources exist — a Reddit post arriving at 3am has no session to belong to. A companion-initiated check-in has no session. Idle-time reflection has no session.

What `session_id` was trying to do, and what replaced it:
- **Group turns for reflection** → replaced by timestamp comparison ("events since last reflection ran")
- **Within-conversation continuity** → replaced by recency weighting and elapsed time since last interaction
- **Strategy inertia** → handled by observations in character_state (not yet built)

### Elapsed Time Since Last Interaction Is a First-Class Signal

Two hours vs three days vs a week carry meaningfully different implications for re-engagement. The companion approaching a user who went quiet for a week is a different situation from continuing a conversation from an hour ago. This is a continuous variable injected into the broker as `idle_time_minutes`, not a binary "new session" flag.

---

## Observations as Language, Not Counters

Session state is a collection of natural language observations written by advisors and classifiers. There are no `deflection_count` fields, no `turns_on_current_strategy` counters, no numeric tallies of social dynamics.

Rigid counters are the wrong abstraction. "User deflected when companion raised social connection — three short responses then topic change" is more useful to a language model than `deflection_count: 2`. The context that makes an observation meaningful is exactly what a counter throws away.

### Capturing vs Interpreting

During interactions, advisors only **capture observations** — they write what they noticed in plain language. They do not interpret. The interpretation — what it means, what worked, what didn't — happens during the background reflection run (detox), not in the moment.

This mirrors how a therapist observes during a session and reflects in supervision afterwards. Attempting to interpret in real time produces premature conclusions on insufficient data. Let the interaction happen. Let observations accumulate. Interpret from a distance.

Observations are stored in the `character_state` Qdrant collection with:
- Provenance: which advisor, timestamp, event context
- Content: natural language
- No `session_id` — just timestamp

### Effectiveness Knowledge Emerges from Reflection

"What worked, what didn't" is **not** recorded at interaction time. It is inferred by the reflection process reading accumulated observations across multiple interactions and noticing patterns. "On three occasions across multiple interactions, cognitive reframing of self-criticism has been met with withdrawal. This user appears to need a longer validation phase." This becomes a calibration note with high `chrono_relevance`, retrieved at future interaction start.

Recording effectiveness inline produces noise. The signal only becomes visible across time.

---

## Session State (Not Yet Implemented)

The coordinator currently has no concept of session continuity. Each turn it starts with a fresh broker and no memory of what happened in previous turns of the same conversation.

This matters because within-session strategy consistency is part of the design (see `PROJECT_VISION.md`). The coordinator should know: "We established a reflective listening strategy three messages ago in this session — unless something has dramatically changed, maintain it."

Session state is not yet designed or implemented. It would need to be injected into the coordinator's prompt at the start of each turn. What form it takes is an open question.

---

## Current State vs Intended State

| | Current | Intended |
|---|---|---|
| Coordinator input | User message + nodes already run + broker state summary | User message + nodes run + broker state summary + session state |
| Coordinator decisions | Adapts based on what analysis found | Same, plus strategy continuity from character_state |
| Skip logic | Guided by node descriptions and broker state | Same |
| Session awareness | None — fresh broker per turn | Within-session observations via character_state layer |
| Broker summary | `get_state_summary()` implemented and wired | Same |

The remaining gap is **session awareness** — the coordinator sees conversation history but has no within-session observations about what strategies were tried and how they landed. That requires the character_state / observation layer (not yet built).

---

## What Not to Change Without Reading This

- Do not add new nodes without considering what coordinator-visible signal they produce
- Do not change `get_state_summary()` without checking that the coordinator's routing logic still works with the new output
- The `_MAX_NODES_PER_REQUEST = 20` limit in `orchestrator.py` is a safety valve; a well-functioning coordinator should complete in 6–9 nodes for a normal interaction
