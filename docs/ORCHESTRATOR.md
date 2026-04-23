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

## The Critical Problem: The Coordinator Is Currently Blind

The coordinator's prompt currently contains:

```
User message: {message}
Nodes already executed this turn: {already_run_str}
```

That is all it sees. It knows the original message and which nodes have run. It does not know what those nodes produced.

This means the coordinator cannot adapt based on intermediate results. If `MessageAnalysisNode` detects acute grief, the coordinator cannot see that. If `MemoryRetrievalNode` returns nothing, the coordinator cannot see that. If `NeedsAnalysisHandler` returns urgency=0.9, the coordinator cannot see that.

The coordinator is making decisions with almost no information. It is flying blind after the first step.

**This is the most significant architectural gap in the current implementation.** The coordinator needs a summary of the broker's current state at each decision point — not the raw data, but enough signal to make informed routing decisions.

### What the Coordinator Needs to See

At minimum, after each node runs, the coordinator's prompt should include a structured summary of what is now known:

```
Current broker state:
  dialogue_input: populated ("I feel so alone today")
  emotional_state: grief=0.85, loneliness=0.70, valence=-0.7, dominance=0.2 [high distress]
  user_facts: 2 facts extracted
  retrieved_documents: 5 memories retrieved
  needs_analysis: belonging=0.8, meaning=0.6, urgency=0.7 [elevated]
  response_strategy: not yet run
  advisor_outputs: none yet
  primary_response: not yet generated
```

With this, the coordinator can reason: "Urgency is elevated and dominance is low — this person is distressed and feeling powerless. I should run the memory advisor before strategy, and I should not skip the strategy node."

Without this, the coordinator is essentially pattern-matching on node names and guessing.

---

## How the Broker Relates to the Coordinator

The `KnowledgeBroker` is the shared typed context that accumulates state across nodes. Every node reads from it and writes to it. The coordinator currently has indirect access to it (it's passed in) but only reads `metadata.execution_order` and `dialogue_input.content` from it.

The broker needs a method that produces a **coordinator-readable summary** — a concise representation of what fields are populated and with what signal strength. This is not the same as dumping the full broker contents; it's a distilled state description designed specifically for the coordinator's decision-making.

This summary should be:
- Concise enough to fit comfortably in the coordinator's context
- Informative enough to support routing decisions
- Structured so the coordinator can scan it quickly

The broker already has `get_execution_summary()` for metadata. It needs a parallel `get_state_summary()` for content.

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

## Session State (Not Yet Implemented)

The coordinator currently has no concept of session continuity. Each turn it starts with a fresh broker and no memory of what happened in previous turns of the same conversation.

This matters because within-session strategy consistency is part of the design (see `PROJECT_VISION.md`). The coordinator should know: "We established a reflective listening strategy three messages ago in this session — unless something has dramatically changed, maintain it."

Session state is not yet designed or implemented. It would need to be injected into the coordinator's prompt at the start of each turn. What form it takes is an open question.

---

## Current State vs Intended State

| | Current | Intended |
|---|---|---|
| Coordinator input | User message + nodes already run | User message + nodes run + broker state summary + session state |
| Coordinator decisions | Guesses at a reasonable order | Adapts based on what analysis found |
| Skip logic | None — runs nodes regardless of upstream results | Skips nodes when inputs are absent or irrelevant |
| Session awareness | None | Maintains within-session context |
| Broker summary | No method exists | `get_state_summary()` for coordinator consumption |

---

## What Not to Change Without Reading This

- Do not add new nodes without considering what coordinator-visible signal they produce
- Do not change the coordinator prompt without adding broker state visibility
- Do not assume the coordinator currently makes good routing decisions — it largely doesn't, because it can't
- The `_MAX_NODES_PER_REQUEST = 20` limit in `orchestrator.py` is a safety valve; a well-functioning coordinator should complete in 6–9 nodes for a normal interaction
