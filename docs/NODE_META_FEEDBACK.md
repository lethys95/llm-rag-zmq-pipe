# Node Meta Feedback Analysis

This document describes a planned background system for evaluating how well each LLM-based node understands and can apply its own instructions. It is a tool for prompt engineering — it tells us where the system prompts are unclear, misaligned, or insufficient for the inputs they receive.

This is a far-future feature. It should not be built until the core pipeline is stable. Document exists so the concept is not lost.

---

## The Problem It Solves

Every LLM-based node in the pipeline operates on a system prompt written by a human. That system prompt defines the task — what to analyse, what to return, how to think about it. The quality of the node's output is directly coupled to how well the LLM comprehends those instructions in the context of a real input.

We have no feedback loop on this. A node might produce plausible-looking output that is systematically wrong because its system prompt is ambiguous, poorly scoped, or breaks down on certain input types. We would not know this without manually reviewing outputs.

NodeMetaFeedback creates that feedback loop automatically, during idle time, without affecting live performance.

---

## How It Works

### Step 1 — Snapshot Collection

During normal operation, a random sample of node executions are captured as snapshots. Not every execution — a configurable sampling rate. Each snapshot records:

- Which node ran
- The system prompt used (at time of execution)
- The exact input passed to the LLM
- The raw output produced by the LLM
- Timestamp and any relevant broker metadata (emotional state, needs urgency, etc. — enough to understand the context)

Snapshots are stored in SQLite. They are write-once — a record of what happened, not modified later.

### Step 2 — Replay During Inactivity

During idle time (no active user conversation), the system randomly selects stored snapshots and replays them — but with a wrapper that intercepts the original instructions and replaces the task.

The structure of the replayed prompt is:

```
You are to ignore the section below marked [ORIGINAL INSTRUCTIONS].
You are instead an evaluator.

Your task: assess how well the system prompt in [ORIGINAL INSTRUCTIONS] enables a 
language model to understand and complete its described task for the given input.

Evaluate two things:
1. Instruction clarity — are the instructions unambiguous and complete for this input?
2. Task comprehension — could a language model reliably apply these instructions to 
   this specific input and produce a correct, well-formed result?

Score each from 0.0 to 10.0:
  0.0 = the instructions provide no useful guidance for this input
  5.0 = partial comprehension, output would likely be inconsistent
  10.0 = instructions are fully clear and applicable to this input

Respond ONLY with this JSON structure:
{
  "instruction_clarity": 0.0,
  "task_comprehension": 0.0,
  "clarity_reasoning": "...",
  "comprehension_reasoning": "...",
  "failure_modes": ["..."],
  "suggested_improvement": "..."
}

failure_modes: list specific ways the instructions might cause incorrect or inconsistent 
output for inputs like this one. Empty list if none identified.

suggested_improvement: one concrete change to the system prompt that would improve 
performance on this input type. Leave empty string if none needed.

--- [ORIGINAL INSTRUCTIONS] ---

{original_system_prompt}

--- [ORIGINAL INPUT] ---

{original_input}

--- [ORIGINAL OUTPUT — for reference only] ---

{original_output}
```

The original output is included so the evaluator can assess whether what was produced reflects genuine comprehension or lucky pattern-matching.

### Step 3 — Store Evaluation Results

Evaluation results are stored in SQLite alongside the original snapshot, linked by snapshot ID. They are static records — not used to automatically modify anything. They exist for human analysis.

---

## What the Scores Mean

**Instruction clarity (0–10):** How well-written is the system prompt for this class of input? A low score means the instructions are ambiguous, contradictory, or missing information the LLM needs to complete the task correctly. This is a critique of the prompt, not the input.

**Task comprehension (0–10):** Could a model reliably apply these instructions to this specific input? A low score here with high clarity means the input is genuinely hard — edge cases, ambiguous user language, unusual emotional content. A low score with low clarity means the prompt is failing the input.

The combination is what matters:
- High clarity, low comprehension → the input is an edge case the prompt doesn't cover. Extend the prompt.
- Low clarity, low comprehension → the prompt is fundamentally unclear. Rewrite it.
- Low clarity, high comprehension → the model is compensating for poor instructions. Clarify for consistency.
- High clarity, high comprehension → no action needed for this snapshot.

---

## What Nodes This Applies To

Any node that calls an LLM with a system prompt:

- `EmotionalStateHandler`
- `UserFactExtractionHandler`
- `NeedsAnalysisHandler`
- `ResponseStrategyHandler`
- `MemoryEvaluationHandler` (the advisor version)
- `ContextInterpreterHandler`
- Any future advisor nodes

It does not apply to algorithmic nodes (memory decay, embedding, storage) — those have no LLM comprehension to evaluate.

It does not apply to the coordinator or the primary LLM. The coordinator's job is routing, not analysis — evaluation there is different. The primary LLM's quality is evaluated differently (through user outcomes, detox/OCEAN drift, etc.).

---

## Timing and Sampling

Runs during inactivity, similar to the detox process. Should not compete with live requests for LLM capacity.

Sampling rate for snapshot collection is configurable. A low rate (e.g. 5–10% of executions) is sufficient — the goal is a statistically useful sample over time, not comprehensive logging.

Replay rate during idle time is also configurable. The system should not exhaust its idle time running evaluations when detox and other background processes also need to run.

---

## Storage Schema (Indicative)

Two SQLite tables:

**`node_snapshots`**
- `id` (primary key)
- `node_name`
- `system_prompt` (text of prompt at time of execution)
- `input` (the exact string passed to the LLM)
- `output` (the raw LLM response)
- `timestamp`
- `context_metadata` (JSON — relevant broker state at time of execution)
- `evaluated` (boolean — has this snapshot been replayed yet)

**`node_evaluations`**
- `id` (primary key)
- `snapshot_id` (foreign key → node_snapshots)
- `instruction_clarity` (float)
- `task_comprehension` (float)
- `clarity_reasoning` (text)
- `comprehension_reasoning` (text)
- `failure_modes` (JSON array of strings)
- `suggested_improvement` (text)
- `evaluated_at` (timestamp)
- `evaluator_model` (which model ran the evaluation)

---

## Open Questions

- Should the evaluator be the same model that originally ran the node, or always a specific evaluation model?
- Should snapshots be taken uniformly across all nodes, or weighted toward nodes with lower historical scores?
- At what volume of evaluations does the SQLite data become actionable? Some minimum sample per node per prompt version is needed before patterns emerge.
- When the system prompt is updated, old evaluations against the previous version are no longer relevant. How is prompt versioning tracked?
- Should there be a simple dashboard or query for surfacing low-scoring nodes, or is raw SQLite access sufficient for now?
