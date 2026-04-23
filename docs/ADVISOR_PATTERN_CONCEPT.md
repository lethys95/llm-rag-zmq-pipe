# Advisor Pattern Concept

This document captures an architectural pattern that emerged from reviewing the current pipeline. It is intentionally conceptual — some parts are settled, some are open questions. Read it before touching the memory evaluation layer, the broker, or the primary response handler.

---

## The Problem It Solves

The pipeline runs several analysis nodes before generating a response: emotional state, memory retrieval, needs analysis, response strategy. These produce structured data — scores, categories, classified values.

The question is: **what does the primary LLM actually do with that data?**

Currently, `PrimaryResponseHandler._format_analyzed_context` takes the broker's contents and formats them as a context string. It dumps emotion scores, need names, user facts. The primary LLM gets something like:

```
Emotions: grief: 0.85, loneliness: 0.50
Primary needs: meaning, belonging
Unmet: meaning, belonging
Urgency: 0.50
What I know about you:
- user lost their mother last year
- user feels disconnected from friends
```

This is **classifier output handed directly to a generative model**. The primary LLM has to figure out what to do with grief=0.85. That's not its job. Its job is to generate a response that feels human. Giving it raw scores is the wrong interface.

---

## Classifiers vs Advisors

**Classifiers** take input and produce structured data. They answer "what is happening":
- `EmotionalStateHandler` → VAD scores, categorical emotion scores
- `NeedsAnalysisHandler` → Maslow dimension scores, urgency, persistence
- `ResponseStrategyHandler` → approach, tone, needs_focus
- Memory retrieval + decay → ranked RAGDocuments

These are valuable. They do real work. They should continue to exist.

**Advisors** consume classifier outputs (and raw context) and produce **natural language guidance** for the primary LLM. They answer "what should you do about it":

Instead of:
```
grief: 0.85, belonging: 0.7, meaning: 0.6
```

An advisor produces:
```
This person has been grieving since their mother died last year. Loneliness has come up
repeatedly across sessions — it's not situational. They tend to shut down when offered
practical advice. Right now they appear to be reaching out for connection rather than
solutions. Prioritise being present over being useful.
```

The primary LLM can use that. It cannot meaningfully use a grief score.

**The flow is: Classifiers → Advisors → Primary LLM.**

Classifiers feed advisors. Advisors feed the primary LLM. The primary LLM never sees raw classifier values.

---

## Potency

Multiple advisors may run in a given turn. Not all advice is equally relevant in every moment.

Each advisor output carries a **potency score** (0.0–1.0) alongside its natural language advice. Potency signals to the primary LLM how much weight to put on this advice right now.

Examples:
- Memory advisor has **high potency** when retrieved memories are recent and directly relevant to what's being said. **Low potency** when memories are old, tangentially related, or sparse.
- A detox/calibration advisor has **high potency** when OCEAN drift has been detected in recent sessions. **Low potency** when the companion has been behaving well.
- A needs advisor has **high potency** when urgency is high. **Low potency** when the message is casual and no strong needs are activated.

Potency is not a confidence score — it's a relevance-to-this-moment score. An advisor can be very confident in its output and still have low potency because the situation doesn't call for that kind of guidance.

**How potency is determined** is not fully settled. Some advisors can compute it from classifier outputs (e.g. needs urgency → needs advisor potency). Others may require a brief LLM assessment. This should be figured out per-advisor when they are built.

---

## What Advisors Exist (Anticipated)

These are the advisors that seem necessary based on the project's goals. Not all are built yet. None should be built without understanding this document first.

### Memory Advisor

**Input**: retrieved memories (from MemoryRetrievalNode), current message, emotional state
**Output**: natural language synthesis of what the companion knows about this person that is relevant right now + potency

This is the most important advisor. The companion's character with a specific user is almost entirely built from memories — what they've shared, what they've been through, what the companion has witnessed. The memory advisor is what makes the companion feel like it *knows* the person rather than just responding to the current message.

It does **not** re-classify memories. It does not produce relevance scores or chrono_relevance values. It reads the memories and synthesises them into guidance the primary LLM can act on.

**Note on the failed MemoryEvaluationNode**: An earlier attempt at this built a node that produced `(RAGDocument, MemoryEvaluation)` tuples with relevance and chrono_relevance scores. This was wrong — it added a classifier on top of a classifier. The `MemoryEvaluation` model in `analysis.py` still exists in stripped-down form (`relevance`, `chrono_relevance`, `reasoning`) but the node that produced it has been removed. If a memory advisor is built, it should produce natural language, not that structure.

### Needs Advisor

**Input**: NeedsAnalysis output (Maslow scores, urgency, persistence), emotional state
**Output**: guidance on what the user appears to need and how the companion should orient

Note: `NeedsAnalysis.context_summary` is already close to this — it's a natural language summary. The needs advisor may be as simple as wrapping that summary with potency derived from urgency.

### Strategy Advisor

**Input**: ResponseStrategy output
**Output**: guidance on how to respond (approach, tone, what to avoid)

Note: `ResponseStrategy.system_prompt_addition` already does some of this. The question is whether this remains a direct system prompt addition or becomes a proper advisor output. The distinction matters because advisor outputs and system prompt content have different roles (see below).

### Detox / Calibration Advisor

**Input**: OCEAN drift detection results (when detox runs in background)
**Output**: self-correction notes — "in recent sessions I have been over-validating catastrophising; this session I should offer gentle challenge when appropriate"

This advisor only has content when detox has run and found something. Its potency is determined by how significant the detected drift is. When nothing has drifted, this advisor produces nothing.

See `PSYCHOLOGY.md` for the full detox/OCEAN framework.

### Trust Advisor (Future)

**Input**: relationship history, interaction count, patterns of openness/withdrawal
**Output**: guidance on appropriate depth and intimacy for this interaction

The companion should not be equally open with a first-time user and someone it has spoken with for months. Trust advisor governs this gradually. The concept is described in `BRAINSTORMING.md` ("trust analysis — gradual push towards the center of the target, with stranger on the outside, loved one dead center").

The `TrustAnalysis` model that previously existed in `models/memory.py` was removed because it contained hallucinated fields with no defined computation (`positive_interactions`, `negative_interactions` with no arbiter). Trust advisor should be designed from scratch when the time comes, starting from what signals are actually observable.

---

## Advisor Output Structure

Not fully settled. What we know:

Each advisor produces:
- `advice: str` — natural language guidance
- `potency: float` — 0.0 to 1.0, how much weight to give this advice this turn

What's unsettled:
- Whether advisor outputs are stored as a typed list on the broker (`list[AdvisorOutput]`) or as separate named fields
- Whether the advisor name/type is included in the output (useful for transparency and debugging)
- Exactly how the primary LLM is instructed to interpret potency (the system prompt needs to explain the concept)

A typed `AdvisorOutput` dataclass is the likely direction:

```python
@dataclass
class AdvisorOutput:
    advisor: str        # name, for transparency
    advice: str         # natural language guidance
    potency: float      # 0.0–1.0
```

The broker would hold `advisor_outputs: list[AdvisorOutput]`, populated by whichever advisor nodes run this turn.

---

## Advisor Outputs vs System Prompt

These are different things and should not be conflated.

**System prompt** defines the companion's persona — who it is, what it values, how it speaks. This should be relatively stable and lean. It does not contain per-turn analysis. The companion's character is not defined by a wall of system prompt text — it emerges from memory (see below).

**Advisor outputs** are per-turn guidance — specific to this conversation, this user, this moment. They inform how the companion responds right now. They are not permanent traits.

Currently `ResponseStrategy.system_prompt_addition` blurs this line by appending per-turn strategy into the system prompt. This works as a stopgap but is architecturally imprecise. Longer term, strategy guidance should come through the advisor layer, not the system prompt.

---

## Character Is Memory

A core design principle: **the companion's character with a specific user is almost entirely built from memories**.

There is no hardcoded backstory. There is no static personality description that defines who the companion is to this particular person. What the companion "knows", what it references, what it notices — all of it comes from what has been stored and retrieved from past interactions.

This is why the memory advisor is the most important advisor. It is the mechanism by which character is expressed. A companion with no memories has no character yet. A companion with months of accumulated memories has a rich, specific character that is unique to this relationship.

The system prompt should not try to compensate for absent memories by defining character in text. It should define the companion's stable dispositional traits (curiosity, warmth, resistance to sycophancy — the OCEAN scaffold described in `PSYCHOLOGY.md`) and leave everything else to memory.

---

## What Is Settled

- Classifiers should not pass raw scores to the primary LLM
- The interface between the analysis pipeline and the primary LLM should be natural language advice + potency
- Multiple advisors, not one monolithic context block
- Potency is a per-turn relevance signal, not a confidence score
- Character emerges from memory, not system prompt text
- `MemoryEvaluation` as a classifier-on-classifier is the wrong pattern
- `TrustAnalysis` from `models/memory.py` was removed — trust advisor needs to be designed from scratch

## What Is Not Settled

- Exact `AdvisorOutput` data structure and broker field
- Whether `NeedsAnalysis.context_summary` and `ResponseStrategy.system_prompt_addition` are refactored into the advisor pattern or left as-is temporarily
- Potency computation method per advisor
- How the primary LLM system prompt explains the advisor concept to the model
- Whether all advisors run every turn or the coordinator selects which ones to run
- The line between what belongs in the system prompt vs advisor outputs
