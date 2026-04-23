# Session Handoff — 2026-04-23

## Current state

128 passing tests. Codebase is clean. The main processing chain exists end-to-end but has not been run against real LLMs yet — the right machine isn't available.

Read these docs before touching anything:
- `docs/ADVISOR_PATTERN_CONCEPT.md` — critical architecture, read first
- `docs/ORCHESTRATOR.md` — event model and coordinator gap
- `docs/PSYCHOLOGY.md` — OCEAN, VAD, tonic/phasic, detox framework

---

## Architectural decisions settled in this session

### The companion is a friend, not a therapist

The psychology and therapy tools are a toolbox that activates proportionally to criticality — not a mode the companion operates in. Most interactions are casual friendship. The machinery is always present but mostly dormant. A cat picture and a grief disclosure both pass through the same system, just at completely different activation levels.

The criticality dial: from "send something light and connective" to "full advisory chain, this person needs careful support." Same system, different activation. The coordinator assesses criticality and decides how much of the pipeline to run. Knowing when NOT to deploy complexity is itself a form of intelligence. "Hi. What's up?" can be the correct output of a full advisory chain.

The tools are invisible to the user. They experience a friend who somehow always knows the right register. They don't see the machinery.

### The event model — continuous stream, not sessions

The orchestrator is an event processor. Events come from: user messages, scheduled internal tasks, external feeds (news, Reddit, anything), the companion's own proactive initiations, background process completions.

**session_id is removed.** The concept doesn't work. The companion's relationship with the user is a single continuous stream from first interaction to whenever. There is no session boundary. "Session" as a temporal unit falls apart entirely once you have external event sources — a Reddit post arriving at 3am has no session to belong to.

What session_id was trying to do:
- Group turns for reflection → replaced by "events since last reflection ran" (timestamp comparison)
- Within-conversation continuity → replaced by recency weight and elapsed time since last interaction
- Strategy inertia → handled by observations in character_state

**Elapsed time since last interaction is a first-class signal.** Two hours vs three days vs a week carry meaningfully different implications for re-engagement. The coordinator should know this. It's a continuous variable, not a binary.

The "sleep state" / reflection run is just a scheduled internal event — not a session boundary. Everything in the system is events with timestamps.

### External event sources are first-class

The companion can receive events from external feeds. A Reddit post can be an event emission. The companion might process /r/cats and decide to share something funny with no therapeutic agenda. It might process /r/worldnews and quietly file context. Most external events probably produce no user-facing output. The coordinator routes accordingly.

The companion having a world beyond the user's messages is what makes it feel like a person.

### Observations as language, not counters

Session state is a collection of natural language observations made by advisors and classifiers. No `deflection_count`, no `turns_on_current_strategy`, no numeric tallies of social dynamics.

Rigid counters are the wrong abstraction. "User deflected when companion raised social connection — three short responses then topic change" is more useful to a language model than `deflection_count: 2`. The context that makes the observation meaningful is exactly what a counter throws away.

**Capturing vs. interpreting:** during interactions, advisors only capture observations. They write what they noticed in plain language. The interpretation — what it means, what worked, what didn't — happens during the background reflection run (detox), not in the moment. Same as how a therapist observes during a session and reflects in supervision afterwards.

Observations stored in character_state Qdrant collection with:
- Provenance: which advisor, timestamp, event context
- Content: natural language
- No session_id — just timestamp

### Effectiveness knowledge emerges from reflection

"What worked, what didn't" is NOT recorded at interaction time. It's inferred by the reflection process reading accumulated observations across time and noticing patterns. "On three occasions across multiple interactions, cognitive reframing of self-criticism has been met with withdrawal. This user appears to need longer in the validation phase." This becomes a calibration note with high chrono_relevance, retrieved at future interaction start.

### Motivational Interviewing informs strategy transition logic

MI is the most directly applicable clinical framework for this system. Key concepts that map to implementation:

**Change talk vs. sustain talk** — detectable signal. Change talk (desire, ability, reasons, need, commitment) indicates readiness. Sustain talk (defending status quo, minimizing) indicates resistance. When sustain talk appears, validate and step back — do not push harder. This is one of the most robust findings in the literature.

**The righting reflex** — the impulse to give advice before the person is ready. This is the primary failure mode. The coordinator and strategy selection must be biased against premature problem-solving.

**OARS** (Open questions, Affirmations, Reflective listening, Summarizing) — the baseline, always appropriate. Strategies are built on top of this, not instead of it.

**Four processes** (Engaging → Focusing → Evoking → Planning) — these are relationship phases, not turn-level states. They inform what class of strategy is appropriate given how established the relationship is.

**Rolling with resistance** — when the user withdraws, reflect and validate, don't continue the strategy that triggered withdrawal.

These are hard rules derived from clinical evidence. They inform transition logic in the strategy/advisor layer.

### RAG for two distinct purposes

**Clinical resources** — MI manuals, CBT, DBT, ACT, psychology literature. The technique layer. Advisor nodes query this when making strategy decisions. "What does the literature say about working with grief-related ambivalence?" Grounds advisor outputs in professional evidence.

**Humanistic/identity resources** — self-help books, accessible psychology writing, potentially fiction, community discussions. The voice layer. Models what warm, honest, non-clinical friendship sounds like. Brené Brown, Viktor Frankl, Mark Manson, etc. Gives the companion an identity register beyond clinical technique.

Hard-defined strategy logic (rules derived from MI, CBT, etc.) + soft RAG'd guidance = reliable rails + contextual nuance.

---

## What needs building — priority order

### 1. Fix coordinator blindness (highest priority, lowest effort)

The coordinator currently sees: user message + nodes already run. It cannot see what those nodes produced. It's making routing decisions without knowing if emotional state was detected, if memories were retrieved, what urgency level was found.

Fix: inject `broker.get_state_summary()` into the coordinator prompt at each decision point. This method already exists on the broker. One prompt change. Coordinator immediately starts making informed routing decisions.

### 2. Design the observation/character_state layer

`RelationshipObservation` as a stored object in character_state Qdrant collection:
- Natural language observation
- Provenance: advisor name, timestamp
- No session_id

Design the reflection process that reads accumulated observations and synthesises calibration notes.

### 3. Build the memory advisor (first real advisor)

Reads retrieved memories + MemoryEvaluation reasoning + current event context.
Produces: natural language synthesis of what the companion knows about this person that's relevant right now + potency score.

This is the most important advisor. Character with a specific user is almost entirely built from memories.

### 4. Design AdvisorOutput and broker field

Settle the unsettled parts of the advisor pattern:
- `AdvisorOutput` dataclass: advisor name, advice (natural language), potency (0.0-1.0)
- `broker.advisor_outputs: list[AdvisorOutput]`
- How the primary LLM system prompt explains potency

### 5. Strategy transition logic informed by MI

The strategy roster needs transition logic, not just selection logic:
- Failure signatures per strategy (what withdrawal/resistance looks like)
- Natural transition paths (what to do when a strategy isn't landing)
- Resistance → roll back to reflective listening (hardcoded rule from MI)
- Premature problem-solving detection (righting reflex guard)

### 6. Criticality/urgency routing in the coordinator

The coordinator should assess event criticality early and route the pipeline accordingly. A cat picture shares very little of the same path as a crisis disclosure. Low criticality → skip most of the chain, go to primary response. High criticality → full chain.

### 7. Parallel Coordinator (after advisor pattern stabilises)

Multi-node selection per coordinator call. Independent nodes run concurrently via asyncio.gather.

### 8. Safety/crisis node

Runs before everything else at high criticality. Not a script, not a hotline link — friend response. Slow down, acknowledge, ask directly.

### 9. Background reflection / detox process

Scheduled internal event. Reads accumulated observations since last run. Detects OCEAN drift. Synthesises calibration notes. Stores in character_state.

### 10. External event sources

Infrastructure for injecting non-user events (news feeds, Reddit, scheduled check-ins) into the event stream. The companion having a world beyond the user's messages.

---

## Known technical issues

- `emotional_state` is suspended from broker — pass `emotional_state=None` in NeedsAnalysis and ResponseStrategy nodes for now. Role in advisor architecture not settled.
- `OpenRouterLLM` reads temperature/max_tokens from global settings singleton — model config is per-instance (fixed), generation params are still global (acceptable for now)
- `ReceiveMessageNode` was deleted — ZMQ receive is handled directly in orchestrator main loop
- `MemoryEvaluationHandler` exists and produces `(RAGDocument, MemoryEvaluation)` tuples with plain-language reasoning. ADVISOR_PATTERN_CONCEPT.md notes this is classifier-on-classifier but the reasoning field points toward the right direction. Leave as-is until memory advisor is built.
- System prompts in all handlers are working stubs — need tuning from real outputs

---

## Working components

See memory file for full list. Everything in `tests/` passes (128 tests).
