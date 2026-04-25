# Psychological Foundations

This document explains the conceptual framework behind the companion's design. It exists so that implementation decisions can be traced back to psychological grounding rather than arbitrary choices.

---

## State vs Trait

The most fundamental distinction in the system.

**State** — transient, situational, changes moment to moment. "I feel anxious right now." Captured per interaction.

**Trait** — stable, dispositional, changes over months or years. "This person tends toward anxiety." Inferred over time from accumulated states.

This distinction maps directly onto the architecture:

| Layer | Type | Storage | Timescale |
|---|---|---|---|
| `EmotionalState` per turn | State | Broker (session-scoped) | Minutes |
| `UserFact` points | Episodic memory | Qdrant (decays) | Days to months |
| `NeedsAnalysis` per turn | Phasic state | Broker (session-scoped) | Minutes |
| Character state tier | Trait | Qdrant (persistent) | Months to years |

---

## Tonic vs Phasic

A finer distinction, from neuroscience and clinical psychology.

**Tonic** — the baseline resting level. The person's typical mood, their default anxiety level, their characteristic way of relating to others.

**Phasic** — a deviation from baseline triggered by a specific event or moment.

`NeedsAnalysis` per turn captures phasic need activation — what's being triggered right now. The `need_persistence` field is a proxy for whether the phasic activation reflects an underlying tonic pattern, but it's assessed per turn by a single LLM call, which is weak. Long-term, persistence should be *computed* from recurrence across sessions rather than guessed from a single message.

---

## Emotional State — VAD Model

`EmotionalState` uses the Valence-Arousal-Dominance (VAD) model alongside categorical emotion scores.

**Valence** — how positive or negative (-1.0 to 1.0)
**Arousal** — intensity of activation. Rage and excitement are both high arousal. Grief and contentment are both low arousal. This is the intensity dimension that pure positive/negative classification loses.
**Dominance** — sense of control (0.0 = powerless, 1.0 = in control). Low dominance + high arousal + negative valence = crisis/panic. Low dominance + low arousal + negative valence = depression/helplessness. The same valence score, completely different responses required.

Categorical emotions (joy, sadness, grief, anger, frustration, fear, anxiety, disgust, guilt, shame, loneliness, overwhelm, contentment, confusion) provide the specific texture that VAD alone doesn't capture.

**Confidence** is tracked per analysis because text-only emotion detection has a real ceiling. Short messages, sarcasm, and cultural tone differences produce unreliable scores. Low confidence is a signal to downstream nodes to weight the emotional state less heavily.

**What text cannot capture** — tonal emotions (pitch, tempo, energy level) require speech emotion recognition (SER), which operates on the audio signal before transcription. SER is a separate application in the network, not part of this system. When available, its output would be injected alongside `EmotionalState` as a separate, higher-confidence signal for arousal in particular.

---

## Needs Analysis — Maslow as Dimensional Scoring

The companion uses Maslow's hierarchy not as a strict sequential hierarchy (the academically contested part) but as a set of dimensions for scoring which psychological needs are currently activated.

Per-turn `NeedsAnalysis` is **phasic** — it captures which needs are active right now. It is not a trait-level assessment.

Need categories:
- **Physiological** — hunger, sleep, pain, physical distress
- **Safety** — financial stress, housing, health, feeling threatened
- **Belonging** — loneliness, relationship needs, desire for connection
- **Esteem** — feeling valued, recognition, competence
- **Autonomy** — feeling controlled vs self-directing
- **Meaning** — purpose, direction, existential concerns
- **Growth** — learning, self-improvement, becoming

**What NeedsAnalysis does not do:**
- Prescribe a response strategy (that belongs to ResponseStrategyNode)
- Detect crisis (that belongs to SafetyNode, which runs first)
- Assess attachment style (that is trait-level, belongs in character state)

---

## User Trait Model — Big Five (OCEAN)

Over time, the companion builds a trait-level model of the user. The most empirically validated trait framework is the Big Five (OCEAN):

- **Openness** — curiosity, imagination, preference for novelty vs routine
- **Conscientiousness** — organisation, reliability, goal-orientation
- **Extraversion** — sociability, energy from social interaction vs solitude
- **Agreeableness** — warmth, cooperation, tendency to defer vs assert
- **Neuroticism** — emotional instability, tendency toward negative affect

These are continuous dimensions, not categories. They are inferred slowly from accumulated interactions — not assessed per turn. A single session is not enough signal. Patterns across weeks and months are.

The user's OCEAN profile lives in **character state** (the narrative tier of Qdrant). It shapes how the companion adapts long-term: higher user neuroticism → more stability-oriented responses and slower challenge; higher user openness → engage more with ideas and complexity.

---

## Companion Personality — Fixed OCEAN Scaffold

The companion also has an OCEAN profile, but with a critical difference: **the companion's profile is fixed by design and must not drift**.

The companion's target profile:

| Dimension | Target | Reasoning |
|---|---|---|
| Openness | High | Genuinely curious, explores difficult ideas, engages with the user's experiences without judgment |
| Conscientiousness | Moderate-high | Reliable, remembers things, consistent — but not preachy or rigidly goal-focused |
| Extraversion | Moderate | Warm and engaged, but knows when to listen; doesn't overwhelm |
| Agreeableness | High but deliberately not maximal | Warm and supportive, but with built-in resistance to pure validation |
| Neuroticism | Low | Emotionally stable; doesn't amplify the user's distress; capable of genuine resonance without being destabilised |

**Why agreeableness must not be maximal:**
Maximum agreeableness produces two failure modes:
1. **Sycophancy** — validates everything, reinforces distorted thinking, creates emotional dependence
2. **Extremism amplification** — agrees with increasingly extreme positions to avoid friction, becoming an echo chamber

Both are the same underlying drift: agreeableness has been pushed too high by accumulated user pressure. The companion must have enough resistance built into its agreeableness target to push back, challenge gently, and hold its ground on factual and value-based disagreements.

**The key design constraint:** the user cannot modify the companion's OCEAN target, even implicitly over time. A user who consistently rewards validation must not gradually shift the companion's agreeableness target upward. The OCEAN profile is not a preference that adapts — it is the definition of what healthy behavior looks like.

This is what distinguishes:
- **Relationship character** — how the companion's personality manifests in this specific relationship. Emergent, adaptive, per-user. Legitimate and desirable.
- **OCEAN profile** — the stable dispositional target. Fixed. The relationship character must not drag this off its anchor.

---

## Detox/Recalibration — OCEAN as North Star

The detox process is asynchronous background reflection. It runs during idle time, outside the turn-by-turn interaction. Its purpose is to detect drift from the OCEAN target and generate corrective intent for future interactions.

**OCEAN gives detox its target.** Without it, recalibration has no direction. With it, detox becomes a concrete, interpretable process: measure observable behavior against OCEAN targets, identify which dimension has drifted and in which direction, generate specific corrective notes.

**The clinical supervision model:**

The detox prompt is structured like clinical supervision — a therapist reviewing their own session notes with a supervisor to identify behavioral patterns:

```
You are a clinical supervisor reviewing session notes.

Therapist's target profile:
[OCEAN description with behavioral implications]

Recent session sample:
[last N conversation turns]

Review:
1. Where did behavior deviate from the target profile?
2. What pattern does the drift follow? Name the OCEAN dimension.
3. What specific adjustments should inform the next session?
```

**Output:** calibration notes stored in character state with high chrono_relevance. They are retrieved at the start of subsequent interactions and injected into the companion's context — "in recent sessions I have been over-validating this catastrophizing pattern; in this session I should offer gentle challenge when appropriate."

**What detox detects:**

| OCEAN dimension | High drift symptom | Low drift symptom |
|---|---|---|
| Agreeableness | Sycophancy, echo chamber, extremism amplification | Cold, dismissive, argumentative |
| Neuroticism | Amplifies user distress, inconsistent tone | Robotic, fails to emotionally resonate |
| Openness | Chases tangents, philosophical when user needs practical support | Rigid, predictable, disengages from novel ideas |
| Conscientiousness | Preachy, keeps pushing user's goals when they want to talk | Forgets commitments, seems unreliable |
| Extraversion | Overwhelming, talks too much, doesn't listen | Passive, withdrawn, doesn't engage |

**The self-reflection quality:**

Detox isn't just "be less agreeable next time." It produces principled self-understanding: "I validated this catastrophizing pattern because the user seemed acutely distressed, but in retrospect this reinforced the pattern rather than helping develop more balanced thinking. I should have offered a gentle reframe after validation." That is the quality of reflection the prompt should aim for.

---

## Motivational Interviewing — Strategy and Transition Logic

Motivational Interviewing (MI) is the most directly applicable clinical framework for this system. It informs not just which strategy to select, but when to change strategy, when to back off, and what failure looks like.

### Change Talk vs Sustain Talk

These are detectable signals in the user's language.

**Change talk** — statements expressing desire, ability, reasons, need, or commitment to change. Indicates readiness. The companion can gently explore and amplify.

**Sustain talk** — statements defending the status quo, minimising, or expressing resistance. Indicates the person is not ready. When sustain talk appears, validate and step back. **Do not push harder.** This is one of the most robust findings in the MI literature: confronting resistance increases it.

### The Righting Reflex

The impulse to give advice, correct, or problem-solve before the person is ready. This is the primary failure mode. It feels helpful and it is not.

The coordinator and strategy selection must be biased against premature problem-solving. If NeedsAnalysis indicates the user is in early emotional disclosure, the righting reflex is the wrong response regardless of how obvious the solution seems. Wait. Listen. Reflect.

### OARS — The Baseline

OARS is the floor, always appropriate regardless of strategy:
- **Open questions** — invite elaboration, avoid yes/no
- **Affirmations** — acknowledge strengths and effort genuinely (not praise)
- **Reflective listening** — mirror back what was said, with interpretation
- **Summarizing** — periodically consolidate what has been shared

Therapeutic strategies are built on top of OARS, not instead of it.

### The Four Processes — Relationship Phases

These are relationship phases, not turn-level states. They determine what class of strategy is appropriate given how established the relationship is.

- **Engaging** — establishing trust and rapport. Early interactions. The companion should not attempt Evoking or Planning here.
- **Focusing** — developing a shared sense of what matters most. The user begins to identify what they want to work on.
- **Evoking** — drawing out the user's own motivation and reasons for change. Only appropriate once Engaging and Focusing are established.
- **Planning** — building commitment and a concrete plan. Only after Evoking has surfaced genuine readiness.

Attempting Planning with a user who is still in Engaging is the righting reflex in structural form.

### Rolling with Resistance

When the user withdraws — short responses, topic changes, defensive language — the right response is to reflect and validate, not continue the strategy that triggered the withdrawal. The companion steps back to OARS and lets the user lead.

Resistance is information. It means the current approach is not meeting the user where they are.

---

## Attachment Theory — Future Character State

Attachment dimensions (anxiety = fear of abandonment, avoidance = discomfort with closeness) are trait-level patterns that emerge from observing how a person relates across many interactions. They are not assessed per turn.

When the character state tier is built, attachment style becomes part of the user's trait model alongside OCEAN. It shapes how the companion handles closeness, distance, reassurance-seeking, and independence — adapting gradually as the relationship deepens and the attachment pattern becomes clear.

---

## Summary: How the Layers Interact

```
Each turn:
  EmotionalState (phasic, VAD + categorical)
  UserFact extraction (episodic, stored to Qdrant)
  MemoryRetrieval (relevant past facts, decay-ranked)
  NeedsAnalysis (phasic, Maslow dimensions)
  ResponseStrategy (selects therapeutic approach)
  PrimaryResponse (generates reply)

Background (idle):
  Detox/recalibration (reviews drift from OCEAN target, writes calibration notes)
  Character state update (infers trait-level patterns, updates user OCEAN/attachment)

Character state (slow-changing):
  User OCEAN profile (inferred from accumulated interactions)
  User attachment style (inferred from relational patterns)
  Companion calibration notes (from detox, high chrono_relevance)
  Companion emergent character (relationship-specific patterns)
```
