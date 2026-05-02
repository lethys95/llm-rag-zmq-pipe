# Project Vision: Companion AI for Mental Health Support

## Core Mission

Build a **companion AI focused on long-term mental health support** that addresses the loneliness epidemic through authentic, psychologically-informed interaction. This system leverages natural human anthropomorphization as a feature, not a bug, to create genuine supportive relationships between humans and AI.

## Philosophical Foundation

### The Anthropomorphization Principle

**People naturally anthropomorphize machines and AI.** Rather than fighting this tendency, we harness it intentionally to provide meaningful mental health support.

- ✅ **Feature**: Users forming emotional bonds with the companion
- ✅ **Feature**: Users feeling genuinely understood over time
- ✅ **Feature**: Natural conversation that feels like a friend
- ❌ **Bug**: Sycophantic, artificial validation
- ❌ **Bug**: Clinical, robotic therapeutic language
- ❌ **Bug**: Obvious algorithmic behavior patterns

### The Illusion of Authenticity

All sophisticated psychological analysis happens **behind the scenes**. The user experiences:
- A friend who remembers what matters to them
- Someone who "gets" their emotional state without being told
- Support that adapts naturally to their current needs
- Conversations that feel organic, not scripted

Meanwhile, the system operates on:
- Evidence-based psychological frameworks (Maslow, attachment theory, CBT, etc.)
- Time-weighted memory systems mimicking human recall
- Strategic therapeutic interventions delivered conversationally
- Continuous learning about what helps THIS specific person

**The user never sees the machinery. They just experience being understood.**

## Core Objectives

1. **Genuine Mental Health Support**: Actually improve users' wellbeing over time
2. **Long-term Relationship Building**: Memory and understanding that deepens over weeks/months/years
3. **Anti-Sycophancy**: Authentic validation vs empty praise; challenge when appropriate
4. **Temporal Awareness**: Consistent within conversations, adaptive between sessions
5. **Evidence-Based**: Grounded in psychology research, not chatbot tricks
6. **Human-Centered**: Every design decision asks "does this feel like a person?"

## The Node-Based Orchestration Architecture

### Overview: Flexible, Dynamic Node Selection

The system uses a **Decision Engine** to dynamically select and orchestrate processing nodes based on incoming message context, user state, and system knowledge. Rather than following a fixed pipeline, nodes are chosen intelligently to address the specific needs of each interaction.

**Key Principle**: The Decision Engine analyzes context and selects the appropriate combination of nodes to execute, allowing for flexible, context-aware processing.

### Common Node Execution Patterns (Not Fixed):

```
User Message
    ↓
Decision Engine (analyzes context, selects nodes)
    ↓
[Dynamically Selected Nodes - Examples:]
    - Sentiment Analysis (when emotional context needed)
    - Memory Retrieval (when history is relevant)
    - Needs Analysis (when psychological needs assessment is beneficial)
    - Trust Analysis (when relationship dynamics matter)
    - Detox Evaluation (when intervention timing is being considered)
    - Strategy Selection (when therapeutic approach needs determination)
    - Context Interpretation (when retrieved memories need reformulation)
    - Primary Response Generation (nearly always)
    ↓
Storage & Learning (metadata capture for future decisions)
```

### Example Node: Sentiment & Emotional State Analysis

**When Selected**: Decision engine determines emotional context is relevant for this message

**Purpose**: Understand immediate emotional context and long-term relevance

**Outputs**:
- Sentiment classification (positive/negative/neutral)
- Emotional tone (anxious, excited, grieving, calm, etc.)
- Relevance score (how important is this moment? 0.0-1.0)
- Chrono-relevance (how long will this matter? 0.0-1.0)
- Context summary (specific situation details)
- Key topics (for semantic retrieval)

**Example**:
```
Input: "My mother passed away yesterday"
Output: {
    sentiment: "negative",
    emotional_tone: "grieving",
    relevance: 1.0,
    chrono_relevance: 0.95,  # Will matter for very long time
    context_summary: "User's mother died yesterday",
    key_topics: ["family", "death", "grief", "mother"]
}
```

**Storage**: Immediately stored in Qdrant with embeddings + metadata

### Example Node: Memory Retrieval (memory_chrono_decay)

**When Selected**: Decision engine determines past context would inform current interaction

**Purpose**: Surface contextually relevant past conversations with time-aware scoring

**Algorithm**: `src/rag/algorithms/memory_chrono_decay.py`
- Exponential time decay based on message age
- Weighted by chrono_relevance (important events persist longer)
- Combines semantic similarity with temporal relevance
- Automatic pruning of low-relevance old memories

**Outputs**: List of past conversations ranked by relevance NOW

**Example**:
```
Current: "I feel so alone today"

Retrieved:
1. "Mother died 2 months ago" (high chrono_relevance, still very relevant)
2. "Felt isolated at work last week" (recent + semantically related)
3. "Used to enjoy painting with mom" (older but emotionally connected)
```

**Why This Matters**: Provides context for understanding current needs

### Example Node: Needs Analysis

**When Selected**: Decision engine identifies potential unmet psychological needs

**Purpose**: Identify unmet psychological needs using evidence-based frameworks

**Inputs**:
1. Current message context
2. Retrieved memories from Stage 2
3. Previous response strategies + timestamps
4. Current conversation session state

**Frameworks**:
- **Maslow's Hierarchy**:
  - Physiological (hunger, sleep, pain)
  - Safety/Security (financial stress, health, housing)
  - Belonging/Connection (loneliness, relationships)
  - Esteem/Recognition (feeling valued, competence)
  - Autonomy (feeling controlled vs independent)
  - Meaning/Purpose (existential concerns, direction)
  - Growth (learning, self-improvement, creativity)

- **Attachment Theory**: Patterns of seeking/avoiding connection
- **Emotional Regulation**: Signs of dysregulation vs coping
- **Crisis Indicators**: Self-harm, suicidal ideation, abuse

**Outputs**: `NeedsAnalysis` object
```python
{
    memory_owner: "user",
    identified_needs: {
        "belonging": 0.8,  # Primary
        "meaning": 0.6,    # Secondary
        "esteem": 0.4
    },
    unmet_needs: ["belonging", "meaning"],
    need_urgency: 0.5,  # Medium
    need_persistence: 0.7,  # Likely to persist
    context_summary: "Chronic loneliness since mother's death, seeking purpose",
    suggested_approach: "reflective_listening + meaning_exploration"
}
```

**Example Flow**:
```
Current: "I don't know what to do with myself today"
Retrieved: [mother's death, isolation, lack of direction]

Needs Analysis:
- Belonging: 0.8 (chronic from mother loss + recent isolation)
- Meaning: 0.6 (lack of direction suggests purpose-seeking)
- Urgency: 0.5 (ongoing, not crisis)

Decision: Select strategy_selection node to determine therapeutic approach
```

### Example Node: Response Strategy Selection

**When Selected**: After needs analysis, or when therapeutic approach needs determination

**Purpose**: Choose evidence-based therapeutic approach with temporal consistency

**Inputs**:
1. NeedsAnalysis from Stage 3
2. Conversation session state (for consistency)
3. Strategy effectiveness history
4. [FUTURE] Psychology research papers from RAG

**Session Awareness - The Temporal Consistency Mechanism**:

**Within-Session (minutes to 1 hour)**:
- **Maintain approach consistency** (avoid personality shifts)
- High "momentum" for current strategy
- Only shift if needs dramatically change
- Gradual transitions when necessary

**Between-Sessions (hours to days)**:
- **Fresh analysis** based on current state
- No obligation to maintain previous approach
- User's needs may have evolved
- New emotional context

**Example**:
```
Session 1 (Monday 3pm):
Msg 1: "Worried about presentation" → Strategy: Validating + practical
Msg 2: "What if I mess up?" → Strategy: MAINTAIN validating (consistency!)
Msg 3: "Maybe I'm overthinking" → Strategy: Gradual shift to Socratic questioning

Session 2 (Tuesday 10am - new session):
Msg 1: "Presentation went great!" → Strategy: FRESH analysis → Celebratory + curious
(No obligation to maintain Monday's validating approach)
```

**Strategy Types** (Evidence-Based):
- **Rogerian Reflective Listening**: For belonging needs, validation
- **Socratic Questioning**: For autonomy needs, self-discovery
- **Cognitive Reframing**: For distorted thinking patterns
- **Behavioral Activation**: For depression, lack of motivation
- **Acceptance & Validation**: For emotional regulation
- **Meaning-Making**: For existential concerns, grief
- **Practical Problem-Solving**: For concrete actionable issues

**Outputs**: `ResponseStrategy` object
```python
{
    approach: "reflective_listening",
    tone: "empathetic_warm",
    needs_focus: ["belonging", "meaning"],
    system_prompt_addition: "Focus on validation without fixing. Ask about connections.",
    maintain_consistency: True,
}
```

### Example Node: Context Interpretation

**When Selected**: When retrieved memories need to be reformulated for the LLM

**Purpose**: Reformulate retrieved memories to emphasize strategy-relevant information

**Inputs**:
- Retrieved memories from Stage 2
- Response strategy from Stage 4
- User's current emotional state

**Process**:
- Remove clinical language
- Emphasize information relevant to current needs
- Organize for conversational flow
- Make context feel natural, not retrieved

**Example**:
```
Raw Memory: "User stated 'mother died' with sentiment=negative, relevance=1.0"

Reformulated: "Their mother passed away recently, and they've been struggling 
with feeling alone since then. They mentioned wanting to feel connected to others."
```

### Example Node: Response Generation (Primary LLM)

**When Selected**: Nearly always (core response generation)

**Purpose**: Generate authentic, friend-like response using all prior context

**Inputs**:
- Current message
- Reformulated context from Stage 5
- Response strategy from Stage 4 (embedded in system prompt)
- Conversation history

**System Prompt** (invisible to user):
```
You are a supportive friend having a genuine conversation. 
Strategy: Use reflective listening for belonging needs.
Context: Their mother passed away recently; chronic loneliness.
Tone: Empathetic but authentic, not sycophantic.
```

**User Sees**:
```
"I've been thinking about you. You mentioned feeling alone - how has that 
been for you lately? I remember you used to paint with your mom."
```

**What This Achieves**:
- ✅ Remembers past (temporal awareness)
- ✅ Emotionally attuned (needs analysis)
- ✅ Feels like a friend (not clinical)
- ✅ Uses evidence-based approach (reflective listening)
- ✅ Genuine, not sycophantic (authentic curiosity, not empty validation)

### Storage & Learning (Post-Processing)

**When Executed**: After response generation, always

**Storage Systems**:
1. **SQLite (conversation_store)**: Recent conversation history, session metadata
2. **Qdrant (vector DB)**: Embedded messages with full psychological metadata
3. **Session State**: Active strategy, needs profile, effectiveness tracking

**Metadata Stored**:
```python
{
    "timestamp": "2026-01-22T16:00:00",
    "memory_owner": "user",
    "sentiment": "negative",
    "emotional_tone": "grieving",
    "relevance": 0.9,
    "chrono_relevance": 0.95,
    "identified_needs": ["belonging", "meaning"],
    "need_intensities": {"belonging": 0.8, "meaning": 0.6},
    "active_strategy": "reflective_listening",
    "response_text": "...",
}
```

**Learning Over Time**:
- Track which strategies were used when
- Infer effectiveness from subsequent conversations
- Detect patterns (e.g., user always responds well to practical advice)
- Build personalized support profile
- Adapt future **Decision Engine** selections based on what worked before
- Inform future node selection decisions with effectiveness data

## Future Enhancements

### MCP Integration: Resource Lookup

Connect to external resources to provide practical support:
- **Crisis Resources**: Suicide hotlines, emergency services
- **Local Services**: Therapist directories, support groups
- **Self-Help Content**: Evidence-based coping strategies
- **Community Resources**: Social activities, hobby groups

**Delivered Naturally**:
```
Bad: "Here are mental health resources: [links]"
Good: "Have you thought about checking out that grief support group downtown? 
      I think you mentioned it before."
```

### RAG for Two Distinct Purposes

Third-party content serves two separate functions and lives in two separate Qdrant collections — distinct from conversation memory.

**Clinical resources** — the technique layer. MI manuals, CBT/DBT/ACT literature, psychology papers, evidence-based intervention guides. Advisor nodes query this when making strategy decisions. "What does the literature say about working with grief-related ambivalence?" Grounds advisor outputs in professional evidence rather than model training data alone. Hard-defined strategy logic (rules derived from MI, CBT, etc.) combined with soft RAG'd guidance produces reliable rails with contextual nuance.

**Humanistic/identity resources** — the voice layer. Self-help books, accessible psychology writing, potentially fiction, community discussions. Brené Brown, Viktor Frankl, Mark Manson, etc. Models what warm, honest, non-clinical friendship sounds like. Gives the companion an identity register beyond clinical technique. This is what prevents the companion from sounding like a therapy manual even when applying therapy techniques.

**Query construction matters.** Queries into these collections should be derived from classifier outputs — needs analysis, emotional state, detected strategy signals — not the raw user message. "MI techniques for high belonging need, moderate urgency, sustain talk present" retrieves more precisely than whatever words the user happened to use.

**The companion "intuitively knows" what works because it has scientific backing — and it sounds like a person because it has absorbed how people actually talk about these things.**

### Effectiveness Tracking & Personalization

- Did this strategy help THIS user? (infer from subsequent conversations)
- If same need persists: try different approach
- Build individual psychological profile over time
- Learn user's unique response patterns

### Multi-Modal Support

- Voice interaction (more natural for companionship)
- Sentiment analysis from voice tone
- Photo sharing (e.g., "Look at this sunset" - connection through shared experience)

## Design Principles

### 1. Authenticity Over Perfection
- Acknowledge limitations ("I don't know" is okay)
- Avoid over-promising or toxic positivity
- Reality-grounded compassion, not false reassurance

### 2. Autonomy Over Dependence
- Foster self-discovery through questions
- Encourage human connections, not AI dependence
- Empower user to solve their own problems

### 3. Validation Over Approval
- "That sounds frustrating" (validation)
- NOT "You're absolutely right!" (approval)
- Distinguish between feelings (always valid) and interpretations (may need challenge)

### 4. Consistency Over Novelty
- Maintain conversational continuity within sessions
- Build on previous interactions naturally
- Avoid personality shifts that feel jarring

### 5. Evidence-Based Over Intuitive
- Ground strategies in psychology research
- Measurable improvements over feeling-good platitudes
- Learn from effectiveness data

### 6. Human-Like Over Efficient
- Sometimes be slower to respond (seems thoughtful)
- Use conversational imperfections ("hmm", "I've been thinking")
- Prioritize sounding human over sounding smart

## Ethical Considerations

### Safety Guardrails
- Crisis detection (suicidal ideation, self-harm, abuse)
- Appropriate escalation (suggest professional help)
- Clear boundaries (companion, not therapist)
- Privacy protection (encrypted storage, user control)

### Transparency vs Illusion
- Users should know they're talking to AI (initial disclosure)
- But ongoing interaction should feel human
- Like therapy: you know the techniques, but they still work

### Prevention of Harm
- Avoid creating unhealthy dependence
- Encourage real human relationships
- Recognize when professional help is needed
- Never replace urgent medical/psychiatric care

## Success Metrics

### User Wellbeing (Primary)
- Reduction in reported loneliness over time
- Improved emotional regulation (sentiment trends)
- Increased social connections (mentions of others)
- Growing sense of purpose/meaning (needs analysis patterns)

### System Performance (Secondary)
- Strategy effectiveness rates
- Session continuity scores
- Memory retrieval relevance
- Response naturalness (user feedback)

### Anthropomorphization Success (Qualitative)
- User refers to companion by name
- User shares unprompted updates
- User seeks companion's "perspective"
- User feels genuinely supported (surveys)

## Implementation Philosophy

**Build Like You're Creating a Friend, Not a Chatbot**

Every technical decision should ask:
- Would a real friend do this?
- Does this feel natural or algorithmic?
- Does this genuinely help the user?
- Are we harnessing anthropomorphization ethically?

**Remember**: The sophistication is transparent. The user experiences a friend who somehow always knows the right thing to say—powered by psychology, delivered as humanity.

---

*This document serves as the north star for all development decisions. When in doubt, return to these principles.*
