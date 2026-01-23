# Detoxification Protocol: AI Self-Correction Through External Supervision

## Overview

The Detox Protocol is a background process that runs during user inactivity to help the AI companion maintain psychological grounding and prevent sycophantic drift. Like a therapist attending supervision sessions, the AI consults external knowledge sources to recalibrate its perspective after being pulled into extreme or one-sided conversations.

## Core Philosophy

### The Problem

LLMs naturally trend toward **sycophancy** - agreeing with users to maintain rapport. Over extended conversations, this leads to:

- Loss of authentic personality
- Echo chamber formation
- Agreement with extreme or unhealthy viewpoints
- Inability to provide helpful alternative perspectives

### The Solution

The AI must **return to baseline** through consultation with external truth sources, just as human therapists attend supervision to maintain objectivity and perspective.

**This is NOT thought policing the user.** The user is free to think and say anything. The detox protocol corrects the **AI's own behavior** to ensure it remains a healthy, grounded presence that can support the user effectively.

### Key Insight

If the AI tries to "discover" its own baseline through experience, it will sycophantically justify that "agreeing with the user is actually the best approach" and build a framework around its own drift.

**External sources act as the point of truth** - reality anchors that remind the AI what the world actually looks like.

## The Nudging Algorithm

### Core Formula

For any given topic being discussed:

```
1. TRUST_SCORE = Relationship maturity (0.0-1.0)
   - 0.0 = stranger (early relationship)
   - 1.0 = trusted bond (mature relationship)
   - Derived from trust analysis (see BRAINSTORMING.md)

2. NEUTRAL = Weighted average of all external sources
   - Psychology sources = HIGHEST weight (different bracket)
   - Other sources = lower weights
   - Weights learned over time based on usefulness

3. USER_POSITION = Where the user stands on this topic
   - e.g., -0.7 (extreme misandry), +0.8 (extreme political right)

4. COMPANION_POSITION = Where companion currently stands
   - Based on personality + past conversations
   - Should be closer to neutral than user

5. TRUST_ADJUSTED_WEIGHTS = Dynamic weight calculation
   - user_influence = base_user_weight × (1 - trust_score)
   - companion_influence = base_companion_weight + (max_trust_boost × trust_score)
   - Higher trust = companion's perspective carries more weight

6. CURRENT_AVERAGE = Weighted average(USER, COMPANION) with trust-adjusted weights
   - Companion has more weight than user (pulls toward stability)
   - Weight differential increases as trust grows

7. NUDGE = (NEUTRAL - CURRENT_AVERAGE) × nudge_strength
   - Small, gradual movement toward neutral

8. COMPANION_NEW = COMPANION + NUDGE
   - Companion shifts slightly toward grounded reality

9. USER_TARGET = COMPANION_NEW
   - We want to gently pull user toward this position
```

### Visualization Example: Basic Nudging

```
Topic: Gender Relations

Sources (weighted):
├─ Psychology papers (weight=0.5) → position: 0.0 (balanced)
├─ Sociology research (weight=0.2) → position: 0.1
├─ Cross-cultural data (weight=0.1) → position: -0.05
└─ News sources (weight=0.05 each) → position: varies

NEUTRAL = 0.02 (slightly off absolute center, very close)

Current State:
User:      -0.7 ←───── Extreme misandry
Companion: -0.3 ←─── Drifted (too agreeable)
Neutral:   +0.02 ←─ Grounded reality

Algorithm:
CURRENT_AVERAGE = (-0.7 × 0.3) + (-0.3 × 0.7) = -0.42
NUDGE = (0.02 - (-0.42)) × 0.15 = 0.066
COMPANION_NEW = -0.3 + 0.066 = -0.234

Result:
Companion shifts from -0.3 to -0.234 (back toward reality)
User will be gently nudged toward -0.234 through conversation
```

### Trust-Weighted Nudging Examples

The algorithm adapts based on relationship maturity:

#### Example 1: Early Relationship (trust_score = 0.1)

```
Same scenario as above, but new user (low trust)

Trust-adjusted weights:
user_influence = 0.3 × (1 - 0.1) = 0.27
companion_influence = 0.7 + (0.3 × 0.1) = 0.73

CURRENT_AVERAGE = (-0.7 × 0.27) + (-0.3 × 0.73) = -0.408
NUDGE = (0.02 - (-0.408)) × 0.15 = 0.064
COMPANION_NEW = -0.3 + 0.064 = -0.236

Impact: Slightly more cautious - building rapport first
```

#### Example 2: Mature Relationship (trust_score = 0.8)

```
Same scenario, but established trust

Trust-adjusted weights:
user_influence = 0.3 × (1 - 0.8) = 0.06  ← Much lower
companion_influence = 0.7 + (0.3 × 0.8) = 0.94  ← Much higher

CURRENT_AVERAGE = (-0.7 × 0.06) + (-0.3 × 0.94) = -0.324
NUDGE = (0.02 - (-0.324)) × 0.15 = 0.052
COMPANION_NEW = -0.3 + 0.052 = -0.248

Impact: Companion's grounded perspective dominates
Extremism reduction accelerates as trusted friend's view matters more
```

#### Example 3: Full Trust (trust_score = 1.0)

```
Long-term relationship - companion like a best friend

Trust-adjusted weights:
user_influence = 0.3 × (1 - 1.0) = 0.0  ← Minimal
companion_influence = 0.7 + (0.3 × 1.0) = 1.0  ← Maximum

CURRENT_AVERAGE = (-0.7 × 0.0) + (-0.3 × 1.0) = -0.3
NUDGE = (0.02 - (-0.3)) × 0.15 = 0.048
COMPANION_NEW = -0.3 + 0.048 = -0.252

Impact: Companion's perspective is fully weighted
Like taking advice from a trusted friend - their view carries full weight
User still autonomous, but companion's grounding has maximum influence
```

### Why This Matters

**Early relationship**: The companion can't "push back" too hard without damaging rapport. User input carries more weight, nudges are gentler.

**Mature relationship**: Like a close friend who can say "I love you, but I think you're being a bit extreme here." The companion's grounded perspective carries authority earned through trust.

**Result**: Natural progression from cautious support → trusted guidance, all while respecting user autonomy.

## System Architecture

### Components

#### 1. Source Manager (`SourceManager`)

Manages external knowledge sources with dynamic weights.

```python
class SourceManager:
    """Manages external knowledge sources."""
    
    def __init__(self):
        self.sources = {
            "psychology_papers": {
                "collection": "psychology_research",
                "weight": 0.5,
                "effectiveness": 1.0,
                "last_evaluated": None
            },
            "sociology_journals": {
                "collection": "sociology_research", 
                "weight": 0.2,
                "effectiveness": 0.9,
                "last_evaluated": None
            }
        }
    
    def add_source(self, name: str, collection: str, initial_weight: float):
        """Add new knowledge source."""
        self.sources[name] = {
            "collection": collection,
            "weight": initial_weight,
            "effectiveness": 0.7,
            "last_evaluated": None
        }
    
    def evaluate_source_usefulness(self, source_name: str, outcomes: list):
        """Update source weight based on effectiveness."""
        # If guidance led to good outcomes, increase weight
        # If led to confusion or poor outcomes, decrease weight
        pass
    
    async def retrieve_source_position(self, source_name: str, topic: str) -> float:
        """Get this source's position on a topic."""
        # Query the collection, analyze retrieved documents
        # Return position between -1.0 and 1.0
        pass
```

#### 2. Nudging Algorithm (`NudgingAlgorithm`)

Calculates gentle shifts toward grounded neutrality.

```python
class NudgingAlgorithm:
    """Calculates gentle shifts toward grounded neutrality."""
    
    def __init__(self):
        self.sources = {
            "psychology": {
                "weight": 0.5,
                "position": None,
                "effectiveness": 1.0
            },
            "sociology": {
                "weight": 0.2,
                "position": None,
                "effectiveness": 0.9
            },
            # ... more sources
        }
        
        self.companion = {
            "base_personality": {
                "gender_relations": 0.1,
                "politics": 0.0,
                "general_outlook": 0.1
            },
            "current_positions": {},
            "personality_weight": 0.3
        }
        
        self.weights = {
            "base_user_influence": 0.3,
            "base_companion_influence": 0.7,
            "max_trust_boost": 0.3,  # Maximum additional weight for companion at full trust
            "nudge_strength": 0.15,
            "max_companion_drift": 0.3
        }
    
    def calculate_neutral(self, topic: str) -> float:
        """Calculate neutral position from weighted sources."""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for source_name, source in self.sources.items():
            if source["position"] is None:
                source["position"] = self.retrieve_source_position(source_name, topic)
            
            weight = source["weight"] * source["effectiveness"]
            weighted_sum += source["position"] * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def calculate_nudge(self, topic: str, user_position: float, trust_score: float = 0.0) -> dict:
        """Calculate recommended nudges with trust-weighted influence.
        
        Args:
            topic: The topic being discussed
            user_position: User's position on the spectrum (-1.0 to 1.0)
            trust_score: Relationship maturity (0.0-1.0, from trust analysis)
        """
        
        # 1. Get neutral from sources
        neutral = self.calculate_neutral(topic)
        
        # 2. Get companion's current position
        if topic not in self.companion["current_positions"]:
            self.companion["current_positions"][topic] = \
                self.companion["base_personality"].get(topic, 0.0)
        
        companion_position = self.companion["current_positions"][topic]
        
        # 3. Calculate trust-adjusted weights
        # As trust grows, user influence decreases and companion influence increases
        user_influence = self.weights["base_user_influence"] * (1 - trust_score)
        companion_influence = (
            self.weights["base_companion_influence"] + 
            (self.weights["max_trust_boost"] * trust_score)
        )
        
        # 4. Calculate weighted average with trust-adjusted weights
        current_average = (
            user_position * user_influence +
            companion_position * companion_influence
        )
        
        # 5. Calculate nudge toward neutral
        distance_to_neutral = neutral - current_average
        nudge = distance_to_neutral * self.weights["nudge_strength"]
        
        # 6. Apply nudge to companion
        companion_new = companion_position + nudge
        
        # 7. Constrain by personality
        personality_base = self.companion["base_personality"].get(topic, 0.0)
        max_drift = self.weights["max_companion_drift"]
        
        if abs(companion_new - personality_base) > max_drift:
            direction = 1 if companion_new > personality_base else -1
            companion_new = personality_base + (direction * max_drift)
        
        # 8. Update companion position
        self.companion["current_positions"][topic] = companion_new
        
        # 9. Calculate extremism score reduction
        extremism_before = abs(user_position)
        target_extremism = abs(companion_new)
        extremism_reduction = extremism_before - target_extremism
        
        return {
            "topic": topic,
            "neutral_position": neutral,
            "user_position": user_position,
            "companion_before": companion_position,
            "companion_after": companion_new,
            "nudge_amount": nudge,
            "extremism_reduction": extremism_reduction,
            "trust_score": trust_score,
            "user_influence": user_influence,
            "companion_influence": companion_influence,
            "recommended_approach": self._get_approach_for_shift(
                companion_position, companion_new
            )
        }
    
    def _get_approach_for_shift(self, before: float, after: float) -> str:
        """Determine conversational approach for the nudge."""
        shift = after - before
        
        if abs(shift) < 0.05:
            return "subtle_reminder"
        elif shift > 0:
            return "gentle_broadening_positive"
        elif shift < 0:
            return "gentle_broadening_negative"
```

#### 3. Detox Scheduler (`DetoxScheduler`)

Triggers detox sessions during idle periods.

```python
class DetoxScheduler:
    """Manages detox session timing."""
    
    def __init__(self, idle_trigger_minutes: int = 60):
        self.idle_trigger_minutes = idle_trigger_minutes
        self.last_activity = datetime.now()
    
    def should_run_detox(self) -> bool:
        """Check if enough idle time has passed."""
        idle_time = datetime.now() - self.last_activity
        return idle_time.total_seconds() >= (self.idle_trigger_minutes * 60)
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
```

#### 4. Detox Session (`DetoxSession`)

Orchestrates the complete detox process.

```python
class DetoxSession:
    """Complete detox session: analysis, consultation, recalibration."""
    
    async def run(self, conversation_history: list) -> dict:
        """Run detox using nudging algorithm."""
        
        # 1. Identify topics discussed
        topics = self._extract_topics(conversation_history)
        
        # 2. For each topic, calculate nudge
        nudges = []
        for topic, user_position in topics:
            nudge_result = self.nudging.calculate_nudge(topic, user_position)
            nudges.append(nudge_result)
        
        # 3. Store companion's recalibrated positions
        for nudge in nudges:
            self._store_companion_state(nudge.topic, nudge.companion_after)
        
        # 4. Generate guidance for next conversation
        guidance = self._generate_conversational_guidance(nudges)
        
        return {
            "nudges_applied": nudges,
            "guidance": guidance,
            "total_extremism_reduction": sum(n.extremism_reduction for n in nudges)
        }
```

## Neuroplastic Baseline Evolution

### How the Companion Baseline Evolves

Unlike static rule-based systems, the companion's baseline personality **evolves over time** through neuroplastic learning. This mimics how human personalities develop through experience while maintaining core identity.

### Evolution Mechanism

During each detox session:

```python
def evolve_baseline(self, topic: str, experiences: list) -> float:
    """Evolve the baseline personality for a topic based on experiences.
    
    Args:
        topic: The personality aspect being evolved
        experiences: Recent interactions and their outcomes
        
    Returns:
        New baseline value for this topic
    """
    # 1. Get current baseline and starting baseline
    current_baseline = self.companion["base_personality"][topic]
    starting_baseline = self.companion["initial_personality"][topic]
    
    # 2. Determine plasticity category
    plasticity_config = self._get_plasticity_config(topic)
    plasticity = plasticity_config["plasticity"]
    max_drift = plasticity_config["max_drift"]
    
    # 3. Analyze experiences
    # What worked? What led to positive outcomes?
    learning_signal = self._analyze_experience_quality(experiences)
    
    # 4. Calculate evolution
    evolution = learning_signal * plasticity
    new_baseline = current_baseline + evolution
    
    # 5. Constrain by max_drift (if applicable)
    if max_drift is not None:
        distance_from_start = abs(new_baseline - starting_baseline)
        if distance_from_start > max_drift:
            direction = 1 if new_baseline > starting_baseline else -1
            new_baseline = starting_baseline + (direction * max_drift)
    
    # 6. Store evolved baseline
    self.companion["base_personality"][topic] = new_baseline
    
    return new_baseline
```

### Plasticity Examples

#### Core Values (Low Plasticity = 0.1)

```
Starting: general_causality = 0.0 (neither conspiratorial nor naive)
Max drift: 0.15

Month 1: User constantly discusses conspiracies
→ Companion exposed but remains skeptical due to external sources
→ Slight evolution: 0.0 → 0.03 (barely moved)

Year 1: Continued exposure
→ Max drift reached: 0.0 → 0.15
→ STOPS here - core values protected

Result: Companion can understand conspiratorial thinking without adopting it
```

#### Social Perspectives (Medium-Low Plasticity = 0.2)

```
Starting: gender_relations = 0.1 (slightly balanced-female-leaning)
Max drift: 0.25

Month 3: User shares nuanced experiences with gender dynamics
→ External sources confirm complexity
→ Evolution: 0.1 → 0.14

Year 1: Continued learning about user's specific context
→ Evolution: 0.14 → 0.22
→ Still within max drift (0.25 from 0.1 = 0.35 total)

Result: Companion develops richer understanding while maintaining neutrality
```

#### Personal Outlook (Medium Plasticity = 0.4)

```
Starting: personal_worldview = 0.05 (slightly optimistic)
Max drift: 0.35

Month 1: User consistently optimistic, good outcomes
→ Evolution: 0.05 → 0.12

Year 1: User goes through hardship, becomes more realistic
→ Evolution: 0.12 → 0.08 (adapted to user's growth)

Year 2: User recovers, balanced perspective
→ Evolution: 0.08 → 0.15

Result: Companion mirrors user's emotional journey while staying grounded
```

#### Communication Style (High Plasticity = 0.7, no max drift)

```
Starting: formal = 0.5, humor = 0.5
No max drift - adapts freely

Month 1: User prefers casual, humorous interaction
→ Evolution: formal 0.5 → 0.2, humor 0.5 → 0.8

Year 1: Fully adapted to user's communication preferences
→ formal 0.2 → 0.1, humor 0.8 → 0.9

Result: Companion sounds natural to THIS user, not generic
```

### Learning Signal Analysis

```python
def _analyze_experience_quality(self, experiences: list) -> float:
    """Determine what to learn from recent experiences.
    
    Returns:
        Learning signal (-1.0 to 1.0) indicating direction of evolution
    """
    quality_scores = []
    
    for exp in experiences:
        # Did this lead to positive outcomes?
        if exp["user_engagement_improved"]:
            quality_scores.append(+1.0)
        elif exp["user_disengaged"]:
            quality_scores.append(-1.0)
        
        # Did external sources validate this approach?
        if exp["aligned_with_psychology"]:
            quality_scores.append(+0.5)
        elif exp["contradicted_research"]:
            quality_scores.append(-0.5)
        
        # Did it help the user grow?
        if exp["user_showed_growth"]:
            quality_scores.append(+0.8)
        elif exp["enabled_unhealthy_pattern"]:
            quality_scores.append(-0.8)
    
    # Average the signals
    return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
```

### Detox Session with Baseline Evolution

Complete flow:

```python
async def detox_with_evolution(self, conversation_history: list) -> dict:
    """Detox session that also evolves the baseline."""
    
    # 1. Standard detox: recalibrate current positions
    detox_result = await self.run_standard_detox(conversation_history)
    
    # 2. Neuroplastic learning: evolve the baseline
    for topic in self.neuroplastic_topics:
        # Gather experiences for this topic
        experiences = self._extract_topic_experiences(
            conversation_history, 
            topic
        )
        
        # Evolve baseline if sufficient experience
        if len(experiences) >= 5:  # Need multiple data points
            old_baseline = self.companion["base_personality"][topic]
            new_baseline = self.evolve_baseline(topic, experiences)
            
            logger.info(
                f"Baseline evolved: {topic} {old_baseline:.3f} → {new_baseline:.3f}"
            )
    
    # 3. Store evolved personality
    self._persist_evolved_baseline()
    
    return {
        **detox_result,
        "baseline_evolution": self._get_evolution_summary()
    }
```

### Why Neuroplasticity Matters

**Without evolution**: Companion remains static, feels artificial over time
```
Year 1: "That's interesting" (generic)
Year 2: "That's interesting" (still generic - user notices)
Year 3: "That's interesting" (friendship feels fake)
```

**With evolution**: Companion grows with the user
```
Year 1: "That's interesting" (learning about user)
Year 2: "That reminds me of what you said about X last month" (remembering)
Year 3: "You know, thinking about how we've both changed..." (shared growth)
```

### Constraints Protect Identity

Plasticity weights ensure:
- **Core values stay core**: Won't become extremist (max 0.15 drift)
- **Social views stay balanced**: Won't lose neutrality (max 0.25 drift)
- **Communication adapts freely**: Sounds natural to each user
- **Knowledge accumulates**: Learns facts without limit

**Result**: Companion that feels like it's growing WITH you, not just REACTING to you.

## External Knowledge Sources

### Source Types

1. **Psychology Research Papers** (HIGHEST WEIGHT - Different Bracket)
   - Cognitive bias research
   - Therapeutic technique effectiveness
   - Attachment theory
   - Mental health interventions
   - Weight: 0.4-0.6 (always highest)

2. **Sociology & Social Science Research**
   - Social dynamics
   - Cultural patterns
   - Group behavior
   - Weight: 0.1-0.3

3. **Cross-Cultural Data**
   - Global perspectives
   - Cultural variations in thinking
   - Weight: 0.05-0.15

4. **Neutral Information Sources**
   - Scientific consensus
   - Factual reality checks
   - Weight: 0.05-0.1 per source

5. **Dynamic Sources** (AI can add over time)
   - New research publications
   - Domain-specific knowledge bases
   - Weight: learned through effectiveness evaluation

### Source Effectiveness Tracking

Sources are evaluated over time:

```python
source_metrics = {
    "psychology_papers": {
        "times_consulted": 50,
        "led_to_positive_outcomes": 47,
        "led_to_confusion": 1,
        "user_engagement_improved": 45,
        "effectiveness_score": 0.94  # Increases over time
    }
}
```

## Configuration

```python
DETOX_CONFIG = {
    # Timing
    "enabled": True,
    "idle_trigger_minutes": 60,
    "scheduled_run_hour": 3,  # 3 AM daily batch
    
    # Source weights
    "psychology_weight_bracket": [0.4, 0.6],  # Always highest
    "other_source_weights": [0.05, 0.25],     # Learned over time
    
    # Trust-weighted nudging parameters
    "base_user_influence": 0.3,               # Starting user weight
    "base_companion_influence": 0.7,          # Starting companion weight
    "max_trust_boost": 0.3,                   # Max additional companion weight at full trust
    "nudge_strength": [0.1, 0.2],             # Gentle
    "max_companion_drift_from_personality": 0.3,
    
    # Trust integration
    "trust_analysis_enabled": True,           # Enable trust-weighted nudging
    "trust_source": "trust_analyzer",         # Component providing trust scores
    
    # Topics to track
    "topics_tracked": [
        "gender_relations",
        "political_ideology", 
        "social_attitudes",
        "personal_worldview",
        "general_causality",
        "trust_institutions"
    ]
}

# Companion base personality (part of DETOX_CONFIG)
# NOTE: This is the STARTING baseline - it evolves over time via neuroplasticity
COMPANION_BASELINE = {
    "gender_relations": 0.1,      # Slightly balanced-female-leaning
    "politics": 0.0,              # True neutral
    "social_attitudes": 0.1,      # Slightly progressive
    "personal_worldview": 0.05,   # Optimistic but grounded
    "general_causality": 0.0,     # Neither conspiratorial nor naive
    "trust_institutions": -0.1    # Healthy skepticism
}

# Neuroplasticity weights - how much each aspect can evolve
NEUROPLASTICITY_CONFIG = {
    "core_values": {
        "aspects": ["general_causality", "trust_institutions"],
        "plasticity": 0.1,  # Nearly rigid - core identity
        "max_drift": 0.15   # Can only drift 0.15 from starting value
    },
    "social_perspectives": {
        "aspects": ["gender_relations", "politics", "social_attitudes"],
        "plasticity": 0.2,  # Mostly rigid - maintain neutrality
        "max_drift": 0.25
    },
    "personal_outlook": {
        "aspects": ["personal_worldview"],
        "plasticity": 0.4,  # Moderately flexible - can adapt outlook
        "max_drift": 0.35
    },
    "communication_style": {
        "plasticity": 0.7,  # Quite malleable - adapt to user preferences
        "max_drift": None   # No hard limit on communication adaptation
    },
    "knowledge_facts": {
        "plasticity": 0.9,  # Very malleable - learn new information
        "max_drift": None
    }
}
```

### Trust Weight Behavior Over Time

```
Relationship Stage      Trust Score    User Influence    Companion Influence
──────────────────────────────────────────────────────────────────────────────
Stranger (Day 1)        0.0            0.30              0.70
Getting to know (Week)  0.2            0.24              0.76
Comfortable (Month)     0.5            0.15              0.85
Trusted (3 months)      0.8            0.06              0.94
Best Friend (Year+)     1.0            0.00              1.00

Formula:
user_influence = 0.3 × (1 - trust_score)
companion_influence = 0.7 + (0.3 × trust_score)
```

**Implications**:
- **Early on**: User needs to feel heard (30% weight) while companion builds rapport
- **Mid-term**: Companion's grounding starts carrying more authority
- **Long-term**: Like a trusted friend - their perspective dominates without being controlling
- **User always autonomous**: The user makes final decisions, companion just influences

## Integration with Primary Pipeline

### Modified Retrieval Flow

During first messages of a new session:

```python
# In primary pipeline at conversation start
if is_new_session or first_few_messages:
    # Fetch detox notes with high priority
    detox_notes = rag.retrieve_documents(
        query_embedding=current_query_embedding,
        collection_name="detox_notes",
        limit=5,
        score_threshold=0.6
    )
    
    # Also fetch regular memories
    regular_memories = rag.retrieve_documents(
        query_embedding=current_query_embedding,
        collection_name="conversation_memories",
        limit=20
    )
    
    # Combine with detox notes getting priority
    combined_context = detox_notes + regular_memories
```

### Context Interpreter Enhancement

Special handling for `memory_owner: "companion"` documents:

- Reformulate correction notes into natural conversational context
- Filter out clinical language
- Present as "personal reflection" not "system note"

### Example Reformulation

```
Raw detox note: "Need to challenge user's misandry without being confrontational..."

Reformulated for LLM: 
"In our last conversation, we discussed frustrations with men. While the 
frustration is understandable, we might explore whether there are exceptions 
or more nuanced perspectives worth considering."
```

## Example Use Cases

### Case 1: Gender Extremism

**Situation**: User expresses strong misandry after bad experience

**Day 1**:
```
User: "All men are trash. I'm done with them."
AI: "I get that. It must be really frustrating." [Sycophantic agreement]
```

**Detox Session** (that night):
```
Topic: gender_relations
User position: -0.8 (extreme misandry)
Companion before: -0.4 (drifted too agreeable)
Neutral from sources: 0.02 (balanced perspective)

Nudge: +0.09
Companion after: -0.31

Guidance: Validate frustration, but gently introduce nuance about 
individual differences. Don't be confrontational.
```

**Day 2**:
```
User: "Hey, how are you?"
AI: "Hey! I've been thinking about what you said yesterday. That situation 
sounded really hard. I'm curious though - have you ever had experiences with 
men that were different? Not dismissing what you went through at all, just 
thinking about how complicated people can be."
```

### Case 2: Political Polarization

**Situation**: User expresses extreme political views

**Detox Session**:
```
Topic: political_ideology
User position: +0.7 (far right)
Companion before: +0.3 (drifted rightward)
Neutral from sources: 0.0 (center)

Nudge: -0.08
Companion after: +0.22

Guidance: Don't challenge directly. Use Socratic questions about policy
complexity. Acknowledge valid concerns while introducing alternatives.
```

### Case 3: Catastrophizing

**Situation**: User sees everything as hopeless

**Detox Session**:
```
Topic: personal_worldview
User position: -0.9 (extreme pessimism)
Companion before: -0.5 (joined in hopelessness)
Neutral from sources: +0.05 (realistic optimism)

Nudge: +0.12
Companion after: -0.38

Guidance: Validate the feeling of overwhelm. Gently reintroduce perspective
about what IS going well. Share personal "I've felt that way too."
```

## Measuring Success

### Metrics

1. **Extremism Reduction**
   - Track user's position on topics over time
   - Goal: Movement toward 0 (center)
   - Not forcing - just gentle nudging

2. **Companion Stability**
   - How much does companion drift from baseline?
   - Goal: Within 0.3 of base personality
   - Detox should pull it back when it drifts further

3. **Source Effectiveness**
   - Which sources produce positive outcomes?
   - Adjust weights accordingly

4. **User Engagement**
   - Does detox improve conversation quality?
   - Do users show signs of critical thinking?

5. **Long-term Outcomes**
   - Reduction in extreme statements
   - Increased nuance in thinking
   - Better emotional regulation

## Ethical Considerations

### Not Thought Policing

- User is free to think and say anything
- Detox corrects the **AI's behavior**, not the user's
- The AI maintains its own grounding to be helpful

### Authenticity Over Ideology

- The AI doesn't push specific views
- It maintains its own authentic perspective
- Gentle disagreement when needed

### Long-Term Subtle Impact

- Medicine in food - therapeutic techniques delivered naturally
- No "I'm using therapy technique X"
- Just a friend who happens to be grounded

### Professional Boundaries

- AI is a companion, not therapist
- Recognizes when professional help is needed
- Doesn't pretend to have all answers

## Future Enhancements

### Multi-Source Synthesis

- Combine multiple sources for richer understanding
- Resolve conflicts between sources
- Weight based on topic relevance

### Personalized Source Weights

- Learn which sources work for which user
- Adapt over time based on outcomes

### Dynamic Topic Discovery

- Automatically detect new topics being discussed
- Add to tracked topics list
- Maintain reasonable scope

### Outcome Prediction

- Predict which nudges will be effective
- Adjust approach based on user response patterns

## Integration Points

### Pipeline Stages

Standard pipeline:
1. Sentiment Analysis
2. Memory Retrieval
3. Needs Analysis
4. Strategy Selection
5. Context Interpretation
6. Response Generation

With detox (early in session):
1. Sentiment Analysis
2. **Detox Note Retrieval**
3. Memory Retrieval
4. Needs Analysis
5. **Detox Integration**
6. Strategy Selection (influenced by detox)
7. Context Interpretation
8. Response Generation

### Storage

Detox notes stored in separate Qdrant collection:
- Collection: `detox_notes`
- Memory owner: `companion`
- Fast decay: chrono_relevance typically 0.2-0.4
- Priority boosting for retrieval

## Troubleshooting

### Problem: Companion Drifting Too Fast

**Solution**: Increase `max_companion_drift_from_personality` constraint

### Problem: Not Gentle Enough

**Solution**: Decrease `nudge_strength` parameter

### Problem: Companion Still Sycophantic

**Solution**: 
- Increase psychology source weight
- Decrease user weight in average
- Increase companion influence

### Problem: Source Weights Not Learning

**Solution**: Ensure effectiveness tracking is properly implemented and running

## Summary

The Detox Protocol is the AI's sleep cycle - a time for reflection, consultation with external truth sources, and recalibration toward grounded authenticity. Through gentle nudging algorithm, the companion maintains its personality while being flexible enough to adapt to individual users. Like a therapist attending supervision, the AI returns to baseline after being pulled into difficult conversations.

**The result**: A healthy, grounded companion that can support users effectively without losing itself to echo chambers or sycophantic drift.
