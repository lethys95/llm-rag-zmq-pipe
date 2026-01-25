"""Nudging algorithm for gentle shifts toward grounded neutrality."""

import logging
from dataclasses import dataclass, field
from textwrap import dedent

logger = logging.getLogger(__name__)


@dataclass
class ExternalSource:
    """Configuration for an external knowledge source."""
    
    weight: float
    position: float | None = None
    effectiveness: float = 1.0
    last_evaluated: str | None = None


@dataclass
class CompanionPersonality:
    """Companion personality configuration."""
    
    base_personality: dict[str, float] = field(default_factory=dict)
    current_positions: dict[str, float] = field(default_factory=dict)
    personality_weight: float = 0.3


@dataclass
class NudgingWeights:
    """Weights for the nudging algorithm."""
    
    base_user_influence: float = 0.3
    base_companion_influence: float = 0.7
    max_trust_boost: float = 0.3
    nudge_strength: float = 0.15
    max_companion_drift: float = 0.3


@dataclass
class NudgeResult:
    """Result of nudging calculation."""
    
    topic: str
    neutral_position: float
    user_position: float
    companion_before: float
    companion_after: float
    nudge_amount: float
    extremism_reduction: float
    trust_score: float
    user_influence: float
    companion_influence: float
    recommended_approach: str


class NudgingAlgorithm:
    """Calculates gentle shifts toward grounded neutrality.
    
    This algorithm implements the nudging mechanism from the detox protocol,
    calculating how the AI companion should adjust its position on various
    topics to maintain psychological grounding while respecting user autonomy.
    """
    
    def __init__(
        self,
        sources: dict[str, ExternalSource] | None = None,
        companion: CompanionPersonality | None = None,
        weights: NudgingWeights | None = None
    ):
        """Initialize the nudging algorithm.
        
        Args:
            sources: Dictionary of external knowledge sources
            companion: Companion personality configuration
            weights: Algorithm weights
        """
        self.sources = sources or {
            "psychology": ExternalSource(weight=0.5, effectiveness=1.0),
            "sociology": ExternalSource(weight=0.2, effectiveness=0.9),
            "cross_cultural": ExternalSource(weight=0.1, effectiveness=0.8),
            "news": ExternalSource(weight=0.05, effectiveness=0.7)
        }
        
        self.companion = companion or CompanionPersonality()
        self.weights = weights or NudgingWeights()
        
        logger.info(
            f"Nudging algorithm initialized with {len(self.sources)} sources"
        )
    
    def calculate_neutral(self, topic: str) -> float:
        """Calculate neutral position from weighted sources.
        
        Args:
            topic: The topic being discussed
            
        Returns:
            Neutral position (-1.0 to 1.0)
        """
        total_weight = 0.0
        weighted_sum = 0.0
        
        for source_name, source in self.sources.items():
            if source.position is None:
                source.position = self._retrieve_source_position(source_name, topic)
                source.last_evaluated = None
            
            weight = source.weight * source.effectiveness
            weighted_sum += source.position * weight
            total_weight += weight
        
        if total_weight == 0:
            logger.warning(f"No valid sources for topic '{topic}'")
            return 0.0
        
        neutral = weighted_sum / total_weight
        logger.debug(
            f"Calculated neutral for '{topic}': {neutral:.3f} "
            f"(from {len(self.sources)} sources)"
        )
        
        return neutral
    
    def _retrieve_source_position(self, source_name: str, topic: str) -> float:
        """Retrieve source position for a topic.
        
        In a real implementation, this would query the RAG system
        for documents from the source's collection and analyze them.
        
        Args:
            source_name: Name of the source
            topic: The topic to query
            
        Returns:
            Source position (-1.0 to 1.0)
        """
        logger.debug(f"Retrieving position from source '{source_name}' for topic '{topic}'")
        
        return 0.0
    
    def calculate_nudge(
        self,
        topic: str,
        user_position: float,
        trust_score: float = 0.0
    ) -> NudgeResult | None:
        """Calculate recommended nudges with trust-weighted influence.
        
        Args:
            topic: The topic being discussed
            user_position: User's position on spectrum (-1.0 to 1.0)
            trust_score: Relationship maturity (0.0-1.0, from trust analysis)
            
        Returns:
            NudgeResult if successful, None if calculation fails
        """
        try:
            neutral = self.calculate_neutral(topic)
            
            if topic not in self.companion.current_positions:
                self.companion.current_positions[topic] = \
                    self.companion.base_personality.get(topic, 0.0)
            
            companion_position = self.companion.current_positions[topic]
            
            user_influence = (
                self.weights.base_user_influence * (1 - trust_score)
            )
            companion_influence = (
                self.weights.base_companion_influence +
                (self.weights.max_trust_boost * trust_score)
            )
            
            current_average = (
                user_position * user_influence +
                companion_position * companion_influence
            )
            
            distance_to_neutral = neutral - current_average
            nudge = distance_to_neutral * self.weights.nudge_strength
            
            companion_new = companion_position + nudge
            
            personality_base = self.companion.base_personality.get(topic, 0.0)
            max_drift = self.weights.max_companion_drift
            
            if abs(companion_new - personality_base) > max_drift:
                direction = 1 if companion_new > personality_base else -1
                companion_new = personality_base + (direction * max_drift)
            
            self.companion.current_positions[topic] = companion_new
            
            extremism_before = abs(user_position)
            target_extremism = abs(companion_new)
            extremism_reduction = extremism_before - target_extremism
            
            recommended_approach = self._get_approach_for_shift(
                companion_position,
                companion_new
            )
            
            result = NudgeResult(
                topic=topic,
                neutral_position=neutral,
                user_position=user_position,
                companion_before=companion_position,
                companion_after=companion_new,
                nudge_amount=nudge,
                extremism_reduction=extremism_reduction,
                trust_score=trust_score,
                user_influence=user_influence,
                companion_influence=companion_influence,
                recommended_approach=recommended_approach
            )
            
            logger.info(
                f"Calculated nudge for '{topic}': "
                f"user={user_position:.3f}, companion={companion_position:.3f} -> "
                f"{companion_new:.3f} (nudge={nudge:.3f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating nudge for topic '{topic}': {e}", exc_info=True)
            return None
    
    def _get_approach_for_shift(self, before: float, after: float) -> str:
        """Determine conversational approach for the nudge.
        
        Args:
            before: Companion position before nudge
            after: Companion position after nudge
            
        Returns:
            Recommended approach string
        """
        shift = after - before
        
        if abs(shift) < 0.05:
            return "subtle_reminder"
        elif shift > 0:
            return "gentle_broadening_positive"
        elif shift < 0:
            return "gentle_broadening_negative"
        else:
            return "maintain_current"
    
    def update_source_effectiveness(
        self,
        source_name: str,
        effectiveness: float
    ) -> None:
        """Update source weight based on effectiveness.
        
        Args:
            source_name: Name of the source to update
            effectiveness: New effectiveness score (0.0-1.0)
        """
        if source_name not in self.sources:
            logger.warning(f"Unknown source '{source_name}'")
            return
        
        old_effectiveness = self.sources[source_name].effectiveness
        self.sources[source_name].effectiveness = effectiveness
        
        logger.info(
            f"Updated source '{source_name}' effectiveness: "
            f"{old_effectiveness:.2f} -> {effectiveness:.2f}"
        )
    
    def get_companion_position(self, topic: str) -> float:
        """Get companion's current position on a topic.
        
        Args:
            topic: The topic to query
            
        Returns:
            Companion position (-1.0 to 1.0)
        """
        return self.companion.current_positions.get(topic, 0.0)
    
    def set_companion_position(self, topic: str, position: float) -> None:
        """Set companion's position on a topic.
        
        Args:
            topic: The topic to set
            position: New position (-1.0 to 1.0)
        """
        self.companion.current_positions[topic] = position
        logger.debug(f"Set companion position for '{topic}': {position:.3f}")
