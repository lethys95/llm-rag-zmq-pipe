"""Trust analysis node for relationship maturity tracking."""

import logging
from datetime import datetime

from src.nodes.core.base import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.models.memory import TrustAnalysis, TrustRecord

logger = logging.getLogger(__name__)


class TrustAnalysisNode(BaseNode):
    """Analyzes and tracks relationship trust over time.
    
    Trust score is calculated based on:
    - Relationship age (time since first interaction)
    - Interaction frequency and count
    - Positive vs negative interaction ratio
    - Consistency of interactions
    - Depth of shared information
    """
    
    def __init__(
        self,
        trust_store: "TrustStore" | None = None,
        **kwargs
    ):
        """Initialize trust analysis node.
        
        Args:
            trust_store: Trust store for persisting records
            **kwargs: Additional arguments passed to BaseNode
        """
        super().__init__(
            name="trust_analysis",
            priority=1,
            queue_type="immediate",
            **kwargs
        )
        self.trust_store = trust_store or TrustStore()
        
        logger.info("Trust analysis node initialized")
    
    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        """Calculate trust score for current user.
        
        Args:
            broker: Knowledge broker containing dialogue input
            
        Returns:
            NodeResult with trust analysis
        """
        # Get user ID
        user_id = None
        if hasattr(broker, "dialogue_input") and broker.dialogue_input:
            user_id = broker.dialogue_input.speaker
        
        if not user_id:
            logger.debug("No user ID available for trust analysis")
            return NodeResult(
                status=NodeStatus.SKIPPED,
                metadata={"reason": "no_user_id"}
            )
        
        # Get conversation history if available
        conversation_history = getattr(broker, "conversation_history", [])
        
        # Analyze current interaction
        current_record = self._analyze_current_interaction(
            broker,
            conversation_history
        )
        
        # Store the record
        if current_record:
            self.trust_store.add_record(current_record)
        
        # Calculate trust score
        trust_analysis = self.trust_store.calculate_trust_analysis(user_id)
        
        # Store in broker
        broker.trust_analysis = trust_analysis
        
        logger.info(
            f"Trust analysis complete: score={trust_analysis.score:.3f}, "
            f"interactions={trust_analysis.interaction_count}"
        )
        
        return NodeResult(
            status=NodeStatus.SUCCESS,
            data={
                "trust_score": trust_analysis.score,
                "interaction_count": trust_analysis.interaction_count
            }
        )
    
    def _analyze_current_interaction(
        self,
        broker: KnowledgeBroker,
        conversation_history: list
    ) -> TrustRecord | None:
        """Analyze the current interaction for trust metrics.
        
        Args:
            broker: Knowledge broker
            conversation_history: Conversation history
            
        Returns:
            TrustRecord if analysis successful, None otherwise
        """
        # Get user ID
        user_id = None
        if hasattr(broker, "dialogue_input") and broker.dialogue_input:
            user_id = broker.dialogue_input.speaker
        
        if not user_id:
            return None
        
        # Get sentiment analysis if available
        sentiment = getattr(broker, "sentiment_analysis", None)
        
        # Determine interaction type
        interaction_type = "neutral"
        if sentiment:
            if sentiment.sentiment == "positive":
                interaction_type = "positive"
            elif sentiment.sentiment == "negative":
                interaction_type = "negative"
        
        # Calculate depth score based on content length and topics
        depth_score = self._calculate_depth_score(broker, sentiment)
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(
            user_id,
            conversation_history
        )
        
        return TrustRecord(
            timestamp=datetime.now(),
            user_id=user_id,
            interaction_type=interaction_type,
            depth_score=depth_score,
            consistency_score=consistency_score
        )
    
    def _calculate_depth_score(
        self,
        broker: KnowledgeBroker,
        sentiment
    ) -> float:
        """Calculate depth score based on interaction content.
        
        Args:
            broker: Knowledge broker
            sentiment: Sentiment analysis result
            
        Returns:
            Depth score (0.0-1.0)
        """
        depth_score = 0.0
        
        # Check for personal information sharing
        if hasattr(broker, "dialogue_input") and broker.dialogue_input:
            content = broker.dialogue_input.content.lower()
            
            # Personal topics indicate deeper sharing
            personal_topics = [
                "family", "mother", "father", "sister", "brother",
                "child", "son", "daughter", "friend", "relationship",
                "feel", "feeling", "love", "hate", "afraid",
                "worried", "anxious", "depressed", "happy", "sad"
            ]
            
            for topic in personal_topics:
                if topic in content:
                    depth_score += 0.1
            
            # Content length indicates depth
            if len(content) > 50:
                depth_score += 0.2
            if len(content) > 100:
                depth_score += 0.2
        
        # Check sentiment for emotional depth
        if sentiment and hasattr(sentiment, "emotional_tone"):
            emotional_tones = [
                "grieving", "anxious", "depressed", "excited",
                "enthusiastic", "frustrated", "hostile", "angry"
            ]
            if sentiment.emotional_tone in emotional_tones:
                depth_score += 0.2
        
        return min(1.0, depth_score)
    
    def _calculate_consistency_score(
        self,
        user_id: str,
        conversation_history: list
    ) -> float:
        """Calculate consistency score based on past interactions.
        
        Args:
            user_id: User ID
            conversation_history: Conversation history
            
        Returns:
            Consistency score (0.0-1.0)
        """
        records = self.trust_store.get_records(user_id)
        
        if len(records) < 2:
            return 0.5  # Default for new users
        
        # Calculate interaction type consistency
        recent_types = [r.interaction_type for r in records[-10:]]
        
        # Count each type
        type_counts = {}
        for interaction_type in recent_types:
            type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1
        
        # Calculate consistency (how balanced are interactions)
        total = len(recent_types)
        if total == 0:
            return 0.5
        
        # More balanced = higher consistency
        unique_types = len(type_counts)
        if unique_types == 1:
            return 0.3  # Too repetitive
        elif unique_types == 2:
            return 0.6
        else:
            return 0.9  # Good variety


class TrustStore:
    """Stores and retrieves trust analysis data."""
    
    def __init__(self):
        self.records: dict[str, list[TrustRecord]] = {}
    
    def add_record(self, record: TrustRecord) -> None:
        """Add a trust record.
        
        Args:
            record: The trust record to add
        """
        if record.user_id not in self.records:
            self.records[record.user_id] = []
        
        self.records[record.user_id].append(record)
        
        logger.debug(
            f"Added trust record for user '{record.user_id}': "
            f"type={record.interaction_type}, depth={record.depth_score:.2f}"
        )
    
    def get_records(self, user_id: str) -> list[TrustRecord]:
        """Get all records for a user.
        
        Args:
            user_id: User ID to retrieve records for
            
        Returns:
            List of trust records
        """
        return self.records.get(user_id, [])
    
    def calculate_trust_analysis(self, user_id: str) -> TrustAnalysis:
        """Calculate trust analysis for a user.
        
        Args:
            user_id: User ID to analyze
            
        Returns:
            TrustAnalysis with calculated metrics
        """
        records = self.get_records(user_id)
        
        if not records:
            return TrustAnalysis(
                score=0.0,
                relationship_age_days=0,
                interaction_count=0,
                positive_interactions=0,
                negative_interactions=0,
                consistency_score=0.5,
                reasoning="No interaction history"
            )
        
        # Calculate metrics
        interaction_count = len(records)
        positive_interactions = sum(
            1 for r in records if r.interaction_type == "positive"
        )
        negative_interactions = sum(
            1 for r in records if r.interaction_type == "negative"
        )
        
        # Calculate relationship age
        first_interaction = records[0].timestamp
        relationship_age_days = (datetime.now() - first_interaction).days
        
        # Calculate consistency score
        recent_records = records[-20:] if len(records) > 20 else records
        consistency_score = self._calculate_consistency_from_records(recent_records)
        
        # Calculate trust score
        trust_score = self._calculate_trust_score(
            relationship_age_days,
            interaction_count,
            positive_interactions,
            negative_interactions,
            consistency_score
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            trust_score,
            relationship_age_days,
            interaction_count,
            positive_interactions,
            negative_interactions
        )
        
        return TrustAnalysis(
            score=trust_score,
            relationship_age_days=relationship_age_days,
            interaction_count=interaction_count,
            positive_interactions=positive_interactions,
            negative_interactions=negative_interactions,
            consistency_score=consistency_score,
            reasoning=reasoning
        )
    
    def _calculate_consistency_from_records(
        self,
        records: list[TrustRecord]
    ) -> float:
        """Calculate consistency score from records.
        
        Args:
            records: List of trust records
            
        Returns:
            Consistency score (0.0-1.0)
        """
        if len(records) < 2:
            return 0.5
        
        # Calculate depth consistency
        depth_scores = [r.depth_score for r in records]
        avg_depth = sum(depth_scores) / len(depth_scores)
        
        # Calculate interaction type balance
        type_counts = {}
        for record in records:
            type_counts[record.interaction_type] = \
                type_counts.get(record.interaction_type, 0) + 1
        
        unique_types = len(type_counts)
        
        # Combine metrics
        depth_consistency = 1.0 - abs(avg_depth - 0.5)
        type_balance = min(1.0, unique_types / 3.0)
        
        return (depth_consistency * 0.6) + (type_balance * 0.4)
    
    def _calculate_trust_score(
        self,
        relationship_age_days: int,
        interaction_count: int,
        positive_interactions: int,
        negative_interactions: int,
        consistency_score: float
    ) -> float:
        """Calculate overall trust score.
        
        Args:
            relationship_age_days: Days since first interaction
            interaction_count: Total number of interactions
            positive_interactions: Number of positive interactions
            negative_interactions: Number of negative interactions
            consistency_score: Consistency of interactions
            
        Returns:
            Trust score (0.0-1.0)
        """
        # Age component (max 0.3)
        age_score = min(0.3, relationship_age_days / 365.0)
        
        # Frequency component (max 0.3)
        frequency_score = min(0.3, interaction_count / 100.0)
        
        # Sentiment ratio component (max 0.2)
        total_sentiment = positive_interactions + negative_interactions
        if total_sentiment > 0:
            sentiment_ratio = positive_interactions / total_sentiment
        else:
            sentiment_ratio = 0.5
        sentiment_score = abs(sentiment_ratio - 0.5) * 0.4
        
        # Consistency component (max 0.2)
        consistency_component = consistency_score * 0.2
        
        # Combine all components
        trust_score = age_score + frequency_score + \
            (0.2 - sentiment_score) + consistency_component
        
        return min(1.0, max(0.0, trust_score))
    
    def _generate_reasoning(
        self,
        trust_score: float,
        relationship_age_days: int,
        interaction_count: int,
        positive_interactions: int,
        negative_interactions: int
    ) -> str:
        """Generate reasoning for trust score.
        
        Args:
            trust_score: Calculated trust score
            relationship_age_days: Days since first interaction
            interaction_count: Total interactions
            positive_interactions: Positive interaction count
            negative_interactions: Negative interaction count
            
        Returns:
            Reasoning string
        """
        parts = []
        
        if relationship_age_days < 7:
            parts.append("New relationship")
        elif relationship_age_days < 30:
            parts.append("Early relationship")
        elif relationship_age_days < 90:
            parts.append("Developing relationship")
        else:
            parts.append("Established relationship")
        
        if interaction_count > 50:
            parts.append("frequent interactions")
        elif interaction_count > 20:
            parts.append("regular interactions")
        
        if positive_interactions > negative_interactions * 2:
            parts.append("mostly positive interactions")
        elif negative_interactions > positive_interactions * 2:
            parts.append("mostly negative interactions")
        
        if not parts:
            return "Limited interaction history"
        
        return ", ".join(parts)
