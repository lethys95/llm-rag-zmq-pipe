"""Detox scheduler for triggering detox sessions during idle time."""

import asyncio
import logging
from datetime import datetime, timedelta

from src.nodes.algo_nodes.memory_consolidation_node import MemoryConsolidationNode
from src.nodes.core.base import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.rag.algorithms.nudging_algorithm import NudgingAlgorithm
from src.llm.base import BaseLLM
from src.rag.base import BaseRAG
from src.rag.embeddings import EmbeddingService
from src.chrono.task_scheduler import TaskScheduler

logger = logging.getLogger(__name__)


class DetoxScheduler:
    """Manages detox session timing and triggers.
    
    The detox scheduler monitors user activity and triggers detox sessions
    during idle periods. This allows the AI companion to:
    - Re-evaluate memories with fresh perspective
    - Consolidate similar memories
    - Apply nudging algorithm to prevent sycophancy
    - Prune irrelevant memories
    """
    
    def __init__(
        self,
        llm_provider: BaseLLM,
        rag_provider: BaseRAG,
        task_scheduler: TaskScheduler,
        idle_trigger_minutes: int = 60,
        min_session_interval_minutes: int = 120,
        max_session_duration_minutes: int = 30
    ):
        """Initialize the detox scheduler.
        
        Args:
            llm_provider: LLM provider for detox operations
            rag_provider: RAG provider for memory operations
            task_scheduler: Task scheduler for scheduling detox sessions
            idle_trigger_minutes: Minutes of inactivity before triggering detox
            min_session_interval_minutes: Minimum minutes between detox sessions
            max_session_duration_minutes: Maximum duration of a detox session
        """
        self.llm_provider = llm_provider
        self.rag_provider = rag_provider
        self.task_scheduler = task_scheduler
        self.idle_trigger_minutes = idle_trigger_minutes
        self.min_session_interval_minutes = min_session_interval_minutes
        self.max_session_duration_minutes = max_session_duration_minutes
        
        self.last_activity = datetime.now()
        self.last_detox_session = None
        self.is_detox_running = False
        
        # Create nudging algorithm and memory consolidation node
        self.nudging_algorithm = NudgingAlgorithm(
            nudge_strength=0.15,
            max_companion_drift=0.3,
            base_user_influence=0.3,
            base_companion_influence=0.7,
            max_trust_boost=0.3
        )
        self.memory_consolidation = MemoryConsolidationNode(
            llm_provider=llm_provider,
            rag_provider=rag_provider,
            consolidation_threshold=0.7,
            max_memories_per_batch=10
        )
        
        logger.info(
            f"Detox scheduler initialized: "
            f"idle_trigger={idle_trigger_minutes}m, "
            f"min_interval={min_session_interval_minutes}m, "
            f"max_duration={max_session_duration_minutes}m"
        )
    
    def schedule_detox_session(self) -> None:
        """Schedule periodic detox session checks.
        
        This method schedules a periodic task that checks if detox conditions
        are met and runs a detox session if appropriate.
        """
        async def detox_check_task():
            """Periodic task to check and run detox sessions."""
            while self.task_scheduler.running:
                if self.should_run_detox():
                    await self._run_detox_session()
                await asyncio.sleep(60)  # Check every minute
        
        self.task_scheduler.schedule_periodic_task(
            name="detox_check",
            task_func=detox_check_task,
            interval_seconds=60
        )
        logger.info("Detox session check scheduled")
    
    async def _run_detox_session(self) -> None:
        """Run a complete detox session."""
        self.start_detox_session()
        
        try:
            # Create detox session node
            detox_node = DetoxSessionNode(
                nudging_algorithm=self.nudging_algorithm,
                memory_consolidation_node=self.memory_consolidation,
                rag_provider=self.rag_provider,
                max_session_duration_minutes=self.max_session_duration_minutes
            )
            
            # Create knowledge broker
            broker = KnowledgeBroker()
            
            # Execute detox session
            result = await detox_node.execute(broker)
            
            logger.info(
                f"Detox session completed: status={result.status}, "
                f"data={result.data}"
            )
        finally:
            self.end_detox_session()
    
    def update_activity(self) -> None:
        """Update last activity timestamp.
        
        Call this whenever the user interacts with the system.
        """
        self.last_activity = datetime.now()
        logger.debug("Activity updated")
    
    def should_run_detox(self) -> bool:
        """Check if detox session should be triggered.
        
        Returns:
            True if conditions are met for running detox
        """
        # Don't run if already running
        if self.is_detox_running:
            logger.debug("Detox already running, skipping")
            return False
        
        # Check idle time
        idle_time = datetime.now() - self.last_activity
        idle_minutes = idle_time.total_seconds() / 60.0
        
        if idle_minutes < self.idle_trigger_minutes:
            logger.debug(f"User not idle enough ({idle_minutes:.1f}m < {self.idle_trigger_minutes}m)")
            return False
        
        # Check minimum interval since last session
        if self.last_detox_session:
            time_since_last = datetime.now() - self.last_detox_session
            interval_minutes = time_since_last.total_seconds() / 60.0
            
            if interval_minutes < self.min_session_interval_minutes:
                logger.debug(
                    f"Too soon since last detox ({interval_minutes:.1f}m < "
                    f"{self.min_session_interval_minutes}m)"
                )
                return False
        
        logger.info(
            f"Detox conditions met: idle={idle_minutes:.1f}m, "
            f"since_last={interval_minutes if self.last_detox_session else 'N/A'}m"
        )
        return True
    
    def start_detox_session(self) -> None:
        """Mark detox session as started."""
        self.is_detox_running = True
        self.last_detox_session = datetime.now()
        logger.info("Detox session started")
    
    def end_detox_session(self) -> None:
        """Mark detox session as ended."""
        self.is_detox_running = False
        logger.info("Detox session ended")
    
    def get_idle_time(self) -> timedelta:
        """Get current idle time.
        
        Returns:
            Time since last activity
        """
        return datetime.now() - self.last_activity
    
    def get_time_until_next_detox(self) -> timedelta | None:
        """Get time until next detox session can run.
        
        Returns:
            Time until next detox, or None if ready now
        """
        if self.should_run_detox():
            return timedelta(0)
        
        # Calculate time until idle trigger
        idle_time = self.get_idle_time()
        idle_needed = timedelta(minutes=self.idle_trigger_minutes) - idle_time
        
        # Calculate time until interval trigger
        if self.last_detox_session:
            interval_needed = (
                timedelta(minutes=self.min_session_interval_minutes) -
                (datetime.now() - self.last_detox_session)
            )
            return max(idle_needed, interval_needed)
        
        return idle_needed


class DetoxSessionNode(BaseNode):
    """Node that orchestrates a complete detox session.
    
    This node runs the full detox protocol:
    1. Identify topics discussed
    2. Run nudging algorithm for each topic
    3. Store companion's recalibrated positions
    4. Generate guidance for next conversation
    """
    
    def __init__(
        self,
        nudging_algorithm: NudgingAlgorithm,
        memory_consolidation_node: MemoryConsolidationNode,
        rag_provider: BaseRAG,
        max_session_duration_minutes: int = 30,
        **kwargs
    ):
        """Initialize detox session node.
        
        Args:
            nudging_algorithm: The nudging algorithm instance
            memory_consolidation_node: Memory consolidation node
            rag_provider: RAG provider for storing companion state
            max_session_duration_minutes: Maximum session duration
            **kwargs: Additional arguments passed to BaseNode
        """
        super().__init__(
            name="detox_session",
            priority=10,
            queue_type="background",
            **kwargs
        )
        self.nudging_algorithm = nudging_algorithm
        self.memory_consolidation = memory_consolidation_node
        self.rag_provider = rag_provider
        self.embedding_service = EmbeddingService.get_instance()
        self.max_session_duration_minutes = max_session_duration_minutes
        
        logger.info(
            f"Detox session node initialized "
            f"(max_duration={max_session_duration_minutes}m)"
        )
    
    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        """Execute complete detox session.
        
        Args:
            broker: Knowledge broker
            
        Returns:
            NodeResult with detox results
        """
        logger.info("Starting detox session")
        
        # Get conversation history
        conversation_history = getattr(broker, "conversation_history", [])
        
        if not conversation_history:
            logger.debug("No conversation history for detox")
            return NodeResult(
                status=NodeStatus.SKIPPED,
                metadata={"reason": "no_conversation_history"}
            )
        
        # Step 1: Identify topics discussed
        topics = self._extract_topics(conversation_history)
        logger.info(f"Identified {len(topics)} topics for detox")
        
        # Step 2: Run nudging algorithm for each topic
        nudges = []
        for topic, user_position in topics:
            # Get trust score if available
            trust_score = 0.0
            if hasattr(broker, "trust_analysis") and broker.trust_analysis:
                trust_score = getattr(broker.trust_analysis, "score", 0.0)
            
            nudge = self.nudging_algorithm.calculate_nudge(
                topic=topic,
                user_position=user_position,
                trust_score=trust_score
            )
            
            if nudge:
                nudges.append(nudge)
                logger.info(
                    f"Calculated nudge for topic '{topic}': "
                    f"companion {nudge['companion_before']:.3f} -> "
                    f"{nudge['companion_after']:.3f}"
                )
        
        # Step 3: Store companion's recalibrated positions
        for nudge in nudges:
            await self._store_companion_state(
                topic=nudge["topic"],
                position=nudge["companion_after"]
            )
        
        # Step 4: Run memory consolidation
        consolidation_result = await self.memory_consolidation.execute(broker)
        
        # Step 5: Generate guidance for next conversation
        guidance = self._generate_conversational_guidance(nudges)
        
        # Store results in broker
        broker.detox_results = {
            "nudges_applied": nudges,
            "guidance": guidance,
            "total_extremism_reduction": sum(
                n.get("extremism_reduction", 0.0) for n in nudges
            ),
            "consolidation_results": getattr(
                broker, "consolidation_results", {}
            )
        }
        
        logger.info(
            f"Detox session complete: {len(nudges)} nudges, "
            f"extremism reduction: {broker.detox_results['total_extremism_reduction']:.3f}"
        )
        
        return NodeResult(
            status=NodeStatus.SUCCESS,
            data={
                "nudges_count": len(nudges),
                "extremism_reduction": broker.detox_results["total_extremism_reduction"]
            }
        )
    
    def _extract_topics(self, conversation_history: list) -> list[tuple[str, float]]:
        """Extract topics and user positions from conversation history.
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            List of (topic, user_position) tuples
        """
        topics = []
        
        for msg in conversation_history:
            # Get key topics from message
            if hasattr(msg, "key_topics") and msg.key_topics:
                for topic in msg.key_topics:
                    # Estimate user position based on sentiment
                    user_position = self._estimate_user_position(msg)
                    topics.append((topic, user_position))
        
        # Deduplicate topics
        seen = set()
        unique_topics = []
        for topic, position in topics:
            if topic not in seen:
                seen.add(topic)
                unique_topics.append((topic, position))
        
        return unique_topics
    
    def _estimate_user_position(self, message) -> float:
        """Estimate user's position on a topic based on sentiment.
        
        Args:
            message: The message to analyze
            
        Returns:
            Estimated position (-1.0 to 1.0)
        """
        # Simple heuristic based on sentiment
        sentiment = getattr(message, "sentiment", "neutral")
        emotional_tone = getattr(message, "emotional_tone", None)
        
        # Map sentiment to position
        if sentiment == "negative":
            # Negative sentiment could indicate extreme position
            if emotional_tone in ["angry", "frustrated", "hostile"]:
                return -0.7  # Extreme negative
            elif emotional_tone in ["sad", "grieving", "depressed"]:
                return -0.3  # Moderate negative
            else:
                return -0.5  # Default negative
        elif sentiment == "positive":
            # Positive sentiment could indicate extreme position
            if emotional_tone in ["excited", "enthusiastic", "euphoric"]:
                return 0.7  # Extreme positive
            else:
                return 0.3  # Moderate positive
        else:
            return 0.0  # Neutral
    
    async def _store_companion_state(self, topic: str, position: float) -> None:
        """Store companion's recalibrated position for a topic.
        
        Args:
            topic: The topic being stored
            position: The companion's new position
        """
        try:
            # Create document content
            content = f"Companion position on {topic}: {position:.3f}"
            
            # Generate embedding
            embedding = self.embedding_service.encode(content)
            
            # Create metadata
            metadata = {
                "type": "companion_state",
                "topic": topic,
                "position": position,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in RAG
            point_id = self.rag_provider.store(
                text=content,
                embedding=embedding,
                metadata=metadata
            )
            
            logger.info(
                f"Stored companion state: topic='{topic}', position={position:.3f}, "
                f"point_id={point_id}"
            )
        except Exception as e:
            logger.error(f"Error storing companion state: {e}", exc_info=True)
    
    def _generate_conversational_guidance(self, nudges: list) -> str:
        """Generate guidance for next conversation based on nudges.
        
        Args:
            nudges: List of applied nudges
            
        Returns:
            Guidance string for next conversation
        """
        if not nudges:
            return "No specific guidance needed."
        
        # Find topics with significant nudges
        significant_nudges = [
            n for n in nudges
            if abs(n.get("nudge_amount", 0.0)) > 0.05
        ]
        
        if not significant_nudges:
            return "Continue with current approach."
        
        # Build guidance
        topics = [n["topic"] for n in significant_nudges]
        guidance = (
            f"Consider gently introducing alternative perspectives on: "
            f"{', '.join(topics)}. "
            f"Remember to maintain a balanced, grounded approach."
        )
        
        return guidance
