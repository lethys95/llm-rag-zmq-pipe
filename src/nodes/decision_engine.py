"""Decision engine for LLM-based node selection."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.base import BaseLLM
    from .knowledge_broker import KnowledgeBroker
    from .base import BaseNode
    from .registry import NodeRegistry

logger = logging.getLogger(__name__)


class DecisionEngine:
    """LLM-based reasoning engine to decide which nodes to run.
    
    The decision engine analyzes the incoming message and current context
    to determine which processing nodes should be executed. It can use:
    - Simple rule-based logic for common patterns
    - LLM analysis for complex decision making
    - Historical patterns and user preferences
    
    Phase 1 Implementation: Simple rule-based with optional LLM enhancement
    Future: More sophisticated LLM-driven decision making
    
    Attributes:
        llm_provider: Optional LLM for complex decision analysis
        use_llm: Whether to use LLM for decisions (vs pure rules)
    """
    
    def __init__(self, llm_provider: "BaseLLM | None" = None, use_llm: bool = False):
        """Initialize the decision engine.
        
        Args:
            llm_provider: Optional LLM provider for decision making
            use_llm: Whether to use LLM analysis (default: False, use rules)
        """
        self.llm_provider = llm_provider
        self.use_llm = use_llm
        
        if use_llm and not llm_provider:
            logger.warning("use_llm=True but no LLM provider given, using rules only")
            self.use_llm = False
        
        mode = "LLM-enhanced" if self.use_llm else "rule-based"
        logger.info(f"Decision engine initialized ({mode})")
    
    async def select_nodes(
        self,
        message: str,
        broker: "KnowledgeBroker",
        registry: "NodeRegistry | None" = None
    ) -> list[str]:
        """Select which nodes should be executed for this message.
        
        Args:
            message: The user's message
            broker: Knowledge broker with current context
            registry: Optional node registry for dynamic node creation
            
        Returns:
            List of node names to execute
        """
        logger.debug(f"Selecting nodes for message: {message[:100]}...")
        
        if self.use_llm:
            node_names = await self._llm_based_selection(message, broker)
        else:
            node_names = self._rule_based_selection(message, broker)
        
        logger.info(f"Selected {len(node_names)} nodes: {node_names}")
        return node_names
    
    def _rule_based_selection(
        self,
        message: str,
        broker: "KnowledgeBroker"
    ) -> list[str]:
        """Rule-based node selection (Phase 1 implementation).
        
        Simple pattern matching and context checking to determine nodes.
        
        Args:
            message: The user's message
            broker: Knowledge broker with current context
            
        Returns:
            List of node names to execute
        """
        nodes = []
        message_lower = message.lower()
        
        # Crisis detection (highest priority)
        crisis_keywords = [
            "suicide", "kill myself", "end it all", "don't want to live",
            "hurt myself", "self harm", "can't go on"
        ]
        if any(keyword in message_lower for keyword in crisis_keywords):
            logger.warning("Crisis keywords detected in message")
            nodes.append("crisis_detection")
        
        # Always run for conversation
        nodes.append("sentiment_analysis")
        nodes.append("primary_response")
        
        # Check for idle time to trigger detox protocol
        idle_time_minutes = broker.get_knowledge("idle_time_minutes", 0)
        if idle_time_minutes and idle_time_minutes > 60:
            logger.info(f"User idle for {idle_time_minutes} minutes, adding detox")
            nodes.append("detox_protocol")
        
        # Contextual additions based on message content
        # (These are placeholders for future nodes)
        
        # If asking questions, might need deeper context
        if "?" in message or any(word in message_lower for word in ["why", "how", "what", "when"]):
            # Could add deeper RAG retrieval node here
            pass
        
        # If expressing emotions, might need emotional support strategy
        emotion_keywords = ["feel", "feeling", "felt", "emotion", "sad", "happy", "angry", "anxious"]
        if any(keyword in message_lower for keyword in emotion_keywords):
            # Could add needs analysis or strategy selection node here
            pass
        
        return nodes
    
    async def _llm_based_selection(
        self,
        message: str,
        broker: "KnowledgeBroker"
    ) -> list[str]:
        """LLM-based node selection (Future enhancement).
        
        Uses lightweight LLM to analyze message and context to determine
        which nodes would be most beneficial.
        
        Args:
            message: The user's message
            broker: Knowledge broker with current context
            
        Returns:
            List of node names to execute
        """
        logger.debug("Using LLM for node selection")
        
        # Build prompt for LLM
        prompt = self._build_selection_prompt(message, broker)
        
        try:
            # Quick LLM call for decision making
            response = await self.llm_provider.generate_async(prompt)
            
            # Parse LLM response (expecting JSON list of node names)
            node_names = self._parse_llm_response(response)
            
            # Validate and filter nodes
            validated_nodes = self._validate_node_selection(node_names)
            
            return validated_nodes
        
        except Exception as e:
            logger.error(f"LLM node selection failed: {e}, falling back to rules")
            return self._rule_based_selection(message, broker)
    
    def _build_selection_prompt(
        self,
        message: str,
        broker: "KnowledgeBroker"
    ) -> str:
        """Build prompt for LLM-based node selection.
        
        Args:
            message: The user's message
            broker: Knowledge broker with current context
            
        Returns:
            Formatted prompt for LLM
        """
        # Get context summary
        context = broker.get_full_context()
        
        prompt = f"""
You are a decision engine for an AI companion system. Analyze the user's message
and current context to determine which processing nodes should be executed.

Available Nodes:
- crisis_detection: Detect safety concerns (suicide, self-harm, etc.)
- sentiment_analysis: Analyze emotional tone and sentiment
- needs_analysis: Identify psychological needs (future)
- primary_response: Generate the main response
- detox_protocol: Background self-correction (future)
- rag_retrieval: Retrieve relevant memories (future)

User Message: {message}

Context Summary:
- Previous sentiment: {context.get('sentiment_analysis', 'unknown')}
- Idle time: {context.get('idle_time_minutes', 0)} minutes

Respond with a JSON array of node names to execute, in priority order.
Example: ["crisis_detection", "sentiment_analysis", "primary_response"]

Your response (JSON only):"""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> list[str]:
        """Parse LLM response to extract node names.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of node names
        """
        import json
        
        try:
            # Try to extract JSON from response
            start = response.find('[')
            end = response.rfind(']') + 1
            
            if start != -1 and end > start:
                json_str = response[start:end]
                node_names = json.loads(json_str)
                return node_names
            
            logger.warning("No JSON array found in LLM response")
            return []
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return []
    
    def _validate_node_selection(self, node_names: list[str]) -> list[str]:
        """Validate and filter node names.
        
        Ensures node names are valid and reasonable.
        
        Args:
            node_names: List of node names from LLM
            
        Returns:
            Filtered list of valid node names
        """
        # Known valid nodes (for Phase 1)
        valid_nodes = {
            "crisis_detection",
            "sentiment_analysis",
            "primary_response",
            "detox_protocol",
            "rag_retrieval",
            "needs_analysis",
        }
        
        validated = [name for name in node_names if name in valid_nodes]
        
        if len(validated) != len(node_names):
            invalid = set(node_names) - set(validated)
            logger.warning(f"Filtered out invalid nodes: {invalid}")
        
        # Ensure primary_response is always included
        if "primary_response" not in validated:
            validated.append("primary_response")
            logger.debug("Added primary_response (always required)")
        
        return validated
    
    def add_conditional_logic(
        self,
        condition_name: str,
        checker: callable,
        nodes: list[str]
    ) -> None:
        """Add custom conditional logic for node selection.
        
        Future enhancement: Allow registering custom conditions.
        
        Args:
            condition_name: Name of the condition
            checker: Callable that returns True if condition is met
            nodes: List of node names to add if condition is True
        """
        # Future implementation for extensible decision logic
        pass
