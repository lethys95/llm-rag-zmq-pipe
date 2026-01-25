"""Decision engine for LLM-based node selection."""

import logging
import json

from src.llm.base import BaseLLM, ToolCall, LLMResponse
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.registry import NodeRegistry

logger = logging.getLogger(__name__)


class DecisionEngine:
    """LLM-based reasoning engine to decide which nodes to run.
    
    The decision engine analyzes the incoming message and current context
    to determine which processing nodes should be executed. It can use:
    - Simple rule-based logic for common patterns
    - LLM analysis for complex decision making
    - Function calling for structured, reliable LLM-based decisions
    
    Phase 1 Implementation: Simple rule-based with optional LLM enhancement
    Phase 2 Implementation: Function calling support for reliable LLM decisions
    Phase 3 Implementation: Node metadata and conditional logic with needs analysis
    Future: More sophisticated LLM-driven decision making
    """
    
    def __init__(self, llm_provider: BaseLLM | None = None, use_llm: bool = False):
        """Initialize the decision engine.
        
        Args:
            llm_provider: Optional LLM for complex decision analysis
            use_llm: Whether to use LLM analysis (default: False, use rules)
        """
        self.llm_provider = llm_provider
        self.use_llm = use_llm
        
        if use_llm and not llm_provider:
            logger.warning("use_llm=True but no LLM provider given, using rules only")
            self.use_llm = False
        
        mode = "LLM-enhanced" if self.use_llm else "rule-based"
        logger.info(f"Decision engine initialized ({mode})")
        
        # Conditional rules storage
        self.conditional_rules = []
        
        # Node dependencies (nodes that must run before this one)
        self.node_dependencies = {
            "memory_evaluator": ["sentiment_analysis"],
            "needs_analysis": ["memory_evaluator", "sentiment_analysis"],
            "trust_analysis": [],  # Can run independently
            "primary_response": ["sentiment_analysis", "memory_evaluator", "needs_analysis"],  # Must run after analysis
        }
    
    async def select_nodes(
        self,
        message: str,
        broker: KnowledgeBroker,
        registry: NodeRegistry | None = None
    ) -> list[str]:
        """Select which nodes should be executed for this message.
        
        Args:
            message: The user's message
            broker: Knowledge broker with current context
            registry: Optional node registry for dynamic node discovery
            
        Returns:
            List of node names to execute
        """
        logger.debug(f"Selecting nodes for message: {message[:100]}...")
        
        if self.use_llm:
            node_names = await self._llm_based_selection(message, broker, registry)
        else:
            node_names = self._rule_based_selection(message, broker, registry)
        
        # Ensure dependencies are met
        node_names = self._resolve_dependencies(node_names, broker, registry)
        
        logger.info(f"Selected {len(node_names)} nodes: {node_names}")
        return node_names
    
    def _resolve_dependencies(
        self,
        node_names: list[str],
        broker: KnowledgeBroker,
        registry: NodeRegistry
    ) -> list[str]:
        """Ensure node dependencies are met.
        
        Args:
            node_names: Ordered list of node names
            broker: Knowledge broker with execution history
            registry: Node registry to check availability
            
        Returns:
            Ordered list with dependencies resolved
        """
        available_nodes = set(registry.list_available()) if registry else set()
        executed = set(broker.metadata.execution_order)
        
        # Build dependency graph
        resolved = []
        seen = set()
        
        for node_name in node_names:
            if node_name in seen:
                continue
            
            # Check if dependencies are met
            if node_name in self.node_dependencies:
                dependencies = self.node_dependencies[node_name]
                for dep in dependencies:
                    if dep not in executed:
                        # Add dependency first
                        if dep in available_nodes and dep not in seen:
                            resolved.append(dep)
                            seen.add(dep)
                            executed.add(dep)
            
            # Now add the node itself
            if node_name in available_nodes and node_name not in seen:
                resolved.append(node_name)
                seen.add(node_name)
                executed.add(node_name)
        
        # primary_response must always be last
        if "primary_response" in node_names and "primary_response" not in resolved:
            if "primary_response" in available_nodes:
                resolved.append("primary_response")
                seen.add("primary_response")
        
        return resolved
    
    def _get_available_nodes(self, registry: NodeRegistry) -> dict[str, str]:
        """Get available nodes with descriptions from registry.
        
        Args:
            registry: NodeRegistry instance
            
        Returns:
            Dictionary mapping node names to descriptions
        """
        nodes = {}
        for name in registry.list_available():
            try:
                info = registry.get_node_info(name)
                description = self._extract_node_description(info)
                nodes[name] = description
            except KeyError:
                logger.warning(f"Could not get info for node '{name}'")
                nodes[name] = "No description available"
        return nodes

    def _extract_node_description(self, info) -> str:
        """Extract description from node info.
        
        Args:
            info: NodeInfo dataclass
            
        Returns:
            Description string
        """
        if info.docstring:
            lines = info.docstring.strip().split('\n')
            first_line = lines[0].strip()
            for prefix in ['"""', "'''", '"""', "'''"]:
                if first_line.startswith(prefix):
                    first_line = first_line[len(prefix):].strip()
            return first_line
        return "No description available"

    def _rule_based_selection(
        self,
        message: str,
        broker: KnowledgeBroker,
        registry: NodeRegistry | None = None
    ) -> list[str]:
        """Rule-based node selection (Phase 3 implementation).
        
        Simple pattern matching and context checking to determine nodes.
        Ensures proper execution order based on dependencies.
        
        Args:
            message: The user's message
            broker: Knowledge broker with current context
            registry: Optional node registry for dynamic node discovery
            
        Returns:
            List of node names to execute
        """
        nodes = []
        message_lower = message.lower()
        
        # Get available nodes from registry if available
        available_nodes = set(registry.list_available()) if registry else set()
        
        # Apply conditional rules
        for rule in self.conditional_rules:
            if rule['checker'](message, broker):
                nodes.extend(rule['nodes'])
                logger.debug(f"Conditional rule '{rule['name']}' matched, adding nodes: {rule['nodes']}")
        
        # Crisis detection (highest priority)
        crisis_keywords = [
            "suicide", "kill myself", "end it all", "don't want to live",
            "hurt myself", "self harm", "can't go on"
        ]
        if any(keyword in message_lower for keyword in crisis_keywords):
            logger.warning("Crisis keywords detected in message")
            if "crisis_detection" in available_nodes or not registry:
                nodes.append("crisis_detection")
        
        # Always run sentiment analysis first (foundation for other nodes)
        if "sentiment_analysis" in available_nodes or not registry:
            nodes.append("sentiment_analysis")
        
        # Memory evaluator (after sentiment, if documents retrieved)
        if hasattr(broker, "retrieved_documents") and broker.retrieved_documents:
            if "memory_evaluator" in available_nodes or not registry:
                nodes.append("memory_evaluator")
        
        # Needs analysis (after memory evaluation)
        if "needs_analysis" in available_nodes or not registry:
            nodes.append("needs_analysis")
        
        # Trust analysis (can run independently)
        message_count = len(broker.conversation_history) if hasattr(broker, "conversation_history") and broker.conversation_history else 0
        if message_count == 0 or message_count % 10 == 0:
            if "trust_analysis" in available_nodes or not registry:
                nodes.append("trust_analysis")
        
        # PRIMARY RESPONSE MUST BE LAST
        if "primary_response" in available_nodes or not registry:
            nodes.append("primary_response")
        
        return nodes
    
    async def _llm_based_selection(
        self,
        message: str,
        broker: KnowledgeBroker,
        registry: NodeRegistry | None = None
    ) -> list[str]:
        """LLM-based node selection with function calling support.
        
        Uses function calling for reliable, structured node selection.
        Falls back to JSON parsing if function calling is not supported.
        
        Args:
            message: The user's message
            broker: Knowledge broker with current context
            registry: Optional node registry for dynamic node discovery
            
        Returns:
            List of node names to execute
        """
        logger.debug("Using LLM for node selection")
        
        # Check if LLM supports function calling and registry is available
        if registry and self._supports_function_calling():
            return await self._llm_selection_with_function_calling(message, broker, registry)
        else:
            return await self._llm_selection_with_json_parsing(message, broker, registry)
    
    def _supports_function_calling(self) -> bool:
        """Check if LLM provider supports function calling.
        
        Returns:
            True if function calling is supported
        """
        return (
            self.llm_provider is not None and
            hasattr(self.llm_provider, 'generate_with_tools')
        )
    
    async def _llm_selection_with_function_calling(
        self,
        message: str,
        broker: KnowledgeBroker,
        registry: NodeRegistry
    ) -> list[str]:
        """LLM-based node selection using function calling.
        
        Args:
            message: The user's message
            broker: Knowledge broker with current context
            registry: NodeRegistry for dynamic node discovery
            
        Returns:
            List of node names to execute
        """
        logger.debug("Using function calling for node selection")
        
        # Build tools schema
        tools = self._build_tools_schema(registry)
        
        # Build prompt
        prompt = self._build_selection_prompt(message, broker, registry)
        
        try:
            # Call LLM with function calling
            response: LLMResponse = self.llm_provider.generate_with_tools(
                prompt,
                tools,
                tool_choice={"type": "function", "function": {"name": "select_nodes"}}
            )
            
            # Parse tool calls
            node_names = self._parse_tool_calls(response)
            
            # Validate and filter nodes
            validated_nodes = self._validate_node_selection(node_names, registry)
            
            return validated_nodes
            
        except Exception as e:
            logger.error(f"Function calling failed: {e}, falling back to JSON parsing")
            # Fall back to JSON parsing
            return await self._llm_selection_with_json_parsing(message, broker, registry)
    
    async def _llm_selection_with_json_parsing(
        self,
        message: str,
        broker: KnowledgeBroker,
        registry: NodeRegistry | None = None
    ) -> list[str]:
        """LLM-based node selection using JSON parsing.
        
        Args:
            message: The user's message
            broker: Knowledge broker with current context
            registry: Optional node registry for dynamic node discovery
            
        Returns:
            List of node names to execute
        """
        logger.debug("Using JSON parsing for node selection")
        
        # Build prompt
        prompt = self._build_selection_prompt(message, broker, registry)
        
        try:
            # Call LLM (synchronous - in production, use async)
            if hasattr(self.llm_provider, 'generate_async'):
                response = await self.llm_provider.generate_async(prompt)
            else:
                response = self.llm_provider.generate(prompt)
            
            # Parse LLM response (expecting JSON list of node names)
            node_names = self._parse_llm_response(response)
            
            # Validate and filter nodes
            validated_nodes = self._validate_node_selection(node_names, registry)
            
            return validated_nodes
            
        except Exception as e:
            logger.error(f"LLM node selection failed: {e}, falling back to rules")
            return self._rule_based_selection(message, broker, registry)
    
    def _build_tools_schema(self, registry: NodeRegistry) -> list[dict]:
        """Build tools schema for function calling.
        
        Args:
            registry: NodeRegistry instance
            
        Returns:
            List of tool definitions following OpenAI format
        """
        available_nodes = self._get_available_nodes(registry)
        node_names = list(available_nodes.keys())
        
        tool = {
            "type": "function",
            "function": {
                "name": "select_nodes",
                "description": "Select which processing nodes to execute for the user's message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nodes": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": node_names
                            },
                            "description": "List of node names to execute, in priority order"
                        }
                    },
                    "required": ["nodes"]
                }
            }
        }
        
        return [tool]
    
    def _parse_tool_calls(self, response: LLMResponse) -> list[str]:
        """Parse tool calls from LLM response.
        
        Args:
            response: LLMResponse with tool_calls
            
        Returns:
            List of node names
        """
        if not response.tool_calls:
            logger.warning("No tool calls in response")
            return []
        
        try:
            tool_call = response.tool_calls[0]
            
            if tool_call.function_name == "select_nodes":
                args = tool_call.arguments
                # Handle both direct dict and nested structure
                if "arguments" in args:
                    args_str = args["arguments"]
                elif "nodes" in args:
                    args_str = json.dumps({"nodes": args["nodes"]})
                else:
                    args_str = json.dumps(args)
                
                parsed = json.loads(args_str)
                nodes = parsed.get("nodes", [])
                
                logger.debug(f"Parsed tool call with nodes: {nodes}")
                return nodes
            else:
                logger.warning(f"Unexpected tool call: {tool_call.function_name}")
                return []
                
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse tool call: {e}")
            return []
    
    def _build_selection_prompt(
        self,
        message: str,
        broker: KnowledgeBroker,
        registry: NodeRegistry | None = None
    ) -> str:
        """Build prompt for LLM-based node selection.
        
        Args:
            message: The user's message
            broker: Knowledge broker with current context
            registry: Optional NodeRegistry for dynamic node discovery
            
        Returns:
            Formatted prompt for LLM
        """
        sentiment_str = str(broker.sentiment_analysis.sentiment) if broker.sentiment_analysis else 'unknown'
        idle_minutes = broker.idle_time_minutes or 0
        
        # Get available nodes dynamically
        if registry:
            available_nodes = self._get_available_nodes(registry)
            nodes_section = "\n".join(
                f"- {name}: {description}"
                for name, description in sorted(available_nodes.items())
            )
        else:
            # Fallback to hardcoded list
            logger.warning("Using hardcoded node list in prompt - registry not provided")
            nodes_section = """- sentiment_analysis: Analyze emotional tone and sentiment
- primary_response: Generate the main response
- memory_evaluator: Evaluate memory importance in current context
- trust_analysis: Analyze user trust and relationship maturity
- needs_analysis: Analyze psychological needs using Maslow's hierarchy
- detox_scheduler: Schedule detox protocol sessions
- detox_session: Execute detox protocol
- ack_preparation: Prepare acknowledgment message
- store_conversation: Store conversation in database"""
        
        prompt = f"""You are a decision engine for an AI companion system. Analyze the user's message
and current context to determine which processing nodes should be executed.

Available Nodes:
{nodes_section}

User Message: {message}

Context Summary:
- Previous sentiment: {sentiment_str}
- Idle time: {idle_minutes} minutes

Respond with a JSON array of node names to execute, in priority order.
Example: ["sentiment_analysis", "primary_response"]

Your response (JSON only):"""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> list[str]:
        """Parse LLM response to extract node names.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of node names
        """
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
    
    def _validate_node_selection(
        self,
        node_names: list[str],
        registry: NodeRegistry | None = None
    ) -> list[str]:
        """Validate and filter node names.
        
        Ensures node names are valid and reasonable.
        
        Args:
            node_names: List of node names from LLM
            registry: Optional NodeRegistry for dynamic validation
            
        Returns:
            Filtered list of valid node names
        """
        # Use registry if available, otherwise use hardcoded fallback
        if registry:
            valid_nodes = set(registry.list_available())
        else:
            # Fallback to hardcoded list for backward compatibility
            valid_nodes = {
                "sentiment_analysis",
                "primary_response",
                "memory_evaluator",
                "trust_analysis",
                "needs_analysis",
                "detox_scheduler",
                "detox_session",
                "ack_preparation",
                "store_conversation",
            }
            logger.warning("Using hardcoded node list - registry not provided")
        
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
        nodes: list[str],
        priority: int = 0
    ) -> None:
        """Add custom conditional logic for node selection.
        
        Future enhancement: Allow registering custom conditions.
        
        Args:
            condition_name: Name of the condition
            checker: Callable that returns True if condition is met
            nodes: List of node names to add if condition is True
            priority: Priority for this condition (higher = evaluated first)
        """
        self.conditional_rules.append({
            'name': condition_name,
            'checker': checker,
            'nodes': nodes,
            'priority': priority
        })
        
        # Sort by priority (higher first)
        self.conditional_rules.sort(key=lambda x: x['priority'], reverse=True)
        logger.info(f"Added conditional rule '{condition_name}' with priority {priority}")