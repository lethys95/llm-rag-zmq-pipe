"""Memory evaluator node for AI-driven memory importance assessment."""

import asyncio
import json
import logging
from textwrap import dedent

from src.llm.base import BaseLLM
from src.models.memory import ConversationState
from src.models.analysis import MemoryEvaluation
from src.nodes.core.base_node import BaseNode
from src.nodes.orchestration.node_registry_decorator import register_node
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.rag.selector import RAGDocument

logger = logging.getLogger(__name__)


@register_node
class MemoryEvaluatorNode(BaseNode):
    """AI-driven memory importance evaluator.

    This node uses an LLM to re-evaluate memory importance in the context of:
    - Current conversation state
    - User relationship maturity (trust score)
    - Recent access patterns
    - Emotional salience

    The evaluation adjusts memory scores dynamically, allowing the system to
    adapt to changing contexts and relationship dynamics.
    """

    SYSTEM_PROMPT = dedent("""
        You are a memory importance evaluator for an AI companion. Your task is to
        evaluate how important a specific memory is in the current context.
        
        Consider the following factors:
        1. Emotional impact: Is this memory still emotionally significant?
        2. Relationship context: How close is the user to this topic?
        3. Recency of access: Has this memory been discussed recently?
        4. Trust level: How much should this influence future interactions?
        5. Current conversation: Is this relevant to what's being discussed now?
        
        Return a JSON response with the following format:
        {
            "relevance": 0.0-1.0,
            "chrono_relevance": 0.0-1.0,
            "reasoning": "brief explanation of your evaluation",
            "should_boost": true or false,
            "boost_factor": 0.0-1.0
        }
        
        IMPORTANT: Respond ONLY with valid JSON. No explanations, no additional text.
    """)

    def __init__(
        self,
        llm_provider: BaseLLM,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        **kwargs,
    ):
        """Initialize the memory evaluator node.

        Args:
            llm_provider: The LLM provider to use for evaluation
            max_retries: Maximum number of retry attempts on failure
            retry_delay: Delay in seconds between retries
            **kwargs: Additional arguments passed to BaseNode
        """
        super().__init__(
            name="memory_evaluator", priority=2, queue_type="immediate", **kwargs
        )
        self.llm = llm_provider
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        logger.info(
            f"Memory evaluator node initialized "
            f"(max_retries={max_retries}, retry_delay={retry_delay}s)"
        )

    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        """Execute memory evaluation on retrieved documents.

        Args:
            broker: Knowledge broker containing retrieved documents

        Returns:
            NodeResult with evaluated memories
        """
        # Get documents to evaluate
        documents = getattr(broker, "retrieved_documents", None)

        if not documents:
            logger.debug("No documents to evaluate")
            return NodeResult(
                status=NodeStatus.SKIPPED, metadata={"reason": "no_documents"}
            )

        # Get trust score if available
        trust_score = 0.0
        if hasattr(broker, "trust_analysis") and broker.trust_analysis:
            trust_score = getattr(broker.trust_analysis, "score", 0.0)

        # Get conversation state
        conversation_state = self._get_conversation_state(broker)

        # Get current message if available
        current_message = None
        if hasattr(broker, "dialogue_input") and broker.dialogue_input:
            current_message = broker.dialogue_input.content

        # Evaluate each document
        evaluated_docs = []
        for doc in documents:
            evaluation = await self._evaluate_memory(
                doc, trust_score, conversation_state, current_message
            )

            if evaluation:
                evaluated_docs.append((doc, evaluation))
                logger.debug(
                    f"Evaluated document: relevance={evaluation.relevance:.2f}, "
                    f"chrono={evaluation.chrono_relevance:.2f}, "
                    f"boost={evaluation.should_boost}"
                )
            else:
                logger.warning(f"Failed to evaluate document: {doc.content[:50]}...")

        # Store evaluated documents in broker
        broker.evaluated_memories = evaluated_docs

        logger.info(
            f"Successfully evaluated {len(evaluated_docs)}/{len(documents)} documents"
        )

        return NodeResult(
            status=NodeStatus.SUCCESS,
            data={
                "evaluated_count": len(evaluated_docs),
                "total_count": len(documents),
            },
        )

    def _get_conversation_state(self, broker: KnowledgeBroker) -> ConversationState:
        """Get conversation state from broker.

        Args:
            broker: Knowledge broker

        Returns:
            ConversationState with conversation state information
        """
        state = ConversationState()

        if hasattr(broker, "conversation_history"):
            history = broker.conversation_history
            state.message_count = len(history) if history else 0

            # Extract recent topics from history
            if history and len(history) > 0:
                recent = history[-5:]  # Last 5 messages
                for msg in recent:
                    if hasattr(msg, "key_topics") and msg.key_topics:
                        state.recent_topics.extend(msg.key_topics)

            # Get emotional tone from most recent sentiment
            if history and len(history) > 0:
                last_msg = history[-1]
                if hasattr(last_msg, "emotional_tone"):
                    state.emotional_tone = last_msg.emotional_tone

        return state

    async def _evaluate_memory(
        self,
        document: RAGDocument,
        trust_score: float,
        conversation_state: ConversationState,
        current_message: str | None,
    ) -> MemoryEvaluation | None:
        """Evaluate a single memory using AI.

        Args:
            document: The document to evaluate
            trust_score: Current trust score with user
            conversation_state: Current conversation state
            current_message: Current message being processed

        Returns:
            MemoryEvaluation if successful, None otherwise
        """
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            document, trust_score, conversation_state, current_message
        )

        # Attempt evaluation with retries
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Evaluation attempt {attempt}/{self.max_retries}")
                response = self.llm.generate(prompt)

                evaluation = self._parse_evaluation(response)

                if evaluation:
                    logger.debug(f"Evaluation successful on attempt {attempt}")
                    return evaluation
                else:
                    logger.warning(f"Evaluation parsing failed on attempt {attempt}")

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error on attempt {attempt}: {e}")
            except Exception as e:
                logger.error(
                    f"Unexpected error on attempt {attempt}: {e}", exc_info=True
                )

            # Wait before retry (except on last attempt)
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay)

        logger.error(f"Evaluation failed after {self.max_retries} attempts")
        return None

    def _build_evaluation_prompt(
        self,
        document: RAGDocument,
        trust_score: float,
        conversation_state: ConversationState,
        current_message: str | None,
    ) -> str:
        """Build the evaluation prompt.

        Args:
            document: The document to evaluate
            trust_score: Current trust score
            conversation_state: Current conversation state
            current_message: Current message

        Returns:
            Complete prompt for LLM
        """
        # Extract document metadata
        metadata = document.metadata
        timestamp = metadata.get("timestamp", "unknown")
        sentiment = metadata.get("sentiment", "unknown")
        emotional_tone = metadata.get("emotional_tone", "unknown")
        context_summary = metadata.get("context_summary", "N/A")
        key_topics = metadata.get("key_topics", [])
        original_relevance = metadata.get("relevance", "unknown")
        original_chrono = metadata.get("chrono_relevance", "unknown")

        # Build context section
        context_section = dedent(f"""
            Context Information:
            - Trust Score: {trust_score:.2f} (0.0 = stranger, 1.0 = trusted friend)
            - Conversation Messages: {conversation_state.message_count}
            - Recent Emotional Tone: {conversation_state.emotional_tone or "unknown"}
            - Recent Topics: {', '.join(conversation_state.recent_topics)}
        """)

        # Build document section
        document_section = dedent(f"""
            Memory to Evaluate:
            - Content: {document.content}
            - Timestamp: {timestamp}
            - Sentiment: {sentiment}
            - Emotional Tone: {emotional_tone}
            - Context Summary: {context_summary}
            - Key Topics: {', '.join(key_topics)}
            - Original Relevance: {original_relevance}
            - Original Chrono Relevance: {original_chrono}
        """)

        # Build current message section if available
        message_section = ""
        if current_message:
            message_section = dedent(f"""
                Current Message:
                {current_message}
            """)

        # Combine all sections
        prompt = dedent(f"""
            {self.SYSTEM_PROMPT}
            
            {context_section}
            
            {document_section}
            
            {message_section}
            
            JSON response:
        """)

        return prompt

    def _parse_evaluation(self, response: str) -> MemoryEvaluation | None:
        """Parse LLM response into MemoryEvaluation.

        Args:
            response: Raw response from LLM

        Returns:
            MemoryEvaluation if parsing successful, None otherwise
        """
        try:
            # Extract JSON from response
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            # Validate required fields
            required_fields = [
                "relevance",
                "chrono_relevance",
                "reasoning",
                "should_boost",
                "boost_factor",
            ]
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field: {field}")
                    return None

            # Validate and clamp values
            relevance = float(data["relevance"])
            if not 0.0 <= relevance <= 1.0:
                logger.error(f"Invalid relevance value: {data['relevance']}")
                relevance = max(0.0, min(1.0, relevance))

            chrono_relevance = float(data["chrono_relevance"])
            if not 0.0 <= chrono_relevance <= 1.0:
                logger.error(
                    f"Invalid chrono_relevance value: {data['chrono_relevance']}"
                )
                chrono_relevance = max(0.0, min(1.0, chrono_relevance))

            boost_factor = float(data["boost_factor"])
            if not 0.0 <= boost_factor <= 1.0:
                logger.error(f"Invalid boost_factor value: {data['boost_factor']}")
                boost_factor = max(0.0, min(1.0, boost_factor))

            evaluation = MemoryEvaluation(
                relevance=relevance,
                chrono_relevance=chrono_relevance,
                reasoning=str(data["reasoning"]),
                should_boost=bool(data["should_boost"]),
                boost_factor=boost_factor,
            )

            logger.debug(
                f"Successfully parsed evaluation: "
                f"relevance={evaluation.relevance:.2f}, "
                f"chrono={evaluation.chrono_relevance:.2f}"
            )
            return evaluation

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response}")
            return None
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid evaluation data: {e}")
            return None

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might contain other content.

        Args:
            text: Text that might contain JSON

        Returns:
            Extracted JSON string

        Raises:
            json.JSONDecodeError: If no valid JSON found
        """
        text = text.strip()

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:
            json_str = text[start : end + 1]
            json.loads(json_str)  # Validate
            return json_str

        return text
