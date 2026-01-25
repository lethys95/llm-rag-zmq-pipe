"""Memory consolidation node for merging and re-evaluating memories."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from textwrap import dedent

from src.llm.base import BaseLLM
from src.rag.base import BaseRAG
from src.nodes.core.base import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.rag.selector import RAGDocument

logger = logging.getLogger(__name__)


@dataclass
class ConsolidatedMemory:
    """Result of memory consolidation."""
    
    consolidated_content: str
    relevance: float
    chrono_relevance: float
    reasoning: str
    source_ids: list[str]
    merged_count: int


class MemoryConsolidationNode(BaseNode):
    """Consolidates and re-evaluates memories during idle time.
    
    This node runs during detox sessions to:
    1. Re-evaluate memory importance with fresh perspective
    2. Merge similar memories
    3. Update metadata based on new understanding
    4. Prune truly irrelevant memories
    """
    
    SYSTEM_PROMPT = dedent("""
        You are a memory consolidation assistant for an AI companion. Your task is to
        analyze a group of related memories and consolidate them into a single,
        coherent summary.
        
        Consider the following:
        1. What is the common theme or topic across these memories?
        2. Which details are most important and should be preserved?
        3. What is the emotional significance of these memories?
        4. How long should this consolidated memory remain relevant?
        
        Return a JSON response with the following format:
        {
            "consolidated_content": "A coherent summary of all memories",
            "relevance": 0.0-1.0,
            "chrono_relevance": 0.0-1.0,
            "reasoning": "Explanation of consolidation decisions",
            "should_consolidate": true or false
        }
        
        IMPORTANT: Respond ONLY with valid JSON. No explanations, no additional text.
    """)
    
    def __init__(
        self,
        llm_provider: BaseLLM,
        rag_provider: BaseRAG,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        consolidation_threshold: float = 0.7,
        max_memories_per_batch: int = 10,
        **kwargs
    ):
        """Initialize memory consolidation node.
        
        Args:
            llm_provider: The LLM provider to use for consolidation
            rag_provider: The RAG provider for storing/retrieving memories
            max_retries: Maximum number of retry attempts on failure
            retry_delay: Delay in seconds between retries
            consolidation_threshold: Minimum similarity to consider consolidation
            max_memories_per_batch: Maximum memories to process in one batch
            **kwargs: Additional arguments passed to BaseNode
        """
        super().__init__(
            name="memory_consolidation",
            priority=5,
            queue_type="background",
            **kwargs
        )
        self.llm = llm_provider
        self.rag = rag_provider
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.consolidation_threshold = consolidation_threshold
        self.max_memories_per_batch = max_memories_per_batch
        
        logger.info(
            f"Memory consolidation node initialized "
            f"(threshold={consolidation_threshold}, max_batch={max_memories_per_batch})"
        )
    
    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        """Execute memory consolidation.
        
        Args:
            broker: Knowledge broker
            
        Returns:
            NodeResult with consolidation results
        """
        # Get user ID if available
        user_id = None
        if hasattr(broker, "dialogue_input") and broker.dialogue_input:
            user_id = broker.dialogue_input.speaker
        
        # Get all memories for user (or all if no user)
        all_memories = await self._get_all_memories(user_id)
        
        if not all_memories:
            logger.debug("No memories to consolidate")
            return NodeResult(
                status=NodeStatus.SKIPPED,
                metadata={"reason": "no_memories"}
            )
        
        logger.info(f"Found {len(all_memories)} memories for consolidation")
        
        # Group memories by topic
        topic_groups = self._group_by_topic(all_memories)
        logger.info(f"Grouped memories into {len(topic_groups)} topics")
        
        # Consolidate each topic group
        consolidated_memories = []
        deleted_ids = []
        
        for topic, memories in topic_groups.items():
            if len(memories) > 1:  # Only consolidate if multiple memories
                consolidated = await self._consolidate_topic(memories)
                
                if consolidated:
                    consolidated_memories.append(consolidated)
                    
                    # Mark source memories for deletion
                    deleted_ids.extend(consolidated.source_ids)
                    
                    logger.info(
                        f"Consolidated {len(memories)} memories for topic '{topic}'"
                    )
        
        # Store consolidated memories
        stored_ids = []
        for consolidated in consolidated_memories:
            point_id = await self._store_consolidated_memory(consolidated, user_id)
            if point_id:
                stored_ids.append(point_id)
        
        # Delete old memories
        if deleted_ids:
            await self._delete_memories(deleted_ids)
            logger.info(f"Deleted {len(deleted_ids)} old memories")
        
        # Store results in broker
        broker.consolidation_results = {
            "consolidated_count": len(consolidated_memories),
            "deleted_count": len(deleted_ids),
            "stored_ids": stored_ids
        }
        
        logger.info(
            f"Consolidation complete: {len(consolidated_memories)} consolidated, "
            f"{len(deleted_ids)} deleted"
        )
        
        return NodeResult(
            status=NodeStatus.SUCCESS,
            data={
                "consolidated_count": len(consolidated_memories),
                "deleted_count": len(deleted_ids),
                "stored_count": len(stored_ids)
            }
        )
    
    async def _get_all_memories(self, user_id: str | None) -> list[RAGDocument]:
        """Get all memories for a user.
        
        Args:
            user_id: User ID to filter by, or None for all
            
        Returns:
            List of RAGDocument objects
        """
        try:
            # Use a dummy query to get all documents
            dummy_embedding = [0.0] * 384  # Assuming 384-dimensional embeddings
            
            documents = self.rag.retrieve_documents(
                query_embedding=dummy_embedding,
                limit=1000,
                score_threshold=0.0
            )
            
            # Filter by user if specified
            if user_id:
                documents = [
                    doc for doc in documents
                    if doc.metadata.get("memory_owner") == user_id
                ]
            
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}", exc_info=True)
            return []
    
    def _group_by_topic(self, memories: list[RAGDocument]) -> dict[str, list[RAGDocument]]:
        """Group memories by topic.
        
        Args:
            memories: List of memories to group
            
        Returns:
            Dictionary mapping topics to lists of memories
        """
        topic_groups = {}
        
        for memory in memories:
            # Get key topics from metadata
            key_topics = memory.metadata.get("key_topics", [])
            
            if not key_topics:
                # Use a default topic if none specified
                topic = "general"
            else:
                # Use first topic as primary
                topic = key_topics[0]
            
            if topic not in topic_groups:
                topic_groups[topic] = []
            
            topic_groups[topic].append(memory)
        
        return topic_groups
    
    async def _consolidate_topic(self, memories: list[RAGDocument]) -> ConsolidatedMemory | None:
        """Consolidate memories for a single topic.
        
        Args:
            memories: List of memories to consolidate
            
        Returns:
            ConsolidatedMemory if successful, None otherwise
        """
        if len(memories) < 2:
            return None
        
        # Build consolidation prompt
        prompt = self._build_consolidation_prompt(memories)
        
        # Attempt consolidation with retries
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Consolidation attempt {attempt}/{self.max_retries}")
                response = self.llm.generate(prompt)
                
                consolidation = self._parse_consolidation(response, memories)
                
                if consolidation:
                    logger.debug(f"Consolidation successful on attempt {attempt}")
                    return consolidation
                else:
                    logger.warning(f"Consolidation parsing failed on attempt {attempt}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error on attempt {attempt}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt}: {e}", exc_info=True)
            
            # Wait before retry (except on last attempt)
            if attempt < self.max_retries:
                import asyncio
                await asyncio.sleep(self.retry_delay)
        
        logger.error(f"Consolidation failed after {self.max_retries} attempts")
        return None
    
    def _build_consolidation_prompt(self, memories: list[RAGDocument]) -> str:
        """Build consolidation prompt.
        
        Args:
            memories: List of memories to consolidate
            
        Returns:
            Complete prompt for LLM
        """
        # Build memories section
        memories_section = "Memories to Consolidate:\n"
        for i, memory in enumerate(memories, 1):
            metadata = memory.metadata
            timestamp = metadata.get("timestamp", "unknown")
            sentiment = metadata.get("sentiment", "unknown")
            emotional_tone = metadata.get("emotional_tone", "unknown")
            context_summary = metadata.get("context_summary", "N/A")
            key_topics = metadata.get("key_topics", [])
            relevance = metadata.get("relevance", "unknown")
            chrono_relevance = metadata.get("chrono_relevance", "unknown")
            
            memories_section += dedent(f"""
                Memory {i}:
                - Content: {memory.content}
                - Timestamp: {timestamp}
                - Sentiment: {sentiment}
                - Emotional Tone: {emotional_tone}
                - Context Summary: {context_summary}
                - Key Topics: {', '.join(key_topics)}
                - Original Relevance: {relevance}
                - Original Chrono Relevance: {chrono_relevance}
                
            """)
        
        # Combine all sections
        prompt = dedent(f"""
            {self.SYSTEM_PROMPT}
            
            {memories_section}
            
            JSON response:
        """)
        
        return prompt
    
    def _parse_consolidation(
        self,
        response: str,
        source_memories: list[RAGDocument]
    ) -> ConsolidatedMemory | None:
        """Parse LLM response into ConsolidatedMemory.
        
        Args:
            response: Raw response from LLM
            source_memories: Original memories being consolidated
            
        Returns:
            ConsolidatedMemory if parsing successful, None otherwise
        """
        try:
            # Extract JSON from response
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            
            # Check if consolidation should proceed
            should_consolidate = data.get("should_consolidate", True)
            if not should_consolidate:
                logger.debug("LLM decided not to consolidate these memories")
                return None
            
            # Validate required fields
            required_fields = [
                "consolidated_content",
                "relevance",
                "chrono_relevance",
                "reasoning"
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
                logger.error(f"Invalid chrono_relevance value: {data['chrono_relevance']}")
                chrono_relevance = max(0.0, min(1.0, chrono_relevance))
            
            # Get source IDs
            source_ids = []
            for memory in source_memories:
                point_id = memory.metadata.get("point_id")
                if point_id:
                    source_ids.append(str(point_id))
            
            consolidation = ConsolidatedMemory(
                consolidated_content=str(data["consolidated_content"]),
                relevance=relevance,
                chrono_relevance=chrono_relevance,
                reasoning=str(data["reasoning"]),
                source_ids=source_ids,
                merged_count=len(source_memories)
            )
            
            logger.debug(
                f"Successfully parsed consolidation: "
                f"content_length={len(consolidation.consolidated_content)}, "
                f"merged_count={consolidation.merged_count}"
            )
            return consolidation
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response}")
            return None
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid consolidation data: {e}")
            return None
    
    async def _store_consolidated_memory(
        self,
        consolidated: ConsolidatedMemory,
        user_id: str | None
    ) -> str | None:
        """Store a consolidated memory in RAG.
        
        Args:
            consolidated: The consolidated memory to store
            user_id: User ID for memory
            
        Returns:
            Point ID if successful, None otherwise
        """
        try:
            from src.rag.embeddings import EmbeddingService
            
            embedding_service = EmbeddingService.get_instance()
            embedding = embedding_service.encode(consolidated.consolidated_content)
            
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "memory_owner": user_id or "system",
                "sentiment": "neutral",
                "confidence": 1.0,
                "emotional_tone": "neutral",
                "relevance": consolidated.relevance,
                "chrono_relevance": consolidated.chrono_relevance,
                "context_summary": f"Consolidated from {consolidated.merged_count} memories",
                "key_topics": [],
                "consolidated_with": consolidated.source_ids,
                "is_consolidated": True,
                "consolidation_reasoning": consolidated.reasoning
            }
            
            point_id = self.rag.store(
                text=consolidated.consolidated_content,
                embedding=embedding,
                metadata=metadata
            )
            
            logger.debug(f"Stored consolidated memory with ID: {point_id}")
            return point_id
            
        except Exception as e:
            logger.error(f"Error storing consolidated memory: {e}", exc_info=True)
            return None
    
    async def _delete_memories(self, point_ids: list[str]) -> None:
        """Delete memories from RAG.
        
        Args:
            point_ids: List of point IDs to delete
        """
        try:
            self.rag.delete(point_ids)
            logger.debug(f"Deleted {len(point_ids)} memories")
        except Exception as e:
            logger.error(f"Error deleting memories: {e}", exc_info=True)
    
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
        
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1:
            json_str = text[start:end + 1]
            json.loads(json_str)  # Validate
            return json_str
        
        return text
