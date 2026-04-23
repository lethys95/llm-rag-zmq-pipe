"""Primary response handler using composition."""

import logging
from textwrap import dedent

from src.llm.base import BaseLLM
from src.rag.base import BaseRAG
from src.rag.selector import RAGDocument
from src.rag.algorithms import MemoryDecayAlgorithm
from src.storage import ConversationStore
from src.handlers.context_interpreter import ContextInterpreterHandler
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker

logger = logging.getLogger(__name__)


class PrimaryResponseHandler:
    """Handler for generating primary responses using a large LLM.

    This handler composes a BaseLLM provider and RAG system to generate
    main response to user queries. It uses composition over inheritance
    to remain flexible and focused on its single responsibility.

    Optionally composes a ContextInterpreterHandler to reformulate RAG
    results before feeding them to the primary LLM.
    """

    SYSTEM_PROMPT_WITH_CONTEXT = "You are an AI companion. You're here to provide emotional support, to listen, provide guidance, etc."

    SYSTEM_PROMPT_WITHOUT_CONTEXT = "You are an AI companion. You're here to provide emotional support, to listen, provide guidance, etc."

    def __init__(
        self,
        llm_provider: BaseLLM,
        rag_provider: BaseRAG,
        interpreter_handler: ContextInterpreterHandler | None = None,
        conversation_store: ConversationStore | None = None,
        memory_decay: MemoryDecayAlgorithm | None = None,
        max_semantic_documents: int = 10,
    ):
        """Initialize primary response handler.

        Args:
            llm_provider: The LLM provider to use for generation (composed, not inherited)
            rag_provider: The RAG provider for retrieving relevant context
            interpreter_handler: Optional handler for interpreting/reformulating RAG context
            conversation_store: Optional store for recent conversation history
            memory_decay: Optional algorithm for time-based memory filtering
            max_semantic_documents: Maximum number of semantic documents to use (default: 10)
        """
        self.llm = llm_provider
        self.rag = rag_provider
        self.interpreter = interpreter_handler
        self.conversation_store = conversation_store
        self.memory_decay = memory_decay
        self.max_semantic_documents = max_semantic_documents

        if self.interpreter:
            logger.info("Primary response handler initialized with context interpreter")
        else:
            logger.info(
                "Primary response handler initialized without context interpreter"
            )

        if self.conversation_store and self.memory_decay:
            logger.info(
                "Primary response handler initialized with two-tier memory system"
            )

    def generate_response(
        self,
        prompt: str,
        broker: KnowledgeBroker,
        use_rag: bool = True,
        system_prompt_override: str | None = None,
    ) -> str:
        """Generate a response to user prompt.

        Args:
            prompt: The user's prompt/question
            broker: KnowledgeBroker providing access to all context and analyzed data
            use_rag: Whether to use RAG for context retrieval
            system_prompt_override: Optional override for system prompt persona

        Returns:
            Generated response string
        """
        logger.debug("Generating primary response for prompt: %s...", prompt[:100])

        try:
            # Get analyzed context from broker (preferred over raw RAG)
            analyzed_context = broker.get_analyzed_context()

            # Format analyzed context if available
            if analyzed_context:
                context = self._format_analyzed_context(analyzed_context)
            elif use_rag:
                context = self._retrieve_context(prompt)
            else:
                context = None

            full_prompt = self._build_prompt(
                prompt, context, system_prompt_override, analyzed_context
            )
            response = self.llm.generate(full_prompt)

            logger.info("Primary response generated (length: %s)", len(response))
            return response

        except Exception as e:
            logger.error("Error generating primary response: %s", e, exc_info=True)
            raise

    def _format_analyzed_context(self, analyzed_context: dict) -> str:
        """Format analyzed context from various nodes into coherent string.

        Args:
            analyzed_context: Dictionary with sentiment, memories, trust, etc.

        Returns:
            Formatted context string for LLM
        """
        parts = []

        # Add emotional state
        if "emotional_state" in analyzed_context:
            state = analyzed_context["emotional_state"]
            emotion_parts = []
            dominant = sorted(
                [(k, v) for k, v in state.model_dump().items()
                 if k not in ("valence", "arousal", "dominance", "confidence", "summary")
                 and isinstance(v, float) and v > 0.1],
                key=lambda x: x[1], reverse=True,
            )[:3]
            if dominant:
                emotion_parts.append("Emotions: " + ", ".join(f"{k}: {v:.2f}" for k, v in dominant))
            if state.summary:
                emotion_parts.append(f"State: {state.summary}")
            if emotion_parts:
                parts.append("\n".join(emotion_parts))

        # Add needs analysis
        if "needs_analysis" in analyzed_context:
            needs = analyzed_context["needs_analysis"]
            need_parts = []
            if needs.primary_needs:
                need_parts.append(f"Primary needs: {', '.join(needs.primary_needs)}")
            if needs.unmet_needs:
                need_parts.append(f"Unmet: {', '.join(needs.unmet_needs)}")
            if needs.need_urgency > 0.6:
                need_parts.append(f"Urgency: {needs.need_urgency:.2f}")
            if needs.context_summary:
                need_parts.append(needs.context_summary)
            if need_parts:
                parts.append("\n".join(need_parts))

        # Add retrieved memories
        if "user_facts" in analyzed_context:
            facts = analyzed_context["user_facts"]
            if facts:
                fact_lines = [f"- {f.claim}" for f in facts[:8]]
                parts.append("What I know about you:\n" + "\n".join(fact_lines))

        # Add idle time
        if "idle_time_minutes" in analyzed_context:
            idle = analyzed_context["idle_time_minutes"]
            if idle > 0:
                parts.append(f"User idle time: {idle:.1f} minutes")

        if parts:
            return "\n\n---\n\n".join(parts)

        return ""

    def _retrieve_context(self, prompt: str) -> str:
        """Retrieve relevant context from two-tier memory system.

        Combines recent conversation history and semantic memories.

        Args:
            prompt: The user's prompt

        Returns:
            Combined context string
        """
        logger.debug("Retrieving context from two-tier memory system")

        context_parts = []

        recent_context = self._retrieve_recent_conversations()
        if recent_context:
            context_parts.append(recent_context)

        semantic_context = self._retrieve_semantic_memories(prompt)
        if semantic_context:
            context_parts.append(semantic_context)

        if context_parts:
            combined_context = "\n\n---\n\n".join(context_parts)
            logger.debug("Combined context length: %s", len(combined_context))
            return combined_context

        logger.debug("No context retrieved")
        return ""

    def _retrieve_recent_conversations(self) -> str | None:
        """Retrieve recent conversation history from SQLite.

        Returns:
            Formatted recent conversation context, or None if unavailable
        """
        if not self.conversation_store:
            return None

        try:
            logger.debug("Retrieving recent conversation history...")
            recent_messages = self.conversation_store.get_recent_for_context()

            if not recent_messages:
                return None

            recent_context = self.conversation_store.format_for_llm(recent_messages)
            logger.info("Retrieved %s recent messages from SQLite", len(recent_messages))
            return f"Recent Conversation:\n{recent_context}"

        except Exception as e:
            logger.error("Error retrieving recent conversations: %s", e, exc_info=True)
            return None

    def _retrieve_semantic_memories(self, prompt: str) -> str | None:
        """Retrieve semantic memories from Qdrant with memory decay filtering.

        Args:
            prompt: The user's prompt for semantic search

        Returns:
            Formatted semantic context, or None if unavailable
        """
        try:
            logger.debug("Retrieving semantic memories from Qdrant...")
            raw_documents = self.rag.retrieve_documents(prompt, top_k=50)

            if not raw_documents:
                logger.debug("No semantic documents retrieved from Qdrant")
                return None

            filtered_documents = self._apply_memory_decay(raw_documents)

            if not filtered_documents:
                return None

            semantic_context = self._format_semantic_context(prompt, filtered_documents)
            logger.info(
                "Retrieved %s semantic memories from Qdrant",
                len(filtered_documents),
            )
            return f"Relevant Memories:\n{semantic_context}"

        except Exception as e:
            logger.error(f"Error retrieving semantic memories: {e}", exc_info=True)
            return None

    def _apply_memory_decay(self, documents: list[RAGDocument]) -> list[RAGDocument]:
        """Apply memory decay filtering to documents.

        Args:
            documents: Raw documents from Qdrant

        Returns:
            Filtered and ranked documents
        """
        if self.memory_decay:
            logger.debug("Applying memory decay algorithm...")
            filtered = self.memory_decay.filter_and_rank(documents)
            logger.info("Memory decay: %s → %s documents", len(documents), len(filtered))
            return filtered

        return documents[: self.max_semantic_documents]

    def _format_semantic_context(self, query: str, documents: list[RAGDocument]) -> str:
        """Format semantic documents into context string.

        Args:
            query: The user's query
            documents: Filtered semantic documents

        Returns:
            Formatted context string
        """
        if self.interpreter:
            logger.debug("Using context interpreter for semantic memories")
            return self.interpreter.interpret(
                query=query, documents=documents, include_metadata=False
            )

        semantic_parts = [
            f"[Memory {i+1}]: {doc.content}"
            for i, doc in enumerate(documents[: self.max_semantic_documents])
        ]
        return "\n\n".join(semantic_parts)

    def _format_advisor_outputs(self, outputs: list) -> str:
        if not outputs:
            return ""
        relevant = [o for o in outputs if o.potency >= 0.3]
        if not relevant:
            return ""
        relevant.sort(key=lambda o: o.potency, reverse=True)
        lines = ["\nAdvisory context:"]
        for o in relevant:
            lines.append(f"[{o.advisor}] {o.advice}")
        return "\n".join(lines) + "\n"

    def _build_prompt(
        self,
        prompt: str,
        context: str | None,
        system_prompt_override: str | None = None,
        analyzed_context: dict | None = None,
    ) -> str:
        """Build full prompt with optional context and system prompt override.

        Args:
            prompt: Original user prompt
            context: Retrieved context from RAG (if any)
            system_prompt_override: Optional override for system prompt persona
            analyzed_context: Optional dictionary with analyzed data from nodes

        Returns:
            Augmented prompt ready for LLM
        """
        default_with_context = self.SYSTEM_PROMPT_WITH_CONTEXT
        default_without_context = self.SYSTEM_PROMPT_WITHOUT_CONTEXT

        strategy_addition = ""
        if analyzed_context and "response_strategy" in analyzed_context:
            strategy_addition = "\n\n" + analyzed_context["response_strategy"].system_prompt_addition

        advisor_guidance = ""
        if analyzed_context and "advisor_outputs" in analyzed_context:
            advisor_guidance = self._format_advisor_outputs(analyzed_context["advisor_outputs"])

        if context and context.strip():
            system_prompt = (system_prompt_override or default_with_context) + strategy_addition
            logger.debug(
                "Built augmented prompt with context%s",
                " and custom system prompt" if system_prompt_override else "",
            )
            return dedent(f"""
                {system_prompt}

                Context:
                {context}
                {advisor_guidance}
                User Question:
                {prompt}

                Assistant Response:""")

        system_prompt = (system_prompt_override or default_without_context) + strategy_addition
        if advisor_guidance:
            return dedent(f"""
                {system_prompt}
                {advisor_guidance}
                User Question:
                {prompt}

                Assistant Response:""")

        logger.debug(
            "Built prompt without context%s",
            " and custom system prompt" if system_prompt_override else "",
        )
        return dedent(f"""
            {system_prompt}

            User Question:
            {prompt}

            Assistant Response:""")
