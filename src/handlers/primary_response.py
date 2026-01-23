"""Primary response handler using composition."""

import logging
from textwrap import dedent

from ..llm.base import BaseLLM
from ..rag.base import BaseRAG
from ..rag.selector import RAGDocument
from ..rag.algorithms import MemoryDecayAlgorithm
from ..storage import ConversationStore
from .context_interpreter import ContextInterpreterHandler

logger = logging.getLogger(__name__)


class PrimaryResponseHandler:
    """Handler for generating primary responses using a large LLM.
    
    This handler composes a BaseLLM provider and RAG system to generate
    the main response to user queries. It uses composition over inheritance
    to remain flexible and focused on its single responsibility.
    
    Optionally composes a ContextInterpreterHandler to reformulate RAG
    results before feeding them to the primary LLM.
    """
    
    def __init__(
        self,
        llm_provider: BaseLLM,
        rag_provider: BaseRAG,
        interpreter_handler: ContextInterpreterHandler | None = None,
        conversation_store: ConversationStore | None = None,
        memory_decay: MemoryDecayAlgorithm | None = None,
        max_semantic_documents: int = 10
    ):
        """Initialize the primary response handler.
        
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
            logger.info("Primary response handler initialized without context interpreter")
        
        if self.conversation_store and self.memory_decay:
            logger.info("Primary response handler initialized with two-tier memory system")
    
    def generate_response(
        self,
        prompt: str,
        context: str | None = None,
        use_rag: bool = True,
        system_prompt_override: str | None = None
    ) -> str:
        """Generate a response to the user prompt.
        
        Args:
            prompt: The user's prompt/question
            context: Optional pre-retrieved context (if None and use_rag=True, will retrieve)
            use_rag: Whether to use RAG for context retrieval
            system_prompt_override: Optional override for the system prompt persona
            
        Returns:
            Generated response string
        """
        logger.debug(f"Generating primary response for prompt: {prompt[:100]}...")
        
        try:
            if use_rag and context is None:
                context = self._retrieve_context(prompt)
            
            full_prompt = self._build_prompt(prompt, context, system_prompt_override)
            response = self.llm.generate(full_prompt)

            print(f"PRIMARY RESPONSE: {response}")
            
            logger.info(f"Primary response generated (length: {len(response)})")
            return response
            
        except Exception as e:
            logger.error(f"Error generating primary response: {e}", exc_info=True)
            raise
    
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
            logger.debug(f"Combined context length: {len(combined_context)}")
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
            logger.info(f"Retrieved {len(recent_messages)} recent messages from SQLite")
            return f"Recent Conversation:\n{recent_context}"
            
        except Exception as e:
            logger.error(f"Error retrieving recent conversations: {e}", exc_info=True)
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
            logger.info(f"Retrieved {len(filtered_documents)} semantic memories from Qdrant")
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
            logger.info(f"Memory decay: {len(documents)} → {len(filtered)} documents")
            return filtered
        
        return documents[:self.max_semantic_documents]
    
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
                query=query,
                documents=documents,
                include_metadata=False
            )
        
        semantic_parts = [f"[Memory {i+1}]: {doc.content}" 
                         for i, doc in enumerate(documents[:self.max_semantic_documents])]
        return "\n\n".join(semantic_parts)
    
    def _build_prompt(
        self,
        prompt: str,
        context: str | None,
        system_prompt_override: str | None = None
    ) -> str:
        """Build the full prompt with optional context and system prompt override.
        
        Args:
            prompt: Original user prompt
            context: Retrieved context from RAG (if any)
            system_prompt_override: Optional override for the system prompt persona
            
        Returns:
            Augmented prompt ready for LLM
        """
        default_with_context = "You are a helpful AI assistant. Use the following context to answer the user's question."
        default_without_context = "You are a helpful AI assistant. Answer the user's question directly and concisely."
        
        if context and context.strip():
            system_prompt = system_prompt_override or default_with_context
            logger.debug(f"Built augmented prompt with context{' and custom system prompt' if system_prompt_override else ''}")
            return dedent(f"""
                {system_prompt}

                Context:
                {context}

                User Question:
                {prompt}

                Assistant Response:""")
        
        system_prompt = system_prompt_override or default_without_context
        logger.debug(f"Built prompt without context{' and custom system prompt' if system_prompt_override else ''}")
        return dedent(f"""
            {system_prompt}

            User Question:
            {prompt}

            Assistant Response:""")
