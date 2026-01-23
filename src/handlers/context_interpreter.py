"""Context interpretation handler for reformulating RAG results."""

import logging
from textwrap import dedent

from ..llm.base import BaseLLM
from ..rag.selector import RAGDocument

logger = logging.getLogger(__name__)


class ContextInterpreterHandler:
    """Handler for interpreting and reformulating RAG context.
    
    This handler takes raw RAG results and uses an LLM to reformulate them
    into a clear, coherent context that the primary response LLM can use
    effectively. It bridges the gap between retrieved documents and
    actionable context.
    """
    
    SYSTEM_PROMPT = dedent(
        """
        You are a context interpretation assistant. Your job is to take retrieved documents and reformulate them into clear, coherent context that will help another AI answer a user's question.

        Your responsibilities:
        1. Synthesize information from multiple documents into a unified context
        2. Remove redundancy and contradictions
        3. Highlight the most relevant information for the given query
        4. Organize information logically
        5. Maintain factual accuracy - do not add information not present in the documents

        Important guidelines:
        - Keep the reformulated context concise but comprehensive
        - Preserve important details, dates, names, and specific facts
        - Use clear, straightforward language
        - If documents contain conflicting information, note the conflict
        - If documents are irrelevant to the query, state that clearly

        Do NOT:
        - Add your own knowledge or assumptions
        - Answer the user's question directly (that's the primary LLM's job)
        - Include metadata or technical details about the retrieval process
        - Be verbose - focus on what's truly relevant""")

    def __init__(self, llm_provider: BaseLLM):
        """Initialize the context interpreter handler.
        
        Args:
            llm_provider: The LLM provider to use for interpretation (composed, not inherited)
        """
        self.llm = llm_provider
        
        logger.info("Context interpreter handler initialized")
    
    def interpret(
        self,
        query: str,
        documents: list[RAGDocument],
        include_metadata: bool = False
    ) -> str:
        """Interpret and reformulate RAG documents into coherent context.
        
        Args:
            query: The original user query
            documents: List of RAGDocument objects from RAG retrieval
            include_metadata: Whether to include document metadata in interpretation
            
        Returns:
            Reformulated context string ready for primary LLM consumption
        """
        logger.debug(f"Interpreting {len(documents)} documents for query: {query[:100]}...")
        
        if not documents:
            logger.info("No documents to interpret, returning empty context")
            return ""
        
        try:
            # Build the interpretation prompt
            full_prompt = self._build_prompt(query, documents, include_metadata)
            
            # Get reformulated context from LLM
            reformulated = self.llm.generate(full_prompt)
            
            logger.info(
                f"Context interpretation successful (input: {len(documents)} docs, "
                f"output: {len(reformulated)} chars)"
            )
            
            return reformulated.strip()
            
        except Exception as e:
            logger.error(f"Error during context interpretation: {e}", exc_info=True)
            # Fallback: return raw documents formatted simply
            logger.warning("Falling back to simple document concatenation")
            return self._simple_concatenation(documents)
    
    def _build_prompt(
        self,
        query: str,
        documents: list[RAGDocument],
        include_metadata: bool
    ) -> str:
        """Build the full prompt for context interpretation.
        
        Args:
            query: The user's query
            documents: Retrieved documents
            include_metadata: Whether to include metadata
            
        Returns:
            Full prompt with system instructions, documents, and query
        """
        # Format documents
        docs_text = self._format_documents(documents, include_metadata)
        
        return dedent(
            f"""
            {self.SYSTEM_PROMPT}

            User's Query:
            {query}

            Retrieved Documents:
            {docs_text}

            Task: Reformulate the above documents into clear, coherent context that will help answer the user's query. Focus on what's relevant to the query.

            Reformulated Context:""")
    
    def _format_documents(
        self,
        documents: list[RAGDocument],
        include_metadata: bool
    ) -> str:
        """Format documents for inclusion in the interpretation prompt.
        
        Args:
            documents: Documents to format
            include_metadata: Whether to include metadata
            
        Returns:
            Formatted documents string
        """
        formatted_parts = []
        
        for i, doc in enumerate(documents, 1):
            if include_metadata:
                # Include score and relevant metadata
                meta_items = []
                meta_items.append(f"Score: {doc.score:.3f}")
                
                # Add timestamp if available
                if "timestamp" in doc.metadata:
                    meta_items.append(f"Time: {doc.metadata['timestamp']}")
                
                # Add source if available
                if "source" in doc.metadata:
                    meta_items.append(f"Source: {doc.metadata['source']}")
                
                meta_str = " | ".join(meta_items)
                formatted_parts.append(
                    f"Document {i} [{meta_str}]:\n{doc.content}\n"
                )
            else:
                # Simple format
                formatted_parts.append(f"Document {i}:\n{doc.content}\n")
        
        return "\n".join(formatted_parts)
    
    def _simple_concatenation(self, documents: list[RAGDocument]) -> str:
        """Simple fallback: concatenate document contents.
        
        Used when interpretation fails.
        
        Args:
            documents: Documents to concatenate
            
        Returns:
            Concatenated document text
        """
        parts = [doc.content for doc in documents]
        return "\n\n".join(parts)
    
    def interpret_with_sentiment(
        self,
        query: str,
        documents: list[RAGDocument],
        sentiment_summary: str | None = None
    ) -> str:
        """Interpret context with additional sentiment information.
        
        This variant allows including sentiment analysis results to help
        the interpreter understand the emotional context and urgency.
        
        Args:
            query: The user's query
            documents: Retrieved documents
            sentiment_summary: Optional summary of sentiment analysis
            
        Returns:
            Reformulated context string
        """
        logger.debug("Interpreting with sentiment context")
        
        if not documents:
            return ""
        
        try:
            # Build enhanced prompt with sentiment
            full_prompt = self._build_sentiment_prompt(
                query, documents, sentiment_summary
            )
            
            # Get reformulated context from LLM
            reformulated = self.llm.generate(full_prompt)
            
            logger.info("Context interpretation with sentiment successful")
            
            return reformulated.strip()
            
        except Exception as e:
            logger.error(f"Error during sentiment-aware interpretation: {e}", exc_info=True)
            # Fallback to regular interpretation
            return self.interpret(query, documents)
    
    def _build_sentiment_prompt(
        self,
        query: str,
        documents: list[RAGDocument],
        sentiment_summary: str | None
    ) -> str:
        """Build prompt with sentiment information included.
        
        Args:
            query: The user's query
            documents: Retrieved documents
            sentiment_summary: Sentiment analysis summary
            
        Returns:
            Full prompt with sentiment context
        """
        docs_text = self._format_documents(documents, include_metadata=False)
        
        sentiment_section = ""
        if sentiment_summary:
            sentiment_section = f"\nUser Sentiment Analysis:\n{sentiment_summary}\n"
        
        return dedent(
            f"""
            {self.SYSTEM_PROMPT}

            User's Query:
            {query}
            {sentiment_section}
            Retrieved Documents:
            {docs_text}

            Task: Reformulate the above documents into clear, coherent context. Consider the user's emotional state and urgency when determining what information to emphasize.

            Reformulated Context:""")
