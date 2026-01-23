"""Pipeline server orchestrating RAG and LLM processing."""

import logging
import signal
from datetime import datetime

from ..models.sentiment import DialogueInput, SentimentAnalysis
from ..config.settings import Settings
from ..llm.factory import create_primary_llm, create_sentiment_llm, create_interpreter_llm
from ..llm.base import BaseLLM
from ..rag.factory import create_rag_provider
from ..rag.base import BaseRAG
from ..rag.embeddings import EmbeddingService
from ..rag.algorithms import MemoryDecayAlgorithm
from ..storage import ConversationStore
from ..handlers.sentiment_analysis import SentimentAnalysisHandler
from ..handlers.context_interpreter import ContextInterpreterHandler
from ..handlers.primary_response import PrimaryResponseHandler
from .zmq_handler import ZMQHandler

logger = logging.getLogger(__name__)


class PipelineServer:
    """Main pipeline server orchestrating the RAG-LLM processing.
    
    This server:
    1. Receives prompts via ZMQ ROUTER socket
    2. Performs sentiment analysis (optional)
    3. Retrieves relevant context using RAG
    4. Generates primary response using LLM
    5. Sends acknowledgment back to requester
    6. Forwards response to downstream pipeline via PUSH socket
    """
    
    def __init__(self, settings: Settings):
        """Initialize the pipeline server.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.running = False
        
        self.zmq_handler: ZMQHandler | None = None
        self.primary_llm: BaseLLM | None = None
        self.sentiment_llm: BaseLLM | None = None
        self.interpreter_llm: BaseLLM | None = None
        self.rag_provider: BaseRAG | None = None
        self.conversation_store: ConversationStore | None = None
        self.embedding_service: EmbeddingService | None = None
        self.memory_decay: MemoryDecayAlgorithm | None = None
        
        self.sentiment_handler: SentimentAnalysisHandler | None = None
        self.interpreter_handler: ContextInterpreterHandler | None = None
        self.primary_handler: PrimaryResponseHandler | None = None
        
        logger.info("Pipeline server initialized")
    
    def setup(self) -> None:
        """Set up all components of the pipeline."""
        logger.info("Setting up pipeline components...")
        
        # Set up ZMQ handler
        self.zmq_handler = ZMQHandler(
            input_endpoint=self.settings.input_endpoint,
            output_endpoint=self.settings.output_endpoint,
        )
        self.zmq_handler.setup()
        
        # Create RAG provider (shared by all handlers)
        self.rag_provider = create_rag_provider(self.settings)
        
        logger.info("Initializing conversation store...")
        self.conversation_store = ConversationStore(
            db_path=self.settings.conversation_store.db_path,
            max_messages=self.settings.conversation_store.max_messages,
            context_limit=self.settings.conversation_store.context_limit
        )
        
        logger.info("Initializing embedding service...")
        self.embedding_service = EmbeddingService.get_instance()
        
        logger.info("Initializing memory decay algorithm...")
        self.memory_decay = MemoryDecayAlgorithm(
            memory_half_life_days=self.settings.memory_decay.half_life_days,
            chrono_weight=self.settings.memory_decay.chrono_weight,
            retrieval_threshold=self.settings.memory_decay.retrieval_threshold,
            prune_threshold=self.settings.memory_decay.prune_threshold,
            max_documents=self.settings.memory_decay.max_documents
        )
        
        logger.info("Creating primary LLM provider...")
        self.primary_llm = create_primary_llm(self.settings)
        
        if self.settings.enable_sentiment_analysis:
            logger.info("Creating sentiment LLM provider...")
            self.sentiment_llm = create_sentiment_llm(self.settings)
        
        if self.settings.enable_context_interpreter:
            logger.info("Creating context interpreter LLM provider...")
            self.interpreter_llm = create_interpreter_llm(self.settings)
        
        # Create handlers using composition
        # Note: Create interpreter first if enabled, then pass to primary handler
        if self.settings.enable_context_interpreter:
            self.interpreter_handler = ContextInterpreterHandler(
                llm_provider=self.interpreter_llm,
            )
        
        if self.settings.enable_sentiment_analysis:
            self.sentiment_handler = SentimentAnalysisHandler(
                llm_provider=self.sentiment_llm,
                rag_provider=self.rag_provider,
                max_retries=self.settings.sentiment_max_retries,
                retry_delay=self.settings.sentiment_retry_delay,
            )
        
        self.primary_handler = PrimaryResponseHandler(
            llm_provider=self.primary_llm,
            rag_provider=self.rag_provider,
            interpreter_handler=self.interpreter_handler if self.settings.enable_context_interpreter else None,
            conversation_store=self.conversation_store,
            memory_decay=self.memory_decay,
            max_semantic_documents=self.settings.memory_decay.max_documents,
        )
        
        logger.info("Pipeline setup complete")
    
    def run(self) -> None:
        """Run the main server loop.
        
        This method blocks until interrupted by signal or error.
        """
        self.setup()
        self.running = True
        
        self._setup_shutdown_signals()
        
        logger.info("Pipeline server started. Waiting for requests...")
        
        try:
            while self.running:
                self._process_request()
        
        except Exception as e:
            logger.error(f"Fatal error in server loop: {e}", exc_info=True)
            raise
        
        finally:
            self.shutdown()

    def _setup_shutdown_signals(self) -> None:
        """Configure signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _process_request(self) -> None:
        """Process a single request from the pipeline.
        
        Internal method that handles the core processing logic.
        """
        identity, dialogue_input = self.zmq_handler.receive_request(timeout=1000)
        
        if identity is None:
            return
        
        if dialogue_input is None:
            logger.error("Received invalid dialogue input, skipping")
            self.zmq_handler.send_acknowledgment(
                identity=identity,
                status="error",
                message="Invalid message format: expected JSON with 'content' and 'speaker' fields"
            )
            return
        
        logger.info(
            f"Processing request from '{dialogue_input.speaker}': "
            f"{dialogue_input.content[:100]}..."
        )
        
        try:
            sentiment = self._perform_sentiment_analysis(dialogue_input=dialogue_input)
            
            logger.info("Generating primary response...")
            response = self.primary_handler.generate_response(
                prompt=dialogue_input.content,
                use_rag=self.settings.rag_enabled,
                system_prompt_override=dialogue_input.system_prompt_override
            )
            
            self._store_conversation(
                speaker=dialogue_input.speaker,
                message=dialogue_input.content,
                response=response,
                sentiment=sentiment
            )
            
            self.zmq_handler.send_acknowledgment(
                identity=identity,
                status="success",
                message=self._build_ack(sentiment=sentiment)
            )
            
            self.zmq_handler.forward_response(response)
            
            feedback = self.zmq_handler.check_downstream_feedback(timeout=0)
            if feedback:
                logger.warning(f"Downstream feedback: {feedback}")
            
            logger.info("Request processed successfully")
        
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            self.zmq_handler.send_acknowledgment(
                identity=identity,
                status="error",
                message=f"Processing failed: {str(e)}"
            )

    def _perform_sentiment_analysis(self, dialogue_input: DialogueInput) -> SentimentAnalysis | None:
        """Perform sentiment analysis on the dialogue input.
        
        Args:
            dialogue_input: The incoming dialogue to analyze
            
        Returns:
            SentimentAnalysis result if analysis is enabled, None otherwise
        """
        sentiment = None
        if self.sentiment_handler:
            logger.info("Performing sentiment analysis...")
            sentiment = self.sentiment_handler.analyze(
                message=dialogue_input.content,
                speaker=dialogue_input.speaker
            )
            if sentiment:
                logger.info(
                    f"Sentiment for '{sentiment.memory_owner}': "
                    f"{sentiment.sentiment} (confidence: {sentiment.confidence})"
                )
        return sentiment

    def _build_ack(self, sentiment: SentimentAnalysis | None) -> str:
        """Build acknowledgment message with optional sentiment info.
        
        Args:
            sentiment: Optional sentiment analysis result to include
            
        Returns:
            Formatted acknowledgment message string
        """
        ack_msg = "Request processed successfully"
        if sentiment:
            ack_msg += f" | Sentiment: {sentiment.sentiment}"
        return ack_msg
    
    def _store_conversation(
        self,
        speaker: str,
        message: str,
        response: str,
        sentiment: SentimentAnalysis | None = None
    ) -> None:
        """Store conversation in both SQLite and Qdrant.
        
        Args:
            speaker: Name of the speaker
            message: User's message
            response: Assistant's response
            sentiment: Optional sentiment analysis result
        """
        try:
            timestamp = datetime.now().isoformat()
            
            logger.debug("Storing conversation in SQLite...")
            self.conversation_store.add_message(
                speaker=speaker,
                message=message,
                response=response,
                timestamp=timestamp
            )
            
            logger.debug("Storing conversation in Qdrant...")
            conversation_text = f"{speaker}: {message}\nAssistant: {response}"
            embedding = self.embedding_service.encode(conversation_text)
            metadata = self._prepare_metadata(timestamp, speaker, sentiment)
            
            self.rag_provider.store(
                text=conversation_text,
                embedding=embedding,
                metadata=metadata
            )
            
            logger.info("Conversation stored successfully in both memory tiers")
            
        except Exception as e:
            logger.error(f"Error storing conversation: {e}", exc_info=True)
    
    def _prepare_metadata(
        self,
        timestamp: str,
        speaker: str,
        sentiment: SentimentAnalysis | None = None
    ) -> dict:
        """Prepare metadata for Qdrant storage.
        
        Args:
            timestamp: ISO format timestamp
            speaker: Name of the speaker
            sentiment: Optional sentiment analysis result
            
        Returns:
            Dictionary of metadata for vector storage
        """
        metadata = {
            "timestamp": timestamp,
            "speaker": speaker,
        }
        
        if sentiment:
            metadata["relevance"] = sentiment.relevance if sentiment.relevance is not None else 0.5
            metadata["chrono_relevance"] = sentiment.chrono_relevance if sentiment.chrono_relevance is not None else 0.5
            metadata["sentiment"] = sentiment.sentiment
            if sentiment.key_topics:
                metadata["topics"] = sentiment.key_topics
            if sentiment.context_summary:
                metadata["context_summary"] = sentiment.context_summary
        else:
            metadata["relevance"] = 0.5
            metadata["chrono_relevance"] = 0.5
        
        return metadata
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def shutdown(self) -> None:
        """Shut down the server and clean up resources."""
        logger.info("Shutting down pipeline server...")
        
        components = [
            self.sentiment_llm,
            self.interpreter_llm,
            self.primary_llm,
            self.rag_provider,
            self.conversation_store,
            self.zmq_handler,
        ]
        
        for component in components:
            if component and hasattr(component, 'close'):
                try:
                    component.close()
                except Exception as e:
                    logger.error(f"Error closing {component.__class__.__name__}: {e}", exc_info=True)
        
        logger.info("Pipeline server shut down complete")
