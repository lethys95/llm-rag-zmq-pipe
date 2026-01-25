"""Node-based orchestrator for LLM RAG response system."""

import logging
import signal
import asyncio

from .config.settings import Settings
from .communication.zmq_connection_manager import ZMQConnectionManager
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.queue_manager import TaskQueueManager
from src.nodes.orchestration.decision_engine import DecisionEngine
from src.nodes import NodeRegistry
from src.nodes.processing.sentiment_analysis_node import SentimentAnalysisNode
from src.nodes.processing.primary_response_node import PrimaryResponseNode
from src.nodes.processing.ack_preparation_node import AckPreparationNode
from src.nodes.storage_nodes.store_conversation_node import StoreConversationNode
from .llm.factory import create_primary_llm, create_sentiment_llm, create_interpreter_llm
from .rag.factory import create_rag_provider
from .rag.embeddings import EmbeddingService
from .rag.algorithms import MemoryDecayAlgorithm
from .storage import ConversationStore
from .handlers.sentiment_analysis import SentimentAnalysisHandler
from .handlers.context_interpreter import ContextInterpreterHandler
from .handlers.primary_response import PrimaryResponseHandler
from src.nodes.algo_nodes import MemoryEvaluatorNode, TrustAnalysisNode, NeedsAnalysisNode
from src.nodes.algo_nodes.detox_scheduler import DetoxScheduler, DetoxSessionNode
from src.chrono.task_scheduler import TaskScheduler

logger = logging.getLogger(__name__)


class Orchestrator:
    """Node-based orchestrator for LLM RAG response system.
    
    This orchestrator replaces the old PipelineServer with a flexible node-based
    architecture that dynamically selects and executes processing nodes based on
    incoming message and context.
    
    Architecture:
        1. Receive message via ZMQ
        2. Use DecisionEngine to select nodes
        3. Execute nodes via TaskQueueManager
        4. Send ACK and forward response
        5. Background nodes continue async
    """
    
    def __init__(self, settings: Settings):
        """Initialize orchestrator.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.running = False
        
        self.zmq_manager: ZMQConnectionManager | None = None
        self.decision_engine: DecisionEngine | None = None
        
        # New algorithmic nodes
        self.memory_evaluator: MemoryEvaluatorNode | None = None
        self.trust_analysis: TrustAnalysisNode | None = None
        self.needs_analysis: NeedsAnalysisNode | None = None
        self.detox_scheduler: DetoxScheduler | None = None
        self.task_scheduler: TaskScheduler | None = None
        
        self.primary_llm = None
        self.sentiment_llm = None
        self.interpreter_llm = None
        self.rag_provider = None
        self.conversation_store = None
        self.embedding_service = None
        self.memory_decay = None
        
        self.sentiment_handler = None
        self.interpreter_handler = None
        self.primary_handler = None
        
        logger.info("Orchestrator initialized")
    
    def setup(self) -> None:
        """Set up all components."""
        logger.info("Setting up orchestrator components...")
        
        self.zmq_manager = ZMQConnectionManager.get_instance(
            input_endpoint=self.settings.input_endpoint,
            output_endpoint=self.settings.output_endpoint,
        )
        self.zmq_manager.setup()
        
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
        
        logger.info("Initializing task scheduler...")
        self.task_scheduler = TaskScheduler.get_instance()
        
        logger.info("Creating LLM providers...")
        self.primary_llm = create_primary_llm(self.settings)
        
        if self.settings.enable_sentiment_analysis:
            self.sentiment_llm = create_sentiment_llm(self.settings)
        
        if self.settings.enable_context_interpreter:
            self.interpreter_llm = create_interpreter_llm(self.settings)
        
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
        
        logger.info("Initializing memory evaluator node...")
        self.memory_evaluator = MemoryEvaluatorNode(
            llm_provider=self.primary_llm
        )
        
        logger.info("Initializing trust analysis node...")
        self.trust_analysis = TrustAnalysisNode(
            llm_provider=self.primary_llm
        )
        
        logger.info("Initializing needs analysis node...")
        self.needs_analysis = NeedsAnalysisNode(
            llm_provider=self.primary_llm
        )
        
        logger.info("Initializing detox scheduler...")
        self.detox_scheduler = DetoxScheduler(
            llm_provider=self.primary_llm,
            rag_provider=self.rag_provider,
            task_scheduler=self.task_scheduler,
            idle_trigger_minutes=self.settings.detox.idle_trigger_minutes,
            min_session_interval_minutes=self.settings.detox.min_interval_minutes,
            max_session_duration_minutes=self.settings.detox.max_duration_minutes
        )
        
        registry = NodeRegistry.get_instance()
        registry.register(SentimentAnalysisNode, "sentiment_analysis")
        registry.register(PrimaryResponseNode, "primary_response")
        registry.register(AckPreparationNode, "ack_preparation")
        registry.register(StoreConversationNode, "store_conversation")
        
        # Register new algorithmic nodes
        registry.register(MemoryEvaluatorNode, "memory_evaluator")
        registry.register(TrustAnalysisNode, "trust_analysis")
        registry.register(NeedsAnalysisNode, "needs_analysis")
        registry.register(DetoxScheduler, "detox_scheduler")
        registry.register(DetoxSessionNode, "detox_session")
        
        self.decision_engine = DecisionEngine(
            llm_provider=None,
            use_llm=False
        )
        
        logger.info("Orchestrator setup complete")
    
    def run(self) -> None:
        """Run the main orchestrator loop."""
        self.setup()
        self.running = True
        
        self._setup_shutdown_signals()
        
        logger.info("Orchestrator started. Waiting for requests...")
        
        try:
            asyncio.run(self._run_with_background_tasks())
        
        except Exception as e:
            logger.error(f"Fatal error in orchestrator loop: {e}", exc_info=True)
            raise
        
        finally:
            self.shutdown()
    
    async def _run_with_background_tasks(self) -> None:
        """Run main loop with background tasks."""
        # Start background tasks
        await self._start_background_tasks()
        
        # Run main loop
        await self._main_loop()
    
    async def _start_background_tasks(self) -> None:
        """Start background task runner for detox and other scheduled tasks."""
        if self.task_scheduler:
            logger.info("Starting background task scheduler...")
            asyncio.create_task(self.task_scheduler.run())
            
            # Schedule detox session if detox scheduler is available
            if self.detox_scheduler:
                logger.info("Scheduling detox sessions...")
                self.detox_scheduler.schedule_detox_session()
    
    async def _main_loop(self) -> None:
        """Main async orchestration loop."""
        while self.running:
            try:
                await self._process_request()
                
                flushed = self.zmq_manager.flush_queue(max_messages=10)
                if flushed > 0:
                    logger.debug(f"Flushed {flushed} queued messages")
            
            except Exception as e:
                logger.error(f"Error in main loop iteration: {e}", exc_info=True)
                await asyncio.sleep(0.1)
    
    async def _process_request(self) -> None:
        """Process a single request using node-based execution."""
        broker = KnowledgeBroker()
        queue_manager = TaskQueueManager()
        registry = NodeRegistry.get_instance()
        
        await queue_manager.enqueue(registry.create("receive_message", timeout_ms=1000))
        await queue_manager.execute_immediate(broker)
        
        if not broker.has_knowledge('dialogue_input'):
            queue_manager.reset()
            return
        
        dialogue_input = broker.dialogue_input
        
        # Update activity for detox scheduler
        if self.detox_scheduler:
            self.detox_scheduler.update_activity()
        
        logger.info(
            f"Processing request from '{dialogue_input.speaker}': "
            f"{dialogue_input.content[:100]}..."
        )
        
        registry = NodeRegistry.get_instance()
        
        node_names = await self.decision_engine.select_nodes(
            dialogue_input.content, 
            broker,
            registry=registry
        )
        
        for name in node_names:
            if name == "sentiment_analysis" and self.sentiment_handler:
                await queue_manager.enqueue(registry.create(name, handler=self.sentiment_handler))
            elif name == "memory_evaluator" and self.memory_evaluator:
                await queue_manager.enqueue(registry.create(name))
            elif name == "trust_analysis" and self.trust_analysis:
                await queue_manager.enqueue(registry.create(name))
            elif name == "needs_analysis" and self.needs_analysis:
                await queue_manager.enqueue(registry.create(name))
            elif name == "primary_response":
                await queue_manager.enqueue(registry.create(name, handler=self.primary_handler))
            else:
                await queue_manager.enqueue(registry.create(name))
        
        # Always enqueue post-processing and comm nodes
        await queue_manager.enqueue(registry.create("ack_preparation"))
        await queue_manager.enqueue(registry.create("store_conversation", 
            conversation_store=self.conversation_store, 
            rag_provider=self.rag_provider, 
            embedding_service=self.embedding_service))
        await queue_manager.enqueue(registry.create("send_acknowledgment"))
        await queue_manager.enqueue(registry.create("forward_response"))
        await queue_manager.enqueue(registry.create("check_feedback"))
        
        await queue_manager.execute_immediate(broker)
        asyncio.create_task(queue_manager.execute_background(broker))
        
        logger.info("Request processed successfully")
    
    
    def _setup_shutdown_signals(self) -> None:
        """Configure signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def shutdown(self) -> None:
        """Shut down orchestrator and clean up resources."""
        logger.info("Shutting down orchestrator...")
        
        # Stop task scheduler
        if self.task_scheduler:
            try:
                self.task_scheduler.stop()
            except Exception as e:
                logger.error(f"Error stopping task scheduler: {e}", exc_info=True)
        
        components = [
            self.sentiment_llm,
            self.interpreter_llm,
            self.primary_llm,
            self.rag_provider,
            self.conversation_store,
            self.zmq_manager,
        ]
        
        for component in components:
            if component and hasattr(component, 'close'):
                try:
                    component.close()
                except Exception as e:
                    logger.error(f"Error closing {component.__class__.__name__}: {e}", exc_info=True)
        
        logger.info("Orchestrator shut down complete")