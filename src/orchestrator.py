"""Node-based orchestrator for the companion AI pipeline."""

import asyncio
import logging
import time

from src.communication.zmq_handler import ZMQHandler
from src.config.settings import Settings
from src.handlers.primary_response import PrimaryResponseHandler
from src.handlers.user_fact_extraction import UserFactExtractionHandler
from src.handlers.memory_retrieval import MemoryRetrievalHandler
from src.handlers.memory_evaluation import MemoryEvaluationHandler
from src.handlers.needs_analysis import NeedsAnalysisHandler
from src.handlers.response_strategy import ResponseStrategyHandler
from src.handlers.memory_advisor import MemoryAdvisorHandler
from src.rag.algorithms.memory_chrono_decay import MemoryDecayAlgorithm
from src.llm.openrouter import OpenRouterLLM
from src.models.sentiment import DialogueInput
from src.nodes.core.result import NodeStatus
from src.nodes.orchestration.coordinator import Coordinator
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.nodes.orchestration.node_registry import NodeRegistry
from src.rag.embeddings import EmbeddingService
from src.rag.qdrant_connector import QdrantRAG
from src.rag.selector import RAGSelector
from src.storage.conversation_store import ConversationStore
from src.nodes.storage_nodes.conversation_storage import ConversationStorage

# Import node packages to trigger @register_node decorators
import src.nodes.algo_nodes  # noqa: F401
import src.nodes.communication_nodes  # noqa: F401
import src.nodes.processing  # noqa: F401
import src.nodes.storage_nodes  # noqa: F401

logger = logging.getLogger(__name__)

_MAX_NODES_PER_REQUEST = 20
_FALLBACK_RESPONSE = "I'm here, but something went wrong on my end. Give me a moment."


class Orchestrator:
    """Wires together all components and runs the request-handling loop.

    Startup sequence:
        1. Build shared deps (LLM, RAG, handlers, stores)
        2. Build NodeRegistry from all @register_node decorated classes
        3. Listen on ZMQ ROUTER for incoming messages
        4. Per request: run the DecisionEngine → node loop until complete
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._running = False
        self._zmq = ZMQHandler()
        self._registry, self._coordinator = self._build()

    def _build(self) -> tuple[NodeRegistry, Coordinator]:
        s = self.settings

        worker_llm = OpenRouterLLM(config=s.sentiment_llm)   # fast: gpt-oss-120b via Cerebras
        primary_llm = OpenRouterLLM(config=s.primary_llm)    # frontier: glm-4.7 via Cerebras
        rag = QdrantRAG(
            collection_name=s.qdrant.collection_name,
            embedding_dim=s.qdrant.embedding_dim,
            url=s.qdrant.url,
            api_key=s.qdrant.api_key,
            path=s.qdrant.path,
            selector=RAGSelector(
                max_documents=s.memory_decay.max_documents,
                min_score=s.memory_decay.retrieval_threshold,
            ),
        )
        embedding_service = EmbeddingService.get_instance()
        conversation_store = ConversationStore()

        primary_response_handler = PrimaryResponseHandler(
            llm_provider=primary_llm,
            rag_provider=rag,
            conversation_store=conversation_store,
        )
        user_fact_extraction_handler = UserFactExtractionHandler(
            llm_provider=worker_llm,
            rag_provider=rag,
            embedding_service=embedding_service,
            max_retries=s.sentiment.max_retries,
            retry_delay=s.sentiment.retry_delay,
        )
        memory_decay = MemoryDecayAlgorithm(
            memory_half_life_days=s.memory_decay.half_life_days,
            chrono_weight=s.memory_decay.chrono_weight,
            retrieval_threshold=s.memory_decay.retrieval_threshold,
            prune_threshold=s.memory_decay.prune_threshold,
            max_documents=s.memory_decay.max_documents,
        )
        memory_retrieval_handler = MemoryRetrievalHandler(
            rag=rag,
            embedding_service=embedding_service,
            memory_decay=memory_decay,
        )
        memory_evaluation_handler = MemoryEvaluationHandler(
            llm_provider=worker_llm,
            max_retries=s.sentiment.max_retries,
            retry_delay=s.sentiment.retry_delay,
        )
        needs_analysis_handler = NeedsAnalysisHandler(
            llm_provider=worker_llm,
            max_retries=s.sentiment.max_retries,
            retry_delay=s.sentiment.retry_delay,
        )
        response_strategy_handler = ResponseStrategyHandler(
            llm_provider=worker_llm,
            max_retries=s.sentiment.max_retries,
            retry_delay=s.sentiment.retry_delay,
        )
        memory_advisor_handler = MemoryAdvisorHandler(
            llm_provider=worker_llm,
            max_retries=s.sentiment.max_retries,
            retry_delay=s.sentiment.retry_delay,
        )

        registry = NodeRegistry.build(
            zmq_handler=self._zmq,
            rag=rag,
            embedding_service=embedding_service,
            conversation_store=conversation_store,
            primary_response_handler=primary_response_handler,
            user_fact_extraction_handler=user_fact_extraction_handler,
            memory_retrieval_handler=memory_retrieval_handler,
            memory_evaluation_handler=memory_evaluation_handler,
            needs_analysis_handler=needs_analysis_handler,
            response_strategy_handler=response_strategy_handler,
            memory_advisor_handler=memory_advisor_handler,
        )

        coordinator = Coordinator(
            _llm_provider=worker_llm,
            _conversation_store=conversation_store,
        )

        self._storage = ConversationStorage(
            conversation_store=conversation_store,
            rag=rag,
            embedding_service=embedding_service,
        )

        logger.info("Built registry with %d nodes", len(registry))
        return registry, coordinator

    # ------------------------------------------------------------------
    # Public interface

    def run(self) -> None:
        self._running = True
        logger.info("Orchestrator starting")
        try:
            asyncio.run(self._main_loop())
        except KeyboardInterrupt:
            pass
        finally:
            logger.info("Orchestrator stopped")
            self._zmq.close()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Main loop

    async def _main_loop(self) -> None:
        logger.info("Listening on %s", self.settings.zmq_input_endpoint)
        while self._running:
            identity, dialogue_input = await asyncio.to_thread(
                self._zmq.receive_request, 1000
            )
            if identity is None or dialogue_input is None:
                continue
            # Fire and move on — don't block the receive loop on processing
            asyncio.create_task(self._handle_request(identity, dialogue_input))

    # ------------------------------------------------------------------
    # Per-request handling

    async def _handle_request(
        self, identity: list[bytes], dialogue_input: DialogueInput
    ) -> None:
        broker = KnowledgeBroker(
            dialogue_input=dialogue_input,
            zmq_identity=identity,
        )

        logger.info("Request from '%s': %.80s", dialogue_input.speaker, dialogue_input.content)

        # ACK immediately — the caller shouldn't wait for processing to complete
        self._zmq.send_acknowledgment(identity, "processing", "Request received")

        try:
            await self._run_node_loop(broker)
        except Exception:
            logger.exception("Unhandled error in node loop for '%s'", dialogue_input.speaker)

        # Guarantee response delivery regardless of what happened in the loop
        response = broker.primary_response or _FALLBACK_RESPONSE
        if not broker.primary_response:
            logger.error("No primary response generated — sending fallback")
        self._zmq.forward_response(response)

        # Storage is not in the latency path
        asyncio.create_task(self._store(broker))

        summary = broker.get_execution_summary()
        logger.info(
            "Request complete — %d nodes, order: %s",
            summary["total_nodes_executed"],
            summary["execution_order"],
        )

    async def _run_node_loop(self, broker: KnowledgeBroker) -> None:
        for _ in range(_MAX_NODES_PER_REQUEST):
            node_name = self._coordinator.select_node(broker, self._registry)
            if node_name is None:
                break

            start = time.monotonic()
            result = await self._registry.execute(node_name, broker)
            duration = time.monotonic() - start

            if result is None:
                logger.error("Node '%s' not found", node_name)
                break

            broker.record_node_execution(node_name, result.status.value, duration)
            logger.debug("Node '%s' → %s (%.3fs)", node_name, result.status.value, duration)

            if result.status == NodeStatus.FAILED:
                logger.error("Node '%s' failed: %s", node_name, result.error)
                break
        else:
            logger.warning("Hit node limit (%d) without completing", _MAX_NODES_PER_REQUEST)

    async def _store(self, broker: KnowledgeBroker) -> None:
        if not broker.dialogue_input or not broker.primary_response:
            return
        try:
            await self._storage.execute(broker)
        except Exception:
            logger.exception("Background storage failed")
