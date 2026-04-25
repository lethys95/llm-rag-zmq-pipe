"""Node-based orchestrator for the companion AI pipeline."""

import asyncio
import logging
import re
import time
from datetime import datetime, timezone

from src.communication.zmq_handler import ZMQHandler
from src.config.settings import Settings
from src.handlers.primary_response import PrimaryResponseHandler
from src.handlers.user_fact_extraction import UserFactExtractionHandler
from src.handlers.memory_retrieval import MemoryRetrievalHandler
from src.handlers.memory_evaluation import MemoryEvaluationHandler
from src.handlers.needs_analysis import NeedsAnalysisHandler
from src.handlers.response_strategy import ResponseStrategyHandler
from src.handlers.emotional_state import EmotionalStateHandler
from src.handlers.memory_advisor import MemoryAdvisorHandler
from src.handlers.needs_advisor import NeedsAdvisorHandler
from src.handlers.strategy_advisor import StrategyAdvisorHandler
from src.handlers.format_advisor import FormatAdvisorHandler
from src.rag.algorithms.memory_chrono_decay import MemoryDecayAlgorithm
from src.llm.base import BaseLLM
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

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F780-\U0001F7FF"  # geometric
    "\U0001F800-\U0001F8FF"  # supplemental arrows
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "]+",
    flags=re.UNICODE,
)


def _strip_emojis(text: str) -> str:
    return _EMOJI_RE.sub("", text).strip()


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

    @staticmethod
    def _build_primary_llm(s: Settings) -> BaseLLM:
        if s.primary_llm.provider in ("llama", "llama_local"):
            from src.llm.llama_local import LlamaLocalLLM  # pylint: disable=import-outside-toplevel
            return LlamaLocalLLM()
        return OpenRouterLLM(config=s.primary_llm)

    def _build(self) -> tuple[NodeRegistry, Coordinator]:
        s = self.settings

        worker_llm = OpenRouterLLM(config=s.worker_llm)   # fast: gpt-oss-120b via Cerebras
        primary_llm: BaseLLM = self._build_primary_llm(s)
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
        )
        user_fact_extraction_handler = UserFactExtractionHandler(
            llm_provider=worker_llm,
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
        emotional_state_handler = EmotionalStateHandler(
            llm_provider=worker_llm,
            max_retries=s.sentiment.max_retries,
            retry_delay=s.sentiment.retry_delay,
        )
        needs_advisor_handler = NeedsAdvisorHandler()
        strategy_advisor_handler = StrategyAdvisorHandler()
        format_advisor_handler = FormatAdvisorHandler()

        self._conversation_store = conversation_store

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
            emotional_state_handler=emotional_state_handler,
            needs_advisor_handler=needs_advisor_handler,
            strategy_advisor_handler=strategy_advisor_handler,
            format_advisor_handler=format_advisor_handler,
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
            try:
                identity, dialogue_input = await asyncio.to_thread(
                    self._zmq.receive_request, 1000
                )
            except Exception:
                logger.exception("Unexpected error in ZMQ receive — continuing")
                continue
            if identity is None or dialogue_input is None:
                continue
            # Fire and move on — don't block the receive loop on processing
            asyncio.create_task(self._handle_request(identity, dialogue_input))

    # ------------------------------------------------------------------
    # Per-request handling

    def _get_idle_time_minutes(self) -> float | None:
        """Calculate minutes elapsed since the last stored conversation turn."""
        try:
            messages = self._conversation_store.get_recent_for_context(limit=1)
            if not messages:
                return None
            last_ts = messages[-1].timestamp
            last_dt = datetime.fromisoformat(last_ts)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return (now - last_dt).total_seconds() / 60.0
        except Exception:
            logger.debug("Could not calculate idle time", exc_info=True)
            return None

    async def _handle_request(
        self, identity: list[bytes], dialogue_input: DialogueInput
    ) -> None:
        broker = KnowledgeBroker(
            dialogue_input=dialogue_input,
            zmq_identity=identity,
            idle_time_minutes=self._get_idle_time_minutes(),
        )

        logger.info("Request from '%s': %.80s", dialogue_input.speaker, dialogue_input.content)

        # ACK immediately — the caller shouldn't wait for processing to complete
        self._zmq.send_acknowledgment(identity, "processing", "Request received")

        turn_start = time.monotonic()
        try:
            await self._run_node_loop(broker)
        except Exception:
            logger.exception("Unhandled error in node loop for '%s'", dialogue_input.speaker)
        turn_ms = (time.monotonic() - turn_start) * 1000

        # Guarantee response delivery regardless of what happened in the loop
        response = broker.primary_response or _FALLBACK_RESPONSE
        if not broker.primary_response:
            logger.error("No primary response generated — sending fallback")

        mode = broker.dialogue_input.mode if broker.dialogue_input else "spoken"
        if mode == "spoken":
            response = _strip_emojis(response)

        self._zmq.forward_response(response)

        # Storage is not in the latency path
        asyncio.create_task(self._store(broker))

        summary = broker.get_execution_summary()
        durations = broker.metadata.durations
        timing_str = "  ".join(
            f"{n}={durations[n]*1000:.0f}ms" for n in summary["execution_order"] if n in durations
        )
        logger.info(
            "Request complete — %d nodes, %.0fms total | %s",
            summary["total_nodes_executed"],
            turn_ms,
            timing_str or "(no timings)",
        )

    async def _run_node_loop(self, broker: KnowledgeBroker) -> None:
        nodes_run = 0
        round_num = 0
        for _ in range(_MAX_NODES_PER_REQUEST):
            coord_start = time.monotonic()
            batch = await asyncio.to_thread(
                self._coordinator.select_nodes, broker, self._registry
            )
            coord_ms = (time.monotonic() - coord_start) * 1000
            logger.debug("[coordinator] round %d selection: %.0fms", round_num, coord_ms)

            if batch is None:
                break

            round_num += 1
            batch_start = time.monotonic()
            results = await asyncio.gather(
                *(self._registry.execute(name, broker) for name in batch),
                return_exceptions=True,
            )
            batch_ms = (time.monotonic() - batch_start) * 1000

            node_timings = []
            for name, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error("Node '%s' raised exception: %s", name, result)
                    broker.record_node_execution(name, NodeStatus.FAILED.value, None)
                    node_timings.append(f"{name}=ERR")
                    continue
                if result is None:
                    logger.error("Node '%s' not found in registry", name)
                    node_timings.append(f"{name}=MISSING")
                    continue
                node_ms = result.metadata.get("duration_ms") if result.metadata else None
                broker.record_node_execution(name, result.status.value, node_ms / 1000 if node_ms else None)
                node_timings.append(f"{name}={result.status.value}")
                if result.status == NodeStatus.FAILED:
                    logger.error("Node '%s' failed: %s — continuing", name, result.error)

            logger.debug(
                "[batch %d] %.0fms total | %s",
                round_num, batch_ms, "  ".join(node_timings),
            )

            nodes_run += len(batch)
            if nodes_run >= _MAX_NODES_PER_REQUEST:
                logger.warning("Hit node limit (%d) without completing", _MAX_NODES_PER_REQUEST)
                break

    async def _store(self, broker: KnowledgeBroker) -> None:
        if not broker.dialogue_input or not broker.primary_response:
            return
        try:
            await self._storage.execute(broker)
        except Exception:
            logger.exception("Background storage failed")
