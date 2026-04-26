"""Composition root — constructs and wires all application dependencies."""

import logging

from src.config.settings import Settings, LocalLLMConfig
from src.communication.zmq_handler import ZMQHandler
from src.llm.base import BaseLLM
from src.llm.openrouter import OpenRouterLLM
from src.rag.embeddings import EmbeddingService
from src.rag.qdrant_connector import QdrantRAG
from src.rag.selector import RAGSelector
from src.rag.algorithms.memory_chrono_decay import MemoryDecayAlgorithm
from src.storage.sqlite_connection import SQLiteConnection
from src.storage.conversation_store import ConversationStore
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
from src.nodes.orchestration.node_registry import NodeRegistry
from src.nodes.orchestration.coordinator import Coordinator
from src.nodes.storage_nodes.conversation_storage import ConversationStorage
from src.nodes.orchestration.orchestrator import Orchestrator

# Trigger @register_node decorators so NodeRegistry.build() can find all nodes
import src.nodes.algo_nodes  # noqa: F401
import src.nodes.communication_nodes  # noqa: F401
import src.nodes.processing  # noqa: F401
import src.nodes.storage_nodes  # noqa: F401

logger = logging.getLogger(__name__)


def _build_primary_llm(settings: Settings) -> BaseLLM:
    if isinstance(settings.primary_llm, LocalLLMConfig):
        from src.llm.llama_local import LlamaLocalLLM  # pylint: disable=import-outside-toplevel
        return LlamaLocalLLM(settings.primary_llm)
    return OpenRouterLLM(settings.primary_llm)


def build_orchestrator(settings: Settings) -> Orchestrator:
    """Construct and wire all dependencies, return a ready Orchestrator."""

    zmq_handler = ZMQHandler(settings)
    embedding_service = EmbeddingService(settings)

    sqlite_connection = SQLiteConnection(settings.conversation_store.db_path)
    conversation_store = ConversationStore(settings, connection=sqlite_connection)

    rag = QdrantRAG(
        collection_name=settings.qdrant.collection_name,
        embedding_service=embedding_service,
        embedding_dim=settings.qdrant.embedding_dim,
        url=settings.qdrant.url,
        api_key=settings.qdrant.api_key,
        path=settings.qdrant.path,
        selector=RAGSelector(
            max_documents=settings.memory_decay.max_documents,
            min_score=settings.memory_decay.retrieval_threshold,
        ),
    )

    worker_llm = OpenRouterLLM(settings.worker_llm)
    primary_llm = _build_primary_llm(settings)

    memory_decay = MemoryDecayAlgorithm(
        memory_half_life_days=settings.memory_decay.half_life_days,
        chrono_weight=settings.memory_decay.chrono_weight,
        retrieval_threshold=settings.memory_decay.retrieval_threshold,
        prune_threshold=settings.memory_decay.prune_threshold,
        max_documents=settings.memory_decay.max_documents,
    )

    primary_response_handler = PrimaryResponseHandler(llm_provider=primary_llm)
    user_fact_extraction_handler = UserFactExtractionHandler(
        llm_provider=worker_llm,
        max_retries=settings.sentiment.max_retries,
        retry_delay=settings.sentiment.retry_delay,
    )
    memory_retrieval_handler = MemoryRetrievalHandler(
        rag=rag,
        embedding_service=embedding_service,
        memory_decay=memory_decay,
    )
    memory_evaluation_handler = MemoryEvaluationHandler(
        llm_provider=worker_llm,
        max_retries=settings.sentiment.max_retries,
        retry_delay=settings.sentiment.retry_delay,
    )
    needs_analysis_handler = NeedsAnalysisHandler(
        llm_provider=worker_llm,
        max_retries=settings.sentiment.max_retries,
        retry_delay=settings.sentiment.retry_delay,
    )
    response_strategy_handler = ResponseStrategyHandler(
        llm_provider=worker_llm,
        max_retries=settings.sentiment.max_retries,
        retry_delay=settings.sentiment.retry_delay,
    )
    memory_advisor_handler = MemoryAdvisorHandler(
        llm_provider=worker_llm,
        max_retries=settings.sentiment.max_retries,
        retry_delay=settings.sentiment.retry_delay,
    )
    emotional_state_handler = EmotionalStateHandler(
        llm_provider=worker_llm,
        max_retries=settings.sentiment.max_retries,
        retry_delay=settings.sentiment.retry_delay,
    )
    needs_advisor_handler = NeedsAdvisorHandler()
    strategy_advisor_handler = StrategyAdvisorHandler()
    format_advisor_handler = FormatAdvisorHandler()

    registry = NodeRegistry.build(
        zmq_handler=zmq_handler,
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

    storage = ConversationStorage(
        conversation_store=conversation_store,
        rag=rag,
        embedding_service=embedding_service,
    )

    logger.info("Built registry with %d nodes", len(registry))

    return Orchestrator(
        settings=settings,
        zmq_handler=zmq_handler,
        registry=registry,
        coordinator=coordinator,
        conversation_store=conversation_store,
        storage=storage,
    )
