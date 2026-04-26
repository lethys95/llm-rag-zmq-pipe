"""Composition root — constructs infrastructure primitives and wires the Orchestrator."""

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
from src.nodes.orchestration.node_registry import NodeRegistry
from src.nodes.orchestration.coordinator import Coordinator
from src.nodes.storage_nodes.conversation_storage import ConversationStorage
from src.nodes.orchestration.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


def _build_primary_llm(settings: Settings) -> BaseLLM:
    if isinstance(settings.primary_llm, LocalLLMConfig):
        from src.llm.llama_local import LlamaLocalLLM  # pylint: disable=import-outside-toplevel
        return LlamaLocalLLM(settings.primary_llm)
    return OpenRouterLLM(settings.primary_llm)


def build_orchestrator(settings: Settings) -> Orchestrator:
    """Construct infrastructure primitives and return a fully-wired Orchestrator."""

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

    registry = NodeRegistry.autowire(
        zmq_handler=zmq_handler,
        worker_llm=worker_llm,
        primary_llm=primary_llm,
        rag=rag,
        embedding_service=embedding_service,
        conversation_store=conversation_store,
        memory_decay=memory_decay,
        worker_call=settings.worker_call,
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
