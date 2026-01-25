"""Sentiment analysis processing node wrapping SentimentAnalysisHandler."""

from src.nodes.core.base import BaseNode
from src.nodes.core.result import NodeResult, NodeStatus
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.handlers.sentiment_analysis import SentimentAnalysisHandler
from src.rag.algorithms import MemoryDecayAlgorithm
from src.rag.embeddings import EmbeddingService
from src.rag.base import BaseRAG


class SentimentAnalysisNode(BaseNode):
    """Node that performs sentiment analysis using provided handler.
    
    Integrates memory decay algorithm to retrieve and filter relevant memories.
    """
    
    def __init__(
        self,
        handler: SentimentAnalysisHandler,
        rag_provider: BaseRAG,
        memory_half_life_days: float = 30.0,
        chrono_weight: float = 1.0,
        memory_retrieval_threshold: float = 0.15,
        max_context_documents: int = 25,
        **kwargs
    ):
        """Initialize sentiment analysis node.
        
        Args:
            handler: Sentiment analysis handler
            rag_provider: RAG provider for memory retrieval
            memory_half_life_days: Half-life for memory decay
            chrono_weight: Weight for chrono relevance
            memory_retrieval_threshold: Minimum score for retrieval
            max_context_documents: Maximum documents to retrieve
            **kwargs: Additional arguments passed to BaseNode
        """
        super().__init__(name="sentiment_analysis", priority=1, queue_type="immediate", **kwargs)
        self.handler = handler
        self.rag = rag_provider
        self.memory_algo = MemoryDecayAlgorithm(
            memory_half_life_days=memory_half_life_days,
            chrono_weight=chrono_weight
        )
        self.memory_retrieval_threshold = memory_retrieval_threshold
        self.max_context_documents = max_context_documents
    
    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        """Execute sentiment analysis with memory retrieval and decay filtering.
        
        Args:
            broker: Knowledge broker containing dialogue input
            
        Returns:
            NodeResult with sentiment analysis and retrieved memories
        """
        dialogue_input = broker.dialogue_input
        if not dialogue_input:
            return NodeResult(status=NodeStatus.FAILED, error="No dialogue_input in broker")
        
        # 1. Analyze sentiment
        sentiment = self.handler.analyze(dialogue_input.content, dialogue_input.speaker)
        if not sentiment:
            return NodeResult(status=NodeStatus.SKIPPED, metadata={'reason': 'analysis_failed'})
        
        broker.sentiment_analysis = sentiment
        
        # 2. Retrieve relevant memories using RAG
        embedding_service = EmbeddingService.get_instance()
        query_embedding = embedding_service.encode(dialogue_input.content)
        
        raw_docs = self.rag.retrieve_documents(
            query_embedding=query_embedding,
            limit=100
        )
        
        # 3. Apply memory decay filtering
        filtered_docs = self.memory_algo.filter_and_rank(
            documents=raw_docs,
            threshold=self.memory_retrieval_threshold,
            max_docs=self.max_context_documents
        )
        
        # 4. Update access counts for retrieved documents
        await self._update_access_counts(filtered_docs)
        
        # 5. Store results in broker
        broker.retrieved_documents = filtered_docs
        
        return NodeResult(
            status=NodeStatus.SUCCESS,
            data={
                'sentiment': sentiment,
                'retrieved_count': len(filtered_docs),
                'filtered_count': len(raw_docs) - len(filtered_docs)
            }
        )
    
    async def _update_access_counts(self, documents: list) -> None:
        """Update access counts for retrieved documents.
        
        Args:
            documents: List of retrieved documents
        """
        for doc in documents:
            point_id = doc.metadata.get("point_id")
            if point_id:
                await self.rag.update_access_count(point_id)
