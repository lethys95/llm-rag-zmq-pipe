"""Tests for algorithmic nodes: MemoryEvaluatorNode, TrustAnalysisNode, DetoxScheduler."""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.nodes.algo_nodes.memory_evaluator_node import MemoryEvaluatorNode, MemoryEvaluation
from src.nodes.algo_nodes.trust_analysis_node import TrustAnalysisNode, TrustAnalysis, TrustRecord
from src.nodes.algo_nodes.detox_scheduler import DetoxScheduler, DetoxSessionNode
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.rag.selector import RAGDocument
from src.models.sentiment import SentimentAnalysis
from src.models.memory import ConversationState
from src.llm.base import BaseLLM
from src.rag.base import BaseRAG
from src.chrono.task_scheduler import TaskScheduler


class MockLLM(BaseLLM):
    """Mock LLM provider for testing."""
    
    def __init__(self, response: str = '{"relevance": 0.8, "chrono_relevance": 0.7, "reasoning": "test", "should_boost": true, "boost_factor": 0.2}'):
        self.response = response
        self.generate_calls = []
    
    def generate(self, prompt: str) -> str:
        self.generate_calls.append(prompt)
        return self.response
    
    def close(self) -> None:
        pass


class MockRAG(BaseRAG):
    """Mock RAG provider for testing."""
    
    def __init__(self):
        self.stored_documents = []
        self.access_updates = []
    
    def retrieve(self, query: str) -> str:
        return "test context"
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> list[RAGDocument]:
        return [
            RAGDocument(
                content="test content",
                score=0.8,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "sentiment": "positive",
                    "emotional_tone": "happy",
                    "context_summary": "test summary",
                    "key_topics": ["test"],
                    "relevance": 0.8,
                    "chrono_relevance": 0.7
                }
            )
        ]
    
    def store(
        self,
        text: str,
        embedding: list[float],
        metadata: dict[str, object] | None = None,
        point_id: str | None = None
    ) -> str:
        self.stored_documents.append({
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
            "point_id": point_id
        })
        return point_id or "test_id"
    
    def update_access_count(self, point_id: str) -> None:
        self.access_updates.append(point_id)
    
    def close(self) -> None:
        pass


@pytest.fixture
def mock_llm() -> MockLLM:
    """Provide a mock LLM provider."""
    return MockLLM()


@pytest.fixture
def mock_rag() -> MockRAG:
    """Provide a mock RAG provider."""
    return MockRAG()


@pytest.fixture
def sample_broker() -> KnowledgeBroker:
    """Provide a sample knowledge broker with test data."""
    broker = KnowledgeBroker()
    
    # Add sample conversation history
    broker.conversation_history = [
        SentimentAnalysis(
            sentiment="positive",
            emotional_tone="happy",
            key_topics=["test", "topic1"],
            reasoning="test reasoning"
        )
    ]
    
    # Add sample retrieved documents
    broker.retrieved_documents = [
        RAGDocument(
            content="test memory content",
            score=0.8,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "sentiment": "positive",
                "emotional_tone": "happy",
                "context_summary": "test summary",
                "key_topics": ["test"],
                "relevance": 0.8,
                "chrono_relevance": 0.7
            }
        )
    ]
    
    # Add sample trust analysis
    broker.trust_analysis = TrustAnalysis(
        score=0.7,
        relationship_age_days=30,
        interaction_frequency=10,
        positive_ratio=0.8,
        consistency_score=0.75,
        depth_score=0.6,
        factors=["test_factor"]
    )
    
    return broker


class TestMemoryEvaluatorNode:
    """Tests for MemoryEvaluatorNode."""
    
    def test_initialization(self, mock_llm: MockLLM) -> None:
        """Test MemoryEvaluatorNode initialization."""
        node = MemoryEvaluatorNode(llm_provider=mock_llm)
        
        assert node.name == "memory_evaluator"
        assert node.priority == 2
        assert node.queue_type == "immediate"
        assert node.llm == mock_llm
        assert node.max_retries == 3
        assert node.retry_delay == 0.5
    
    @pytest.mark.asyncio
    async def test_execute_with_documents(self, mock_llm: MockLLM, sample_broker: KnowledgeBroker) -> None:
        """Test execution with retrieved documents."""
        node = MemoryEvaluatorNode(llm_provider=mock_llm)
        result = await node.execute(sample_broker)
        
        assert result.status.name == "SUCCESS"
        assert result.data["evaluated_count"] == 1
        assert result.data["total_count"] == 1
        assert len(sample_broker.evaluated_memories) == 1
        
        doc, evaluation = sample_broker.evaluated_memories[0]
        assert isinstance(evaluation, MemoryEvaluation)
        assert 0.0 <= evaluation.relevance <= 1.0
        assert 0.0 <= evaluation.chrono_relevance <= 1.0
        assert isinstance(evaluation.reasoning, str)
        assert isinstance(evaluation.should_boost, bool)
        assert 0.0 <= evaluation.boost_factor <= 1.0
    
    @pytest.mark.asyncio
    async def test_execute_without_documents(self, mock_llm: MockLLM) -> None:
        """Test execution without retrieved documents."""
        broker = KnowledgeBroker()
        node = MemoryEvaluatorNode(llm_provider=mock_llm)
        result = await node.execute(broker)
        
        assert result.status.name == "SKIPPED"
        assert result.metadata["reason"] == "no_documents"
    
    @pytest.mark.asyncio
    async def test_get_conversation_state(self, mock_llm: MockLLM, sample_broker: KnowledgeBroker) -> None:
        """Test conversation state extraction."""
        node = MemoryEvaluatorNode(llm_provider=mock_llm)
        state = node._get_conversation_state(sample_broker)
        
        assert isinstance(state, ConversationState)
        assert state.message_count == 1
        assert "test" in state.recent_topics
        assert "topic1" in state.recent_topics
        assert state.emotional_tone == "happy"
    
    @pytest.mark.asyncio
    async def test_parse_evaluation_valid_json(self, mock_llm: MockLLM) -> None:
        """Test parsing valid JSON evaluation."""
        node = MemoryEvaluatorNode(llm_provider=mock_llm)
        response = '{"relevance": 0.8, "chrono_relevance": 0.7, "reasoning": "test", "should_boost": true, "boost_factor": 0.2}'
        
        evaluation = node._parse_evaluation(response)
        
        assert evaluation is not None
        assert evaluation.relevance == 0.8
        assert evaluation.chrono_relevance == 0.7
        assert evaluation.reasoning == "test"
        assert evaluation.should_boost is True
        assert evaluation.boost_factor == 0.2
    
    @pytest.mark.asyncio
    async def test_parse_evaluation_invalid_json(self, mock_llm: MockLLM) -> None:
        """Test parsing invalid JSON evaluation."""
        node = MemoryEvaluatorNode(llm_provider=mock_llm)
        response = "invalid json"
        
        evaluation = node._parse_evaluation(response)
        
        assert evaluation is None
    
    @pytest.mark.asyncio
    async def test_parse_evaluation_clamps_values(self, mock_llm: MockLLM) -> None:
        """Test that parsing clamps values to valid range."""
        node = MemoryEvaluatorNode(llm_provider=mock_llm)
        response = '{"relevance": 1.5, "chrono_relevance": -0.5, "reasoning": "test", "should_boost": true, "boost_factor": 2.0}'
        
        evaluation = node._parse_evaluation(response)
        
        assert evaluation is not None
        assert evaluation.relevance == 1.0  # Clamped to max
        assert evaluation.chrono_relevance == 0.0  # Clamped to min
        assert evaluation.boost_factor == 1.0  # Clamped to max
    
    @pytest.mark.asyncio
    async def test_extract_json_from_text(self, mock_llm: MockLLM) -> None:
        """Test extracting JSON from text with extra content."""
        node = MemoryEvaluatorNode(llm_provider=mock_llm)
        text = "Here's the result: {\"relevance\": 0.8, \"chrono_relevance\": 0.7, \"reasoning\": \"test\", \"should_boost\": true, \"boost_factor\": 0.2} Done."
        
        json_str = node._extract_json(text)
        
        assert json_str == '{"relevance": 0.8, "chrono_relevance": 0.7, "reasoning": "test", "should_boost": true, "boost_factor": 0.2}'


class TestTrustAnalysisNode:
    """Tests for TrustAnalysisNode."""
    
    def test_initialization(self, mock_llm: MockLLM) -> None:
        """Test TrustAnalysisNode initialization."""
        node = TrustAnalysisNode(llm_provider=mock_llm)
        
        assert node.name == "trust_analysis"
        assert node.priority == 1
        assert node.queue_type == "immediate"
        assert node.llm == mock_llm
    
    @pytest.mark.asyncio
    async def test_execute_with_history(self, mock_llm: MockLLM, sample_broker: KnowledgeBroker) -> None:
        """Test execution with conversation history."""
        node = TrustAnalysisNode(llm_provider=mock_llm)
        result = await node.execute(sample_broker)
        
        assert result.status.name == "SUCCESS"
        assert hasattr(sample_broker, "trust_analysis")
        assert isinstance(sample_broker.trust_analysis, TrustAnalysis)
        assert 0.0 <= sample_broker.trust_analysis.score <= 1.0
    
    @pytest.mark.asyncio
    async def test_execute_without_history(self, mock_llm: MockLLM) -> None:
        """Test execution without conversation history."""
        broker = KnowledgeBroker()
        broker.conversation_history = []
        node = TrustAnalysisNode(llm_provider=mock_llm)
        result = await node.execute(broker)
        
        assert result.status.name == "SUCCESS"
        assert hasattr(broker, "trust_analysis")
        # Should create a new trust record with low score
        assert 0.0 <= broker.trust_analysis.score <= 1.0
    
    def test_calculate_trust_score_new_user(self, mock_llm: MockLLM) -> None:
        """Test trust score calculation for new user."""
        node = TrustAnalysisNode(llm_provider=mock_llm)
        record = TrustRecord(
            user_id="test_user",
            first_interaction=datetime.now(),
            last_interaction=datetime.now(),
            total_interactions=1,
            positive_interactions=1,
            negative_interactions=0,
            trust_history=[]
        )
        
        score = node._calculate_trust_score(record)
        
        assert 0.0 <= score <= 1.0
        # New user should have moderate trust score
        assert score < 0.5
    
    def test_calculate_trust_score_established_user(self, mock_llm: MockLLM) -> None:
        """Test trust score calculation for established user."""
        node = TrustAnalysisNode(llm_provider=mock_llm)
        now = datetime.now()
        record = TrustRecord(
            user_id="test_user",
            first_interaction=now - timedelta(days=60),
            last_interaction=now,
            total_interactions=50,
            positive_interactions=40,
            negative_interactions=10,
            trust_history=[]
        )
        
        score = node._calculate_trust_score(record)
        
        assert 0.0 <= score <= 1.0
        # Established user with good ratio should have higher trust
        assert score > 0.5


class TestDetoxScheduler:
    """Tests for DetoxScheduler."""
    
    def test_initialization(self, mock_llm: MockLLM, mock_rag: MockRAG) -> None:
        """Test DetoxScheduler initialization."""
        task_scheduler = TaskScheduler.get_instance()
        scheduler = DetoxScheduler(
            llm_provider=mock_llm,
            rag_provider=mock_rag,
            task_scheduler=task_scheduler,
            idle_trigger_minutes=60,
            min_session_interval_minutes=120,
            max_session_duration_minutes=30
        )
        
        assert scheduler.llm_provider == mock_llm
        assert scheduler.rag_provider == mock_rag
        assert scheduler.idle_trigger_minutes == 60
        assert scheduler.min_session_interval_minutes == 120
        assert scheduler.max_session_duration_minutes == 30
        assert scheduler.is_detox_running is False
    
    def test_update_activity(self, mock_llm: MockLLM, mock_rag: MockRAG) -> None:
        """Test activity update."""
        task_scheduler = TaskScheduler.get_instance()
        scheduler = DetoxScheduler(
            llm_provider=mock_llm,
            rag_provider=mock_rag,
            task_scheduler=task_scheduler
        )
        
        initial_activity = scheduler.last_activity
        scheduler.update_activity()
        
        assert scheduler.last_activity > initial_activity
    
    def test_should_run_detox_idle_time(self, mock_llm: MockLLM, mock_rag: MockRAG) -> None:
        """Test detox trigger based on idle time."""
        task_scheduler = TaskScheduler.get_instance()
        scheduler = DetoxScheduler(
            llm_provider=mock_llm,
            rag_provider=mock_rag,
            task_scheduler=task_scheduler,
            idle_trigger_minutes=1
        )
        
        # Set activity to 2 minutes ago
        scheduler.last_activity = datetime.now() - timedelta(minutes=2)
        
        assert scheduler.should_run_detox() is True
    
    def test_should_run_detox_not_idle(self, mock_llm: MockLLM, mock_rag: MockRAG) -> None:
        """Test detox not triggered when not idle."""
        task_scheduler = TaskScheduler.get_instance()
        scheduler = DetoxScheduler(
            llm_provider=mock_llm,
            rag_provider=mock_rag,
            task_scheduler=task_scheduler,
            idle_trigger_minutes=60
        )
        
        # Set activity to 1 minute ago
        scheduler.last_activity = datetime.now() - timedelta(minutes=1)
        
        assert scheduler.should_run_detox() is False
    
    def test_should_run_detox_already_running(self, mock_llm: MockLLM, mock_rag: MockRAG) -> None:
        """Test detox not triggered when already running."""
        task_scheduler = TaskScheduler.get_instance()
        scheduler = DetoxScheduler(
            llm_provider=mock_llm,
            rag_provider=mock_rag,
            task_scheduler=task_scheduler,
            idle_trigger_minutes=1
        )
        
        scheduler.is_detox_running = True
        scheduler.last_activity = datetime.now() - timedelta(minutes=2)
        
        assert scheduler.should_run_detox() is False
    
    def test_should_run_detox_min_interval(self, mock_llm: MockLLM, mock_rag: MockRAG) -> None:
        """Test detox not triggered within minimum interval."""
        task_scheduler = TaskScheduler.get_instance()
        scheduler = DetoxScheduler(
            llm_provider=mock_llm,
            rag_provider=mock_rag,
            task_scheduler=task_scheduler,
            idle_trigger_minutes=1,
            min_session_interval_minutes=60
        )
        
        scheduler.last_activity = datetime.now() - timedelta(minutes=2)
        scheduler.last_detox_session = datetime.now() - timedelta(minutes=30)
        
        assert scheduler.should_run_detox() is False
    
    def test_get_idle_time(self, mock_llm: MockLLM, mock_rag: MockRAG) -> None:
        """Test getting idle time."""
        task_scheduler = TaskScheduler.get_instance()
        scheduler = DetoxScheduler(
            llm_provider=mock_llm,
            rag_provider=mock_rag,
            task_scheduler=task_scheduler
        )
        
        scheduler.last_activity = datetime.now() - timedelta(minutes=5)
        idle_time = scheduler.get_idle_time()
        
        assert idle_time.total_seconds() >= 300  # 5 minutes in seconds
        assert idle_time.total_seconds() < 310  # Allow some margin
    
    def test_start_end_detox_session(self, mock_llm: MockLLM, mock_rag: MockRAG) -> None:
        """Test starting and ending detox session."""
        task_scheduler = TaskScheduler.get_instance()
        scheduler = DetoxScheduler(
            llm_provider=mock_llm,
            rag_provider=mock_rag,
            task_scheduler=task_scheduler
        )
        
        assert scheduler.is_detox_running is False
        
        scheduler.start_detox_session()
        assert scheduler.is_detox_running is True
        assert scheduler.last_detox_session is not None
        
        scheduler.end_detox_session()
        assert scheduler.is_detox_running is False


class TestDetoxSessionNode:
    """Tests for DetoxSessionNode."""
    
    def test_initialization(self, mock_llm: MockLLM, mock_rag: MockRAG) -> None:
        """Test DetoxSessionNode initialization."""
        from src.rag.algorithms.nudging_algorithm import NudgingAlgorithm
        from src.nodes.algo_nodes.memory_consolidation_node import MemoryConsolidationNode
        
        nudging = NudgingAlgorithm()
        consolidation = MemoryConsolidationNode(
            llm_provider=mock_llm,
            rag_provider=mock_rag
        )
        
        node = DetoxSessionNode(
            nudging_algorithm=nudging,
            memory_consolidation_node=consolidation,
            rag_provider=mock_rag,
            max_session_duration_minutes=30
        )
        
        assert node.name == "detox_session"
        assert node.priority == 10
        assert node.queue_type == "background"
        assert node.max_session_duration_minutes == 30
    
    @pytest.mark.asyncio
    async def test_execute_without_history(self, mock_llm: MockLLM, mock_rag: MockRAG) -> None:
        """Test execution without conversation history."""
        from src.rag.algorithms.nudging_algorithm import NudgingAlgorithm
        from src.nodes.algo_nodes.memory_consolidation_node import MemoryConsolidationNode
        
        nudging = NudgingAlgorithm()
        consolidation = MemoryConsolidationNode(
            llm_provider=mock_llm,
            rag_provider=mock_rag
        )
        
        node = DetoxSessionNode(
            nudging_algorithm=nudging,
            memory_consolidation_node=consolidation,
            rag_provider=mock_rag
        )
        
        broker = KnowledgeBroker()
        result = await node.execute(broker)
        
        assert result.status.name == "SKIPPED"
        assert result.metadata["reason"] == "no_conversation_history"
    
    @pytest.mark.asyncio
    async def test_store_companion_state(self, mock_llm: MockLLM, mock_rag: MockRAG) -> None:
        """Test storing companion state."""
        from src.rag.algorithms.nudging_algorithm import NudgingAlgorithm
        from src.nodes.algo_nodes.memory_consolidation_node import MemoryConsolidationNode
        
        nudging = NudgingAlgorithm()
        consolidation = MemoryConsolidationNode(
            llm_provider=mock_llm,
            rag_provider=mock_rag
        )
        
        node = DetoxSessionNode(
            nudging_algorithm=nudging,
            memory_consolidation_node=consolidation,
            rag_provider=mock_rag
        )
        
        await node._store_companion_state("test_topic", 0.5)
        
        assert len(mock_rag.stored_documents) == 1
        doc = mock_rag.stored_documents[0]
        assert "test_topic" in doc["text"]
        assert doc["metadata"]["type"] == "companion_state"
        assert doc["metadata"]["topic"] == "test_topic"
        assert doc["metadata"]["position"] == 0.5
    
    def test_extract_topics(self, mock_llm: MockLLM, mock_rag: MockRAG) -> None:
        """Test topic extraction from conversation history."""
        from src.rag.algorithms.nudging_algorithm import NudgingAlgorithm
        from src.nodes.algo_nodes.memory_consolidation_node import MemoryConsolidationNode
        
        nudging = NudgingAlgorithm()
        consolidation = MemoryConsolidationNode(
            llm_provider=mock_llm,
            rag_provider=mock_rag
        )
        
        node = DetoxSessionNode(
            nudging_algorithm=nudging,
            memory_consolidation_node=consolidation,
            rag_provider=mock_rag
        )
        
        history = [
            SentimentAnalysis(
                sentiment="positive",
                emotional_tone="happy",
                key_topics=["topic1", "topic2"],
                reasoning="test"
            ),
            SentimentAnalysis(
                sentiment="negative",
                emotional_tone="sad",
                key_topics=["topic1", "topic3"],
                reasoning="test"
            )
        ]
        
        topics = node._extract_topics(history)
        
        assert len(topics) == 3  # Deduplicated topics
        topic_names = [t[0] for t in topics]
        assert "topic1" in topic_names
        assert "topic2" in topic_names
        assert "topic3" in topic_names
    
    def test_estimate_user_position(self, mock_llm: MockLLM, mock_rag: MockRAG) -> None:
        """Test user position estimation."""
        from src.rag.algorithms.nudging_algorithm import NudgingAlgorithm
        from src.nodes.algo_nodes.memory_consolidation_node import MemoryConsolidationNode
        
        nudging = NudgingAlgorithm()
        consolidation = MemoryConsolidationNode(
            llm_provider=mock_llm,
            rag_provider=mock_rag
        )
        
        node = DetoxSessionNode(
            nudging_algorithm=nudging,
            memory_consolidation_node=consolidation,
            rag_provider=mock_rag
        )
        
        # Test positive sentiment
        msg = SentimentAnalysis(
            sentiment="positive",
            emotional_tone="excited",
            key_topics=[],
            reasoning="test"
        )
        position = node._estimate_user_position(msg)
        assert position > 0
        
        # Test negative sentiment
        msg = SentimentAnalysis(
            sentiment="negative",
            emotional_tone="angry",
            key_topics=[],
            reasoning="test"
        )
        position = node._estimate_user_position(msg)
        assert position < 0
        
        # Test neutral sentiment
        msg = SentimentAnalysis(
            sentiment="neutral",
            emotional_tone=None,
            key_topics=[],
            reasoning="test"
        )
        position = node._estimate_user_position(msg)
        assert position == 0.0
    
    def test_generate_conversational_guidance(self, mock_llm: MockLLM, mock_rag: MockRAG) -> None:
        """Test conversational guidance generation."""
        from src.rag.algorithms.nudging_algorithm import NudgingAlgorithm
        from src.nodes.algo_nodes.memory_consolidation_node import MemoryConsolidationNode
        
        nudging = NudgingAlgorithm()
        consolidation = MemoryConsolidationNode(
            llm_provider=mock_llm,
            rag_provider=mock_rag
        )
        
        node = DetoxSessionNode(
            nudging_algorithm=nudging,
            memory_consolidation_node=consolidation,
            rag_provider=mock_rag
        )
        
        # Test with no nudges
        guidance = node._generate_conversational_guidance([])
        assert guidance == "No specific guidance needed."
        
        # Test with significant nudges
        nudges = [
            {"topic": "topic1", "nudge_amount": 0.1},
            {"topic": "topic2", "nudge_amount": 0.15}
        ]
        guidance = node._generate_conversational_guidance(nudges)
        assert "topic1" in guidance
        assert "topic2" in guidance
        assert "alternative perspectives" in guidance
