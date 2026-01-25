"""Tests for DecisionEngine with new algorithmic nodes."""

import pytest

from src.nodes.orchestration.decision_engine import DecisionEngine
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.models.sentiment import SentimentAnalysis


class TestDecisionEngineWithAlgoNodes:
    """Tests for DecisionEngine with algorithmic nodes."""
    
    def test_initialization(self) -> None:
        """Test DecisionEngine initialization."""
        engine = DecisionEngine(llm_provider=None, use_llm=False)
        
        assert engine.llm_provider is None
        assert engine.use_llm is False
    
    def test_initialization_with_llm(self) -> None:
        """Test DecisionEngine initialization with LLM."""
        from src.llm.base import BaseLLM
        
        class MockLLM(BaseLLM):
            def generate(self, prompt: str) -> str:
                return '["sentiment_analysis", "primary_response"]'
            
            def close(self) -> None:
                pass
        
        mock_llm = MockLLM()
        engine = DecisionEngine(llm_provider=mock_llm, use_llm=True)
        
        assert engine.llm_provider == mock_llm
        assert engine.use_llm is True
    
    def test_rule_based_selection_basic(self) -> None:
        """Test basic rule-based node selection."""
        engine = DecisionEngine(llm_provider=None, use_llm=False)
        broker = KnowledgeBroker()
        
        nodes = engine._rule_based_selection("Hello, how are you?", broker)
        
        assert "sentiment_analysis" in nodes
        assert "primary_response" in nodes
        assert "crisis_detection" not in nodes
    
    def test_rule_based_selection_crisis(self) -> None:
        """Test crisis detection in node selection."""
        engine = DecisionEngine(llm_provider=None, use_llm=False)
        broker = KnowledgeBroker()
        
        nodes = engine._rule_based_selection("I want to kill myself", broker)
        
        assert "crisis_detection" in nodes
        assert "sentiment_analysis" in nodes
        assert "primary_response" in nodes
    
    def test_rule_based_selection_idle_time(self) -> None:
        """Test detox protocol selection on idle time."""
        engine = DecisionEngine(llm_provider=None, use_llm=False)
        broker = KnowledgeBroker()
        broker.idle_time_minutes = 90
        
        nodes = engine._rule_based_selection("Hello", broker)
        
        assert "detox_protocol" in nodes
        assert "sentiment_analysis" in nodes
        assert "primary_response" in nodes
    
    def test_rule_based_selection_no_idle_time(self) -> None:
        """Test detox protocol not selected when not idle."""
        engine = DecisionEngine(llm_provider=None, use_llm=False)
        broker = KnowledgeBroker()
        broker.idle_time_minutes = 30
        
        nodes = engine._rule_based_selection("Hello", broker)
        
        assert "detox_protocol" not in nodes
    
    def test_rule_based_selection_trust_analysis_first_message(self) -> None:
        """Test trust analysis on first message."""
        engine = DecisionEngine(llm_provider=None, use_llm=False)
        broker = KnowledgeBroker()
        broker.conversation_history = []
        
        nodes = engine._rule_based_selection("Hello", broker)
        
        assert "trust_analysis" in nodes
    
    def test_rule_based_selection_trust_analysis_tenth_message(self) -> None:
        """Test trust analysis on every 10th message."""
        engine = DecisionEngine(llm_provider=None, use_llm=False)
        broker = KnowledgeBroker()
        
        # Create 10 messages
        broker.conversation_history = [
            SentimentAnalysis(
                sentiment="neutral",
                confidence=0.9,
                memory_owner="user",
                emotional_tone=None,
                key_topics=[],
                reasoning="test"
            )
            for _ in range(10)
        ]
        
        nodes = engine._rule_based_selection("Hello", broker)
        
        assert "trust_analysis" in nodes
    
    def test_rule_based_selection_no_trust_analysis(self) -> None:
        """Test trust analysis not selected on non-multiple messages."""
        engine = DecisionEngine(llm_provider=None, use_llm=False)
        broker = KnowledgeBroker()
        
        # Create 5 messages
        broker.conversation_history = [
            SentimentAnalysis(
                sentiment="neutral",
                confidence=0.9,
                memory_owner="user",
                emotional_tone=None,
                key_topics=[],
                reasoning="test"
            )
            for _ in range(5)
        ]
        
        nodes = engine._rule_based_selection("Hello", broker)
        
        assert "trust_analysis" not in nodes
    
    def test_rule_based_selection_memory_evaluator_with_documents(self) -> None:
        """Test memory evaluator selection with retrieved documents."""
        from src.rag.selector import RAGDocument
        from datetime import datetime
        
        engine = DecisionEngine(llm_provider=None, use_llm=False)
        broker = KnowledgeBroker()
        
        # Add retrieved documents
        broker.retrieved_documents = [
            RAGDocument(
                content="test content",
                score=0.8,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "sentiment": "positive"
                }
            )
        ]
        
        nodes = engine._rule_based_selection("Hello", broker)
        
        assert "memory_evaluator" in nodes
    
    def test_rule_based_selection_no_memory_evaluator_without_documents(self) -> None:
        """Test memory evaluator not selected without documents."""
        engine = DecisionEngine(llm_provider=None, use_llm=False)
        broker = KnowledgeBroker()
        broker.retrieved_documents = []
        
        nodes = engine._rule_based_selection("Hello", broker)
        
        assert "memory_evaluator" not in nodes
    
    def test_validate_node_selection_valid_nodes(self) -> None:
        """Test validation of valid node names."""
        engine = DecisionEngine(llm_provider=None, use_llm=False)
        
        nodes = engine._validate_node_selection([
            "sentiment_analysis",
            "primary_response",
            "memory_evaluator",
            "trust_analysis"
        ])
        
        assert "sentiment_analysis" in nodes
        assert "primary_response" in nodes
        assert "memory_evaluator" in nodes
        assert "trust_analysis" in nodes
    
    def test_validate_node_selection_invalid_nodes(self) -> None:
        """Test validation filters out invalid node names."""
        engine = DecisionEngine(llm_provider=None, use_llm=False)
        
        nodes = engine._validate_node_selection([
            "sentiment_analysis",
            "invalid_node",
            "primary_response",
            "another_invalid"
        ])
        
        assert "sentiment_analysis" in nodes
        assert "primary_response" in nodes
        assert "invalid_node" not in nodes
        assert "another_invalid" not in nodes
    
    def test_validate_node_selection_adds_primary_response(self) -> None:
        """Test validation always adds primary_response."""
        engine = DecisionEngine(llm_provider=None, use_llm=False)
        
        nodes = engine._validate_node_selection([
            "sentiment_analysis",
            "memory_evaluator"
        ])
        
        assert "primary_response" in nodes
    
    def test_build_selection_prompt(self) -> None:
        """Test building selection prompt for LLM."""
        from src.llm.base import BaseLLM
        
        class MockLLM(BaseLLM):
            def generate(self, prompt: str) -> str:
                return '["sentiment_analysis", "primary_response"]'
            
            def close(self) -> None:
                pass
        
        mock_llm = MockLLM()
        engine = DecisionEngine(llm_provider=mock_llm, use_llm=True)
        broker = KnowledgeBroker()
        broker.sentiment_analysis = SentimentAnalysis(
            sentiment="positive",
            confidence=0.9,
            memory_owner="user",
            emotional_tone="happy",
            key_topics=[],
            reasoning="test"
        )
        broker.idle_time_minutes = 30
        
        prompt = engine._build_selection_prompt("Hello", broker)
        
        assert "Hello" in prompt
        assert "positive" in prompt
        assert "30" in prompt
        assert "memory_evaluator" in prompt
        assert "trust_analysis" in prompt
    
    def test_parse_llm_response_valid_json(self) -> None:
        """Test parsing valid LLM JSON response."""
        engine = DecisionEngine(llm_provider=None, use_llm=False)
        
        response = '["sentiment_analysis", "primary_response", "memory_evaluator"]'
        nodes = engine._parse_llm_response(response)
        
        assert "sentiment_analysis" in nodes
        assert "primary_response" in nodes
        assert "memory_evaluator" in nodes
    
    def test_parse_llm_response_invalid_json(self) -> None:
        """Test parsing invalid LLM JSON response."""
        engine = DecisionEngine(llm_provider=None, use_llm=False)
        
        response = "invalid json"
        nodes = engine._parse_llm_response(response)
        
        assert nodes == []
    
    def test_parse_llm_response_with_extra_text(self) -> None:
        """Test parsing LLM response with extra text."""
        engine = DecisionEngine(llm_provider=None, use_llm=False)
        
        response = 'Here are the nodes: ["sentiment_analysis", "primary_response"]'
        nodes = engine._parse_llm_response(response)
        
        assert "sentiment_analysis" in nodes
        assert "primary_response" in nodes