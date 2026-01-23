"""Tests for Phase 1 node system components."""

import asyncio
import pytest
from src.nodes import (
    BaseNode,
    NodeResult,
    NodeStatus,
    KnowledgeBroker,
    TaskQueueManager,
    DecisionEngine,
    NodeRegistry,
)


# Mock nodes for testing
class MockSentimentNode(BaseNode):
    """Mock sentiment analysis node for testing."""
    
    def __init__(self):
        super().__init__(
            name="sentiment_analysis",
            priority=1,
            queue_type="immediate"
        )
    
    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        """Mock sentiment analysis."""
        message = broker.get_knowledge("user_message", "")
        
        # Simple mock analysis
        sentiment = "positive" if "good" in message.lower() else "neutral"
        
        return NodeResult(
            status=NodeStatus.SUCCESS,
            data={
                "sentiment": sentiment,
                "confidence": 0.85
            }
        )


class MockPrimaryResponseNode(BaseNode):
    """Mock primary response node for testing."""
    
    def __init__(self):
        super().__init__(
            name="primary_response",
            priority=3,
            dependencies=["sentiment_analysis"],
            queue_type="immediate"
        )
    
    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        """Mock response generation."""
        message = broker.get_knowledge("user_message", "")
        sentiment = broker.get_knowledge("sentiment", "neutral")
        
        response = f"I understand you said: '{message}'. Sentiment: {sentiment}"
        
        return NodeResult(
            status=NodeStatus.SUCCESS,
            data={
                "response": response
            }
        )


class ConditionalNode(BaseNode):
    """Node that only runs under certain conditions."""
    
    def __init__(self):
        super().__init__(
            name="conditional_test",
            priority=5,
            queue_type="background"
        )
    
    def should_run(self, broker: KnowledgeBroker) -> bool:
        """Only run if special flag is set."""
        return broker.get_knowledge("run_conditional", False)
    
    async def execute(self, broker: KnowledgeBroker) -> NodeResult:
        """Execute conditional logic."""
        return NodeResult(
            status=NodeStatus.SUCCESS,
            data={"conditional_ran": True}
        )


# Tests for KnowledgeBroker
def test_knowledge_broker_add_get():
    """Test adding and retrieving knowledge."""
    broker = KnowledgeBroker()
    
    broker.add_knowledge("test_key", "test_value")
    assert broker.get_knowledge("test_key") == "test_value"
    assert broker.has_knowledge("test_key") is True
    assert broker.has_knowledge("nonexistent") is False


def test_knowledge_broker_full_context():
    """Test getting full context."""
    broker = KnowledgeBroker()
    
    broker.add_knowledge("key1", "value1")
    broker.add_knowledge("key2", "value2")
    
    context = broker.get_full_context()
    assert context["key1"] == "value1"
    assert context["key2"] == "value2"


def test_knowledge_broker_clear():
    """Test clearing the broker."""
    broker = KnowledgeBroker()
    
    broker.add_knowledge("test", "data")
    assert broker.has_knowledge("test")
    
    broker.clear()
    assert not broker.has_knowledge("test")


# Tests for BaseNode
def test_base_node_dependencies():
    """Test dependency validation."""
    node = MockPrimaryResponseNode()
    
    # Dependencies not met
    assert node.validate_dependencies(set()) is False
    assert node.validate_dependencies({"other_node"}) is False
    
    # Dependencies met
    assert node.validate_dependencies({"sentiment_analysis"}) is True
    assert node.validate_dependencies({"sentiment_analysis", "extra"}) is True


def test_base_node_priority_comparison():
    """Test node priority ordering."""
    node1 = MockSentimentNode()  # priority 1
    node2 = MockPrimaryResponseNode()  # priority 3
    
    # Lower priority number should be "less than"
    assert node1 < node2
    assert not (node2 < node1)


# Tests for NodeRegistry
def test_node_registry_singleton():
    """Test registry singleton pattern."""
    registry1 = NodeRegistry.get_instance()
    registry2 = NodeRegistry.get_instance()
    
    assert registry1 is registry2


def test_node_registry_register_create():
    """Test registering and creating nodes."""
    registry = NodeRegistry()
    
    registry.register(MockSentimentNode, "test_sentiment")
    
    assert registry.is_registered("test_sentiment")
    
    node = registry.create("test_sentiment")
    assert isinstance(node, MockSentimentNode)
    assert node.name == "sentiment_analysis"


def test_node_registry_list_available():
    """Test listing available nodes."""
    registry = NodeRegistry()
    registry.clear()
    
    registry.register(MockSentimentNode, "sentiment")
    registry.register(MockPrimaryResponseNode, "response")
    
    available = registry.list_available()
    assert "sentiment" in available
    assert "response" in available
    assert len(available) == 2


# Tests for DecisionEngine
def test_decision_engine_rule_based():
    """Test rule-based node selection."""
    engine = DecisionEngine()
    broker = KnowledgeBroker()
    
    # Normal message
    nodes = engine._rule_based_selection("Hello, how are you?", broker)
    assert "sentiment_analysis" in nodes
    assert "primary_response" in nodes


def test_decision_engine_crisis_detection():
    """Test crisis keyword detection."""
    engine = DecisionEngine()
    broker = KnowledgeBroker()
    
    # Crisis message
    nodes = engine._rule_based_selection("I want to kill myself", broker)
    assert "crisis_detection" in nodes


def test_decision_engine_idle_detection():
    """Test idle time detection for detox."""
    engine = DecisionEngine()
    broker = KnowledgeBroker()
    
    # Set idle time
    broker.add_knowledge("idle_time_minutes", 90)
    
    nodes = engine._rule_based_selection("Hi there", broker)
    assert "detox_protocol" in nodes


# Tests for TaskQueueManager
@pytest.mark.asyncio
async def test_queue_manager_enqueue():
    """Test enqueueing nodes."""
    manager = TaskQueueManager()
    node = MockSentimentNode()
    
    await manager.enqueue(node)
    
    assert manager.immediate_queue.qsize() == 1


@pytest.mark.asyncio
async def test_queue_manager_execute_simple():
    """Test executing a simple node."""
    manager = TaskQueueManager()
    broker = KnowledgeBroker()
    
    # Setup
    broker.add_knowledge("user_message", "This is good news!")
    
    # Enqueue node
    node = MockSentimentNode()
    await manager.enqueue(node)
    
    # Execute
    await manager.execute_immediate(broker)
    
    # Check results
    assert "sentiment_analysis" in manager.completed_nodes
    assert broker.get_knowledge("sentiment") == "positive"


@pytest.mark.asyncio
async def test_queue_manager_dependencies():
    """Test dependency resolution."""
    manager = TaskQueueManager()
    broker = KnowledgeBroker()
    
    broker.add_knowledge("user_message", "Hello")
    
    # Enqueue in reverse priority order to test dependency handling
    response_node = MockPrimaryResponseNode()  # depends on sentiment
    sentiment_node = MockSentimentNode()
    
    await manager.enqueue(response_node)
    await manager.enqueue(sentiment_node)
    
    # Execute
    await manager.execute_immediate(broker)
    
    # Both should complete
    assert "sentiment_analysis" in manager.completed_nodes
    assert "primary_response" in manager.completed_nodes
    
    # Response should have access to sentiment data
    response = broker.get_knowledge("response")
    assert response is not None
    assert "sentiment" in response.lower()


@pytest.mark.asyncio
async def test_queue_manager_conditional_skip():
    """Test conditional node execution."""
    manager = TaskQueueManager()
    broker = KnowledgeBroker()
    
    # Don't set the condition flag
    node = ConditionalNode()
    await manager.enqueue(node)
    
    await manager.execute_background(broker)
    
    # Node should have been skipped
    assert "conditional_test" not in manager.completed_nodes
    assert broker.get_knowledge("conditional_ran") is None


@pytest.mark.asyncio
async def test_queue_manager_conditional_run():
    """Test conditional node that should run."""
    manager = TaskQueueManager()
    broker = KnowledgeBroker()
    
    # Set the condition flag
    broker.add_knowledge("run_conditional", True)
    
    node = ConditionalNode()
    await manager.enqueue(node)
    
    await manager.execute_background(broker)
    
    # Node should have executed
    assert "conditional_test" in manager.completed_nodes
    assert broker.get_knowledge("conditional_ran") is True


# Integration test
@pytest.mark.asyncio
async def test_full_integration():
    """Test full node execution flow."""
    # Setup components
    broker = KnowledgeBroker()
    manager = TaskQueueManager()
    engine = DecisionEngine()
    
    # Simulate user message
    message = "I'm feeling good today!"
    broker.add_knowledge("user_message", message)
    broker.add_knowledge("speaker", "user")
    
    # Decision engine selects nodes
    node_names = await engine.select_nodes(message, broker)
    
    # Create and enqueue nodes (mock implementation)
    if "sentiment_analysis" in node_names:
        await manager.enqueue(MockSentimentNode())
    if "primary_response" in node_names:
        await manager.enqueue(MockPrimaryResponseNode())
    
    # Execute
    await manager.execute_immediate(broker)
    
    # Verify results
    assert broker.get_knowledge("sentiment") == "positive"
    assert broker.get_knowledge("response") is not None
    
    # Check execution summary
    summary = broker.get_execution_summary()
    assert summary["metadata"]["total_nodes_executed"] >= 2
    assert len(summary["execution_order"]) >= 2


if __name__ == "__main__":
    print("Running Phase 1 Node System Tests...")
    print("\nNote: Run with pytest for full async support:")
    print("  pytest tests/test_nodes_phase1.py -v\n")
    
    # Run some simple sync tests
    print("Testing KnowledgeBroker...")
    test_knowledge_broker_add_get()
    test_knowledge_broker_full_context()
    test_knowledge_broker_clear()
    print("✓ KnowledgeBroker tests passed")
    
    print("\nTesting BaseNode...")
    test_base_node_dependencies()
    test_base_node_priority_comparison()
    print("✓ BaseNode tests passed")
    
    print("\nTesting NodeRegistry...")
    test_node_registry_singleton()
    test_node_registry_register_create()
    test_node_registry_list_available()
    print("✓ NodeRegistry tests passed")
    
    print("\nTesting DecisionEngine...")
    test_decision_engine_rule_based()
    test_decision_engine_crisis_detection()
    test_decision_engine_idle_detection()
    print("✓ DecisionEngine tests passed")
    
    print("\nFor async tests (TaskQueueManager, integration), run with pytest")
    print("\n✅ All synchronous tests passed!")
