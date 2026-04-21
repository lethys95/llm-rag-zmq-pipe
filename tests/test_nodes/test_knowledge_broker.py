"""Tests for KnowledgeBroker data flow."""

import pytest
from src.nodes.orchestration.knowledge_broker import KnowledgeBroker
from src.models.sentiment import DialogueInput, SentimentAnalysis


@pytest.fixture
def empty_broker():
    return KnowledgeBroker()


def test_broker_initialises_with_no_fields(empty_broker):
    assert empty_broker.dialogue_input is None
    assert empty_broker.sentiment_analysis is None
    assert empty_broker.primary_response is None
    assert empty_broker.zmq_identity is None
    assert empty_broker.retrieved_documents == []


def test_broker_accepts_dialogue_input(dialogue_input):
    broker = KnowledgeBroker(dialogue_input=dialogue_input)
    assert broker.dialogue_input.content == "I feel really alone today."
    assert broker.dialogue_input.speaker == "user"


def test_record_node_execution_tracks_order(empty_broker):
    empty_broker.record_node_execution("NodeA", "success", 0.1)
    empty_broker.record_node_execution("NodeB", "success", 0.2)
    assert empty_broker.metadata.execution_order == ["NodeA", "NodeB"]


def test_record_node_execution_counts_total(empty_broker):
    empty_broker.record_node_execution("NodeA", "success")
    empty_broker.record_node_execution("NodeB", "success")
    assert empty_broker.metadata.total_nodes_executed == 2


def test_record_node_execution_tracks_failures(empty_broker):
    empty_broker.record_node_execution("NodeA", "failed")
    assert "NodeA" in empty_broker.metadata.failed_nodes


def test_record_node_execution_tracks_skips(empty_broker):
    empty_broker.record_node_execution("NodeA", "skipped")
    assert "NodeA" in empty_broker.metadata.skipped_nodes


def test_get_execution_summary_structure(empty_broker):
    empty_broker.record_node_execution("NodeA", "success")
    empty_broker.record_node_execution("NodeB", "failed")

    summary = empty_broker.get_execution_summary()
    assert summary["total_nodes_executed"] == 2
    assert summary["execution_order"] == ["NodeA", "NodeB"]
    assert summary["failed_nodes"] == ["NodeB"]
    assert summary["skipped_nodes"] == []


def test_get_analyzed_context_excludes_none_fields(empty_broker):
    context = empty_broker.get_analyzed_context()
    assert "sentiment" not in context
    assert "needs_analysis" not in context


def test_get_analyzed_context_includes_populated_fields(broker):
    sentiment = SentimentAnalysis(
        sentiment="negative",
        confidence=0.9,
        memory_owner="user",
    )
    broker.sentiment_analysis = sentiment
    context = broker.get_analyzed_context()
    assert "sentiment" in context
    assert context["sentiment"] is sentiment
