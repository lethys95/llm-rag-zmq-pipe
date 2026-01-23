"""Unit tests for conversation storage."""

from datetime import datetime

import pytest

from src.storage.conversation_store import ConversationStore, ConversationMessage


@pytest.mark.unit
class TestConversationMessage:
    """Tests for ConversationMessage dataclass."""
    
    def test_create_message(self):
        """Test creating a conversation message."""
        msg = ConversationMessage(
            id=1,
            timestamp="2024-01-01T12:00:00",
            speaker="User",
            message="Hello",
            response="Hi there!"
        )
        
        assert msg.id == 1
        assert msg.timestamp == "2024-01-01T12:00:00"
        assert msg.speaker == "User"
        assert msg.message == "Hello"
        assert msg.response == "Hi there!"
        
    def test_to_context_string_with_response(self):
        """Test formatting message with response to context string."""
        msg = ConversationMessage(
            id=1,
            timestamp="2024-01-01T12:00:00",
            speaker="User",
            message="How are you?",
            response="I'm doing well, thanks!"
        )
        
        context = msg.to_context_string()
        
        assert "User: How are you?" in context
        assert "Assistant: I'm doing well, thanks!" in context
        
    def test_to_context_string_without_response(self):
        """Test formatting message without response to context string."""
        msg = ConversationMessage(
            id=1,
            timestamp="2024-01-01T12:00:00",
            speaker="User",
            message="What's the weather?",
            response=None
        )
        
        context = msg.to_context_string()
        
        assert context == "User: What's the weather?"
        assert "Assistant" not in context


@pytest.mark.unit
class TestConversationStore:
    """Tests for ConversationStore class."""
    
    def test_init_with_memory_db(self):
        """Test initializing store with in-memory database."""
        store = ConversationStore(db_path=":memory:")
        
        assert store.db_path == ":memory:"
        assert store.max_messages == 200
        assert store.context_limit == 15
        
    def test_init_with_custom_config(self):
        """Test initializing store with custom configuration."""
        store = ConversationStore(
            db_path=":memory:",
            max_messages=500,
            context_limit=25
        )
        
        assert store.max_messages == 500
        assert store.context_limit == 25
        
    def test_add_message(self):
        """Test adding a message to the store."""
        store = ConversationStore(db_path=":memory:")
        
        msg_id = store.add_message(
            speaker="User",
            message="Hello, world!",
            response="Hi there!"
        )
        
        assert msg_id == 1
        assert store.get_count() == 1
        
    def test_add_message_with_custom_timestamp(self):
        """Test adding a message with custom timestamp."""
        store = ConversationStore(db_path=":memory:")
        timestamp = "2024-01-01T12:00:00"
        
        store.add_message(
            speaker="User",
            message="Test",
            timestamp=timestamp
        )
        
        messages = store.get_all()
        assert len(messages) == 1
        assert messages[0].timestamp == timestamp
        
    def test_add_message_without_response(self):
        """Test adding a message without response."""
        store = ConversationStore(db_path=":memory:")
        
        msg_id = store.add_message(
            speaker="User",
            message="Question?"
        )
        
        assert msg_id == 1
        messages = store.get_all()
        assert messages[0].response is None
        
    def test_update_response(self):
        """Test updating response for a message."""
        store = ConversationStore(db_path=":memory:")
        
        msg_id = store.add_message(
            speaker="User",
            message="Question?",
            response=None
        )
        
        store.update_response(msg_id, "Answer!")
        
        messages = store.get_all()
        assert messages[0].response == "Answer!"
        
    def test_get_recent_for_context(self):
        """Test retrieving recent messages for context."""
        store = ConversationStore(db_path=":memory:", context_limit=3)
        
        # Add some messages
        for i in range(5):
            store.add_message(
                speaker="User",
                message=f"Message {i}",
                response=f"Response {i}"
            )
        
        recent = store.get_recent_for_context()
        
        # Should get 3 most recent (default context_limit)
        assert len(recent) == 3
        # Should be in chronological order (oldest first)
        assert recent[0].message == "Message 2"
        assert recent[1].message == "Message 3"
        assert recent[2].message == "Message 4"
        
    def test_get_recent_with_custom_limit(self):
        """Test retrieving recent messages with custom limit."""
        store = ConversationStore(db_path=":memory:")
        
        for i in range(10):
            store.add_message(speaker="User", message=f"Message {i}")
        
        recent = store.get_recent_for_context(limit=5)
        
        assert len(recent) == 5
        assert recent[0].message == "Message 5"
        assert recent[4].message == "Message 9"
        
    def test_get_all(self):
        """Test retrieving all messages."""
        store = ConversationStore(db_path=":memory:")
        
        for i in range(5):
            store.add_message(speaker="User", message=f"Message {i}")
        
        all_messages = store.get_all()
        
        assert len(all_messages) == 5
        # Should be in chronological order
        assert all_messages[0].message == "Message 0"
        assert all_messages[4].message == "Message 4"
        
    def test_get_all_with_limit(self):
        """Test retrieving all messages with limit."""
        store = ConversationStore(db_path=":memory:")
        
        for i in range(10):
            store.add_message(speaker="User", message=f"Message {i}")
        
        messages = store.get_all(limit=3)
        
        assert len(messages) == 3
        # Should get 3 most recent
        assert messages[0].message == "Message 7"
        
    def test_get_count(self):
        """Test getting message count."""
        store = ConversationStore(db_path=":memory:")
        
        assert store.get_count() == 0
        
        store.add_message(speaker="User", message="Message 1")
        assert store.get_count() == 1
        
        store.add_message(speaker="User", message="Message 2")
        assert store.get_count() == 2
        
    def test_auto_cleanup(self):
        """Test automatic cleanup when exceeding max_messages."""
        store = ConversationStore(db_path=":memory:", max_messages=5)
        
        # Add more messages than max_messages
        for i in range(7):
            store.add_message(speaker="User", message=f"Message {i}")
        
        # Should have cleaned up to max_messages
        assert store.get_count() == 5
        
        # Should have kept the most recent
        messages = store.get_all()
        assert messages[0].message == "Message 2"
        assert messages[-1].message == "Message 6"
        
    def test_manual_cleanup(self):
        """Test manual cleanup of old messages."""
        store = ConversationStore(db_path=":memory:")
        
        for i in range(10):
            store.add_message(speaker="User", message=f"Message {i}")
        
        deleted = store.cleanup_old_messages(max_messages=5)
        
        assert deleted == 5
        assert store.get_count() == 5
        
        messages = store.get_all()
        assert messages[0].message == "Message 5"
        
    def test_cleanup_when_not_needed(self):
        """Test cleanup when message count is below threshold."""
        store = ConversationStore(db_path=":memory:")
        
        for i in range(3):
            store.add_message(speaker="User", message=f"Message {i}")
        
        deleted = store.cleanup_old_messages(max_messages=10)
        
        assert deleted == 0
        assert store.get_count() == 3
        
    def test_clear_all(self):
        """Test clearing all messages."""
        store = ConversationStore(db_path=":memory:")
        
        for i in range(5):
            store.add_message(speaker="User", message=f"Message {i}")
        
        assert store.get_count() == 5
        
        store.clear_all()
        
        assert store.get_count() == 0
        
    def test_format_for_llm(self):
        """Test formatting messages for LLM context."""
        store = ConversationStore(db_path=":memory:")
        
        messages = [
            ConversationMessage(
                id=1,
                timestamp="2024-01-01T12:00:00",
                speaker="User",
                message="Hello",
                response="Hi!"
            ),
            ConversationMessage(
                id=2,
                timestamp="2024-01-01T12:01:00",
                speaker="User",
                message="How are you?",
                response="I'm good!"
            ),
        ]
        
        context = store.format_for_llm(messages)
        
        assert "User: Hello" in context
        assert "Assistant: Hi!" in context
        assert "User: How are you?" in context
        assert "Assistant: I'm good!" in context
        
    def test_format_for_llm_empty_list(self):
        """Test formatting empty message list."""
        store = ConversationStore(db_path=":memory:")
        
        context = store.format_for_llm([])
        
        assert context == ""
        
    def test_close(self):
        """Test closing the store."""
        store = ConversationStore(db_path=":memory:")
        
        # Should not raise any exceptions
        store.close()
