"""SQLite-based conversation history storage."""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from src.config.settings import settings
from src.storage.sqlite_connection import SQLiteConnection

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """Represents a single conversation message."""

    id: int
    timestamp: str
    speaker: str
    message: str
    response: str | None

    def to_context_string(self) -> str:
        """Format as context string for LLM.

        Returns:
            Formatted string for inclusion in LLM context
        """
        if self.response:
            return f"{self.speaker}: {self.message}\nAssistant: {self.response}"
        return f"{self.speaker}: {self.message}"


class ConversationStore:
    """SQLite-based storage for conversation history.

    This provides short-term conversation context that can be quickly
    retrieved without embeddings. It's designed to complement the vector
    database by providing immediate conversational context.

    Features:
    - Store conversation messages with metadata
    - Retrieve recent messages for context (configurable limit)
    - Auto-cleanup to maintain max message count
    - Thread-safe operations
    """

    def __init__(
        self,
        connection: SQLiteConnection | None = None,
        max_messages: int | None = None,
        context_limit: int | None = None,
    ):
        """Initialize the conversation store.

        Args:
            connection: SQLiteConnection instance (creates from settings if None)
            max_messages: Maximum messages to keep before pruning (defaults to settings)
            context_limit: Default messages to retrieve for context (defaults to settings)
        """
        config = settings.conversation_store
        self.max_messages = max_messages or config.max_messages
        self.context_limit = context_limit or config.context_limit

        self._connection = connection or SQLiteConnection(config.db_path)
        self._init_db()

        logger.info(
            "Conversation store initialized (max: %s, context: %s)",
            self.max_messages,
            self.context_limit,
        )

    def _init_db(self) -> None:
        """Initialize the database schema."""
        try:
            with self._connection.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("PRAGMA foreign_keys = ON")

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        speaker TEXT NOT NULL,
                        message TEXT NOT NULL,
                        response TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp
                    ON conversations(timestamp DESC)
                """)

                conn.commit()
                logger.debug("Database schema initialized")

        except sqlite3.Error as e:
            logger.error("Error initializing database: %s", e, exc_info=True)
            raise

    def add_message(
        self,
        speaker: str,
        message: str,
        response: str | None = None,
        timestamp: str | None = None,
    ) -> int:
        """Add a conversation message to the store.

        Args:
            speaker: Name of the speaker (e.g., "User", "Character")
            message: The message content
            response: Optional response from the assistant
            timestamp: Optional timestamp (ISO format, defaults to now)

        Returns:
            ID of the inserted message

        Raises:
            sqlite3.Error: If insertion fails
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        try:
            with self._connection.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO conversations (timestamp, speaker, message, response)
                    VALUES (?, ?, ?, ?)
                """,
                    (timestamp, speaker, message, response),
                )

                message_id = cursor.lastrowid
                conn.commit()

                logger.debug("Added message %s from %s", message_id, speaker)

                self._cleanup_if_needed(conn)

                return message_id or -1

        except sqlite3.Error as e:
            logger.error("Error adding message: %s", e, exc_info=True)
            raise

    def update_response(self, message_id: int, response: str) -> None:
        """Update the response for a message.

        Args:
            message_id: ID of the message to update
            response: The response text

        Raises:
            sqlite3.Error: If update fails
        """
        try:
            with self._connection.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE conversations
                    SET response = ?
                    WHERE id = ?
                """,
                    (response, message_id),
                )

                conn.commit()
                logger.debug("Updated response for message %s", message_id)

        except sqlite3.Error as e:
            logger.error("Error updating response: %s", e, exc_info=True)
            raise

    def get_recent_for_context(
        self, limit: int | None = None
    ) -> list[ConversationMessage]:
        """Get recent messages for LLM context.

        Args:
            limit: Number of messages to retrieve (defaults to self.context_limit)

        Returns:
            List of recent messages ordered chronologically (oldest first)
        """
        if limit is None:
            limit = self.context_limit

        try:
            with self._connection.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT id, timestamp, speaker, message, response
                    FROM conversations
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (limit,),
                )

                rows = cursor.fetchall()

                messages = [
                    ConversationMessage(
                        id=row["id"],
                        timestamp=row["timestamp"],
                        speaker=row["speaker"],
                        message=row["message"],
                        response=row["response"],
                    )
                    for row in reversed(rows)
                ]

                logger.debug("Retrieved %s messages for context", len(messages))
                return messages

        except sqlite3.Error as e:
            logger.error("Error retrieving messages: %s", e, exc_info=True)
            return []

    def get_all(self, limit: int = 200) -> list[ConversationMessage]:
        """Get all messages (for debugging/review).

        Args:
            limit: Maximum number of messages to retrieve

        Returns:
            List of messages ordered chronologically (oldest first)
        """
        try:
            with self._connection.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT id, timestamp, speaker, message, response
                    FROM conversations
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (limit,),
                )

                rows = cursor.fetchall()

                return [
                    ConversationMessage(
                        id=row["id"],
                        timestamp=row["timestamp"],
                        speaker=row["speaker"],
                        message=row["message"],
                        response=row["response"],
                    )
                    for row in reversed(rows)
                ]

        except sqlite3.Error as e:
            logger.error("Error retrieving all messages: %s", e, exc_info=True)
            return []

    def get_count(self) -> int:
        """Get the total number of messages in the database.

        Returns:
            Message count
        """
        try:
            with self._connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM conversations")
                count = cursor.fetchone()[0]
                return count

        except sqlite3.Error as e:
            logger.error("Error getting count: %s", e, exc_info=True)
            return 0

    def _cleanup_if_needed(self, conn: sqlite3.Connection) -> None:
        """Clean up old messages if over max_messages limit.

        Args:
            conn: Existing database connection
        """
        try:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM conversations")
            count = cursor.fetchone()[0]

            if count > self.max_messages:
                to_delete = count - self.max_messages

                cursor.execute(
                    """
                    DELETE FROM conversations
                    WHERE id IN (
                        SELECT id FROM conversations
                        ORDER BY timestamp ASC
                        LIMIT ?
                    )
                """,
                    (to_delete,),
                )

                conn.commit()
                logger.info(
                    "Cleaned up %s old messages (keeping %s)",
                    to_delete,
                    self.max_messages,
                )

        except sqlite3.Error as e:
            logger.error("Error during cleanup: %s", e, exc_info=True)

    def cleanup_old_messages(self, max_messages: int | None = None) -> int:
        """Manually trigger cleanup of old messages.

        Args:
            max_messages: Maximum to keep (defaults to self.max_messages)

        Returns:
            Number of messages deleted
        """
        if max_messages is None:
            max_messages = self.max_messages

        try:
            with self._connection.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM conversations")
                count = cursor.fetchone()[0]

                if count <= max_messages:
                    logger.debug("No cleanup needed (%s <= %s)", count, max_messages)
                    return 0

                to_delete = count - max_messages

                cursor.execute(
                    """
                    DELETE FROM conversations
                    WHERE id IN (
                        SELECT id FROM conversations
                        ORDER BY timestamp ASC
                        LIMIT ?
                    )
                """,
                    (to_delete,),
                )

                conn.commit()
                logger.info("Cleaned up %s messages", to_delete)
                return to_delete

        except sqlite3.Error as e:
            logger.error("Error cleaning up: %s", e, exc_info=True)
            return 0

    def clear_all(self) -> None:
        """Clear all messages from the database.

        WARNING: This deletes all conversation history!
        """
        try:
            with self._connection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM conversations")
                conn.commit()
                logger.warning("All conversation messages cleared from database")

        except sqlite3.Error as e:
            logger.error("Error clearing messages: %s", e, exc_info=True)
            raise

    def format_for_llm(self, messages: list[ConversationMessage]) -> str:
        """Format messages as context string for LLM.

        Args:
            messages: List of messages to format

        Returns:
            Formatted context string
        """
        if not messages:
            return ""

        context_lines = [msg.to_context_string() for msg in messages]
        return "\n\n".join(context_lines)

    def close(self) -> None:
        """Close the store and clean up resources."""
        self._connection.close()
        logger.info("Conversation store closed")
