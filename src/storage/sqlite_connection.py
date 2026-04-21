"""SQLite connection management with proper resource cleanup."""

import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


class SQLiteConnection:
    """Manages SQLite connections with proper resource cleanup.

    This class provides a context manager for SQLite connections that ensures
    proper cleanup of resources. For :memory: databases, it maintains a
    persistent connection. For file-based databases, it creates and closes
    connections per use.
    """

    def __init__(self, db_path: str):
        """Initialize the connection manager.

        Args:
            db_path: Path to the SQLite database file, or ":memory:" for
                in-memory database
        """
        self.db_path = db_path
        self._memory_conn: sqlite3.Connection | None = None

        if db_path == ":memory:":
            self._memory_conn = sqlite3.Connection(db_path, check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection as a context manager.

        Yields:
            SQLite connection object that is properly closed on exit
            (except for :memory: databases which remain persistent)
        """
        if self._memory_conn is not None:
            yield self._memory_conn
            return

        conn = sqlite3.Connection(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def close(self) -> None:
        """Close the store and clean up resources."""
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None
            logger.debug("Memory database connection closed")

    def execute_pragma(self, pragma: str) -> None:
        """Execute a PRAGMA statement on a new connection.

        Args:
            pragma: The PRAGMA statement to execute
        """
        with self.get_connection() as conn:
            conn.execute(pragma)
            conn.commit()
