"""
Database connection management for the Blockchain Analysis project.

Provides connection pooling, context managers, and utility functions
for PostgreSQL database operations.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

import psycopg2
from psycopg2.extensions import connection as PgConnection

from .config import Config

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass


class DatabaseManager:
    """
    Manages PostgreSQL database connections with context manager support.
    
    Usage:
        # As context manager (recommended)
        with DatabaseManager() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM tx_basic LIMIT 10")
                results = cur.fetchall()
        
        # Manual management
        db = DatabaseManager()
        conn = db.connect()
        try:
            # ... operations
        finally:
            db.close()
    """
    
    def __init__(self, autocommit: bool = False):
        """
        Initialize database manager.
        
        Args:
            autocommit: If True, enable autocommit mode for the connection.
        """
        self._config = Config().database
        self._conn: Optional[PgConnection] = None
        self._autocommit = autocommit
    
    def connect(self) -> PgConnection:
        """
        Establish a database connection.
        
        Returns:
            psycopg2 connection object.
            
        Raises:
            DatabaseError: If connection fails.
        """
        try:
            self._conn = psycopg2.connect(**self._config.to_dict())
            self._conn.autocommit = self._autocommit
            logger.debug("Database connection established")
            return self._conn
        except psycopg2.Error as exc:
            logger.error("Failed to connect to database: %s", exc)
            raise DatabaseError(f"Database connection failed: {exc}") from exc
    
    def close(self) -> None:
        """Close the database connection if open."""
        if self._conn is not None:
            try:
                self._conn.close()
                logger.debug("Database connection closed")
            except psycopg2.Error as exc:
                logger.warning("Error closing database connection: %s", exc)
            finally:
                self._conn = None
    
    @property
    def connection(self) -> Optional[PgConnection]:
        """Returns the current connection or None."""
        return self._conn
    
    @property
    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._conn is not None and not self._conn.closed
    
    def __enter__(self) -> PgConnection:
        """Context manager entry."""
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic rollback on exception."""
        if self._conn is not None:
            if exc_type is not None and not self._autocommit:
                try:
                    self._conn.rollback()
                    logger.debug("Transaction rolled back due to exception")
                except psycopg2.Error:
                    pass
            self.close()


@contextmanager
def get_db_connection(autocommit: bool = False) -> Generator[PgConnection, None, None]:
    """
    Context manager for database connections.
    
    This is a convenience function that creates a DatabaseManager
    and yields its connection.
    
    Args:
        autocommit: If True, enable autocommit mode.
        
    Yields:
        psycopg2 connection object.
        
    Example:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM tx_basic")
                count = cur.fetchone()[0]
    """
    db = DatabaseManager(autocommit=autocommit)
    try:
        yield db.connect()
    except Exception:
        if db.is_connected and not autocommit:
            try:
                db.connection.rollback()
            except psycopg2.Error:
                pass
        raise
    finally:
        db.close()


def execute_query(query: str, params: tuple = None, fetch: bool = True) -> Optional[list]:
    """
    Execute a query and optionally fetch results.
    
    Args:
        query: SQL query string.
        params: Query parameters (optional).
        fetch: If True, fetch and return results.
        
    Returns:
        List of results if fetch=True, None otherwise.
        
    Example:
        results = execute_query("SELECT * FROM tx_basic WHERE fee > %s LIMIT 10", (1000,))
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if fetch:
                return cur.fetchall()
            conn.commit()
            return None
