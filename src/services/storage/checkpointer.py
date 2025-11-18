"""
Checkpointer Module

Provides SQLite-based checkpointing for LangGraph workflows.
Supports both async and sync checkpointer implementations.
"""
import logging
import sqlite3
import aiosqlite
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from config.settings import settings

logger = logging.getLogger(__name__)
_connection = None
_checkpointer = None


async def get_sqlite_connection() -> aiosqlite.Connection:
    """
    Get the SQLite connection used by the checkpointer.
    
    Returns:
        The aiosqlite.Connection instance used for checkpoint storage.
    """
    global _connection
    if _connection is None:
        db_path = settings.SQLITE_DB_PATH
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _connection = await aiosqlite.connect(str(db_path))
        logger.info(f"SQLite connection established: {db_path}")
    return _connection


async def get_checkpointer() -> AsyncSqliteSaver:
    """
    Get checkpointer asynchronously.
    
    This function returns a singleton AsyncSqliteSaver instance that shares
    the same SQLite connection, ensuring no concurrent database access issues.
    
    Returns:
        AsyncSqliteSaver instance ready for use with LangGraph workflows.
    """
    global _checkpointer
    if _checkpointer is None:
        connection = await get_sqlite_connection()
        _checkpointer = AsyncSqliteSaver(connection)
        logger.info(f"AsyncSqliteSaver initialized and cached")
    return _checkpointer


def get_checkpointer_sync() -> SqliteSaver:
    """
    Get checkpointer synchronously for use in non-async contexts.
    
    This function uses SqliteSaver (synchronous) instead of AsyncSqliteSaver,
    which is the correct approach for synchronous contexts. Using AsyncSqliteSaver
    with asyncio.run() would create a temporary event loop that closes immediately,
    leaving the checkpointer with an invalid connection.
    
    Use this when you cannot use the async version (e.g., during module initialization).
    
    Returns:
        SqliteSaver instance ready for use with LangGraph workflows.
    """
    db_path = settings.SQLITE_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use synchronous SqliteSaver for synchronous contexts
    # Create a synchronous SQLite connection and pass it to SqliteSaver
    # This creates a proper connection that persists throughout the application lifecycle
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    logger.info(f"SqliteSaver initialized: {db_path}")
    return checkpointer
