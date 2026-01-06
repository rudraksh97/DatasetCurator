"""Database configuration and session management.

This module provides async SQLAlchemy database configuration for PostgreSQL
with pgvector extension support for vector similarity search.

Example:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(DatasetRecord))
"""
from __future__ import annotations

from functools import lru_cache
from typing import AsyncIterator, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from config import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Create and return an async SQLAlchemy engine.
    
    Returns:
        AsyncEngine configured for PostgreSQL with asyncpg driver.
    
    Raises:
        ValueError: If DATABASE_URL is not configured.
    """
    settings.database.validate()
    return create_async_engine(settings.database.url, echo=False, future=True)


# Lazy initialization - only create when first accessed
_engine: Optional[AsyncEngine] = None
_session_maker: Optional[async_sessionmaker] = None


def _get_engine() -> AsyncEngine:
    """Get or create the database engine (lazy initialization)."""
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def _get_session_maker() -> async_sessionmaker:
    """Get or create the session maker (lazy initialization)."""
    global _session_maker
    if _session_maker is None:
        _session_maker = async_sessionmaker(_get_engine(), expire_on_commit=False)
    return _session_maker


# For backward compatibility, expose as properties
@property
def engine() -> AsyncEngine:
    """Get the database engine."""
    return _get_engine()


# Create a proxy object for backward compatibility
class _EngineProxy:
    """Proxy that lazily accesses the engine."""
    
    def begin(self):
        return _get_engine().begin()
    
    def __getattr__(self, name):
        return getattr(_get_engine(), name)


class _SessionMakerProxy:
    """Proxy that lazily accesses the session maker."""
    
    def __call__(self):
        return _get_session_maker()()
    
    def __getattr__(self, name):
        return getattr(_get_session_maker(), name)


# Backward compatible module-level variables
engine = _EngineProxy()
AsyncSessionLocal = _SessionMakerProxy()


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that provides a database session.
    
    Yields:
        AsyncSession for database operations.
    """
    async with _get_session_maker()() as session:
        yield session


async def init_pgvector() -> None:
    """Initialize the pgvector extension in PostgreSQL.
    
    This enables vector similarity search capabilities used for
    semantic search over dataset embeddings.
    """
    async with _get_engine().begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.commit()
