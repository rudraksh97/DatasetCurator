"""Database configuration and session management.

This module provides async SQLAlchemy database configuration for PostgreSQL
with pgvector extension support for vector similarity search.

Example:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(DatasetRecord))
"""
from __future__ import annotations

import os
from typing import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://dataset_curator:dataset_curator@localhost:5432/dataset_curator",
)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


def get_engine() -> AsyncEngine:
    """Create and return an async SQLAlchemy engine.
    
    Returns:
        AsyncEngine configured for PostgreSQL with asyncpg driver.
    """
    return create_async_engine(DATABASE_URL, echo=False, future=True)


engine = get_engine()
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that provides a database session.
    
    Yields:
        AsyncSession for database operations.
    """
    async with AsyncSessionLocal() as session:
        yield session


async def init_pgvector() -> None:
    """Initialize the pgvector extension in PostgreSQL.
    
    This enables vector similarity search capabilities used for
    semantic search over dataset embeddings.
    """
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.commit()
