"""Pytest configuration and fixtures for API tests.

This module provides fixtures for:
- Database setup with SQLite for testing
- FastAPI test client
- Async session management
"""
import asyncio
import os
from typing import AsyncIterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# Set test database URL before importing db module
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import db
from main import app
from models import db_models  # noqa: F401 - ensure models imported
from db import Base


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_db_url(tmp_path_factory) -> str:
    """Generate a temporary SQLite database URL for testing."""
    db_file = tmp_path_factory.mktemp("db") / "test.db"
    return f"sqlite+aiosqlite:///{db_file}"


@pytest.fixture(scope="session", autouse=True)
def setup_test_db(test_db_url, event_loop):
    """Set up the test database with tables created."""
    engine = create_async_engine(test_db_url, future=True)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    # Patch globals for the app
    db._engine = engine
    db._session_maker = session_maker

    async def init_models():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    event_loop.run_until_complete(init_models())
    yield
    
    async def drop_models():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()
    
    event_loop.run_until_complete(drop_models())


@pytest.fixture()
async def session() -> AsyncIterator[AsyncSession]:
    """Provide an async database session for tests."""
    async with db._get_session_maker()() as session:
        yield session


@pytest.fixture()
def client():
    """Provide a FastAPI test client."""
    return TestClient(app)
