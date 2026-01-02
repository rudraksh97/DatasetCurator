import os
import tempfile
import asyncio
import pytest
from typing import AsyncIterator

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

import db
from main import app
from models import db_models  # ensure models imported
from db import Base


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_db_url(tmp_path_factory) -> str:
    db_file = tmp_path_factory.mktemp("db") / "test.db"
    return f"sqlite+aiosqlite:///{db_file}"


@pytest.fixture(scope="session", autouse=True)
def setup_test_db(test_db_url, event_loop):
    engine = create_async_engine(test_db_url, future=True)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    # patch globals for the app
    db.engine = engine
    db.AsyncSessionLocal = session_maker

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
    async with db.AsyncSessionLocal() as session:
        yield session


@pytest.fixture()
def client():
    return TestClient(app)
