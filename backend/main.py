"""FastAPI application entry point for the Dataset Curator API.

This module initializes the FastAPI application, configures CORS middleware,
and sets up database connections on startup.

To run locally:
    uvicorn main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router
from config import settings
from db import Base, engine, init_pgvector
from models import db_models  # noqa: F401  # ensure models are imported for metadata


app = FastAPI(
    title="Agentic Dataset Curator",
    description="AI-powered dataset curation and transformation API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.on_event("startup")
async def startup() -> None:
    """Initialize database and extensions on application startup."""
    await init_pgvector()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.get("/healthcheck")
async def healthcheck() -> dict:
    """Health check endpoint.
    
    Returns:
        Dict with status "ok" if the service is running.
    """
    return {"status": "ok"}
