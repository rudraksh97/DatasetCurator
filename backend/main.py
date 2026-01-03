from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # Load .env file before other imports

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router
from db import Base, engine, init_pgvector
from models import db_models  # noqa: F401  # ensure models are imported for metadata

app = FastAPI(title="Agentic Dataset Curator")

allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.on_event("startup")
async def startup() -> None:
    # Initialize pgvector extension
    await init_pgvector()
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.get("/healthcheck")
async def healthcheck() -> dict:
    return {"status": "ok"}


# To run locally:
# uvicorn main:app --reload

