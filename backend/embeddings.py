"""Vector embeddings for semantic search using pgvector and sentence-transformers."""
from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from models.db_models import DatasetEmbedding

# Embedding model configuration (configurable via environment variables)
# Default: all-MiniLM-L6-v2 - fast, 384 dimensions, good quality
# Other options: all-mpnet-base-v2 (768d), paraphrase-MiniLM-L6-v2 (384d)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))


@lru_cache(maxsize=1)
def _get_model():
    """Lazily load the sentence transformer model (cached)."""
    from sentence_transformers import SentenceTransformer
    print(f"[Embeddings] Loading model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


def _generate_embedding_sync(text: str) -> List[float]:
    """Generate embedding for a single text (synchronous)."""
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIM
    
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def _generate_embeddings_batch_sync(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts (synchronous)."""
    if not texts:
        return []
    
    # Clean texts
    cleaned = [t if t and t.strip() else "empty" for t in texts]
    
    model = _get_model()
    embeddings = model.encode(cleaned, convert_to_numpy=True, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


async def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a single text (async wrapper)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _generate_embedding_sync, text)


async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts in batches (async wrapper)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _generate_embeddings_batch_sync, texts)


def _create_row_text(row: Dict[str, Any], text_columns: List[str]) -> str:
    """Create searchable text from a row's key columns."""
    parts = []
    for col in text_columns:
        if col in row and row[col]:
            value = str(row[col])
            if value and value.lower() != "nan":
                parts.append(f"{col}: {value}")
    return " | ".join(parts)


async def embed_dataset(
    session: AsyncSession,
    dataset_id: str,
    df: pd.DataFrame,
    text_columns: Optional[List[str]] = None,
) -> int:
    """Embed a dataset and store in pgvector.
    
    Args:
        session: Database session
        dataset_id: ID of the dataset
        df: DataFrame to embed
        text_columns: Columns to use for embedding text (auto-detected if None)
    
    Returns:
        Number of rows embedded
    """
    # Auto-detect text columns if not specified
    if not text_columns:
        text_columns = []
        for col in df.columns:
            # Include columns that look like they contain searchable text
            if df[col].dtype == "object":  # String columns
                sample = df[col].dropna().head(5).tolist()
                if sample and any(len(str(s)) > 10 for s in sample):
                    text_columns.append(col)
        
        # Prioritize common text column names
        priority_cols = ["title", "name", "text", "description", "question", "content"]
        text_columns = sorted(
            text_columns,
            key=lambda x: next((i for i, p in enumerate(priority_cols) if p in x.lower()), 999)
        )[:5]  # Limit to top 5 text columns
    
    if not text_columns:
        print(f"[Embeddings] No text columns found for dataset {dataset_id}")
        return 0
    
    print(f"[Embeddings] Embedding dataset {dataset_id} using columns: {text_columns}")
    
    # Delete existing embeddings for this dataset
    await session.execute(
        delete(DatasetEmbedding).where(DatasetEmbedding.dataset_id == dataset_id)
    )
    
    # Create text for each row
    rows_data = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        text = _create_row_text(row_dict, text_columns)
        if text:
            rows_data.append({
                "row_index": int(idx),
                "content": text,
                "metadata": {col: row_dict.get(col) for col in text_columns[:3]}  # Store key fields
            })
    
    if not rows_data:
        return 0
    
    # Generate embeddings in batches
    texts = [r["content"] for r in rows_data]
    print(f"[Embeddings] Generating {len(texts)} embeddings...")
    embeddings = await generate_embeddings_batch(texts)
    
    # Store in database
    for row_data, embedding in zip(rows_data, embeddings):
        db_embedding = DatasetEmbedding(
            dataset_id=dataset_id,
            row_index=row_data["row_index"],
            content=row_data["content"],
            embedding=embedding,
            metadata_=row_data["metadata"],
        )
        session.add(db_embedding)
    
    await session.commit()
    print(f"[Embeddings] Stored {len(rows_data)} embeddings for dataset {dataset_id}")
    return len(rows_data)


async def semantic_search(
    session: AsyncSession,
    dataset_id: str,
    query: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Search for similar rows using semantic similarity.
    
    Args:
        session: Database session
        dataset_id: ID of the dataset to search
        query: Search query
        limit: Maximum number of results
    
    Returns:
        List of matching rows with similarity scores
    """
    # Generate query embedding
    query_embedding = await generate_embedding(query)
    
    # Format embedding as PostgreSQL array string for pgvector
    # pgvector expects array format: '[0.1, 0.2, ...]'
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    
    # Search using cosine similarity
    # pgvector uses <=> for cosine distance (lower is more similar)
    result = await session.execute(
        text("""
            SELECT 
                row_index,
                content,
                metadata,
                1 - (embedding <=> CAST(:query_embedding AS vector)) as similarity
            FROM dataset_embeddings
            WHERE dataset_id = :dataset_id
            ORDER BY embedding <=> CAST(:query_embedding AS vector)
            LIMIT :limit
        """),
        {
            "dataset_id": dataset_id,
            "query_embedding": embedding_str,
            "limit": limit,
        }
    )
    
    rows = result.fetchall()
    return [
        {
            "row_index": row.row_index,
            "content": row.content,
            "metadata": row.metadata,
            "similarity": float(row.similarity),
        }
        for row in rows
    ]


async def has_embeddings(session: AsyncSession, dataset_id: str) -> bool:
    """Check if a dataset has embeddings stored."""
    result = await session.execute(
        select(DatasetEmbedding.id)
        .where(DatasetEmbedding.dataset_id == dataset_id)
        .limit(1)
    )
    return result.scalar() is not None
