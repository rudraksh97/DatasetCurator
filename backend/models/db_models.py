"""SQLAlchemy ORM models for database persistence.

This module defines the database schema for:
- DatasetRecord: Stores dataset metadata and state
- DatasetEmbedding: Stores vector embeddings for semantic search

Note: EMBEDDING_DIM is loaded from the centralized config module.
"""
from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, String, JSON, Text, func
from pgvector.sqlalchemy import Vector

from config import settings
from db import Base


class DatasetRecord(Base):
    """SQLAlchemy model for dataset state persistence.
    
    Stores all metadata and state for datasets being curated,
    including paths, schema, transformation history, and chat history.
    
    Attributes:
        dataset_id: Primary key, unique identifier for the dataset.
        current_version: Version number incremented on each transformation.
        raw_path: Path to the original uploaded CSV file.
        curated_path: Path to the latest curated version.
        schema: JSON storing column names and data types.
        transformation_log: JSON array of applied transformations.
        chat_history: JSON array of chat messages.
        created_at: Timestamp when the dataset was first uploaded.
        updated_at: Timestamp of the last modification.
    """
    
    __tablename__ = "dataset_states"

    dataset_id = Column(String, primary_key=True, index=True)
    current_version = Column(Integer, default=0)
    raw_path = Column(String, nullable=True)
    curated_path = Column(String, nullable=True)
    schema = Column(JSON, nullable=True)
    transformation_log = Column(JSON, default=list)
    chat_history = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class DatasetEmbedding(Base):
    """Store embeddings for semantic search over dataset rows.
    
    Each row from a dataset can have an embedding generated from
    its text content, enabling semantic similarity search.
    
    Attributes:
        id: Auto-incrementing primary key.
        dataset_id: Foreign reference to the dataset.
        row_index: Original row index in the CSV file.
        content: The text that was embedded.
        embedding: Vector from sentence-transformers (dimension from config).
        metadata_: Additional row data stored as JSON.
        created_at: Timestamp when the embedding was created.
    """
    
    __tablename__ = "dataset_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String, index=True, nullable=False)
    row_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.embedding.dimension), nullable=False)
    metadata_ = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
