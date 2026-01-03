from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, String, JSON, Text, func
from pgvector.sqlalchemy import Vector

from db import Base


class DatasetRecord(Base):
    """SQLAlchemy model for dataset state persistence."""
    
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
    """Store embeddings for semantic search over dataset rows."""
    
    __tablename__ = "dataset_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String, index=True, nullable=False)
    row_index = Column(Integer, nullable=False)  # Original row index in CSV
    content = Column(Text, nullable=False)  # The text that was embedded
    embedding = Column(Vector(384), nullable=False)  # sentence-transformers all-MiniLM-L6-v2
    metadata_ = Column("metadata", JSON, nullable=True)  # Additional row data
    created_at = Column(DateTime(timezone=True), server_default=func.now())
