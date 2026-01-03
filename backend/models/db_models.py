from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, String, JSON, func

from db import Base


class DatasetRecord(Base):
    """SQLAlchemy model for dataset state persistence."""
    
    __tablename__ = "dataset_states"

    dataset_id = Column(String, primary_key=True, index=True)
    current_version = Column(Integer, default=0)
    raw_path = Column(String, nullable=True)
    curated_path = Column(String, nullable=True)
    schema = Column(JSON, nullable=True)
    quality_issues = Column(JSON, default=list)
    transformation_log = Column(JSON, default=list)
    chat_history = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
