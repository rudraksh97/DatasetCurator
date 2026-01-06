"""Dataset state management models.

This module provides the DatasetState Pydantic model for managing
the in-memory state of datasets being curated, including paths,
schema, transformation history, and chat history.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from models.db_models import DatasetRecord


class DatasetState(BaseModel):
    """State for a dataset being curated.
    
    Attributes:
        dataset_id: Unique identifier for the dataset.
        current_version: Current version number (incremented on changes).
        raw_path: Path to the original uploaded file.
        curated_path: Path to the latest curated version.
        schema: Column names and their data types.
        transformation_log: History of transformations applied.
        chat_history: Conversation history for context.
    """
    
    dataset_id: str
    current_version: int = 0
    raw_path: Optional[Path] = None
    curated_path: Optional[Path] = None
    schema: Optional[Dict[str, Any]] = None
    transformation_log: List[Dict[str, Any]] = Field(default_factory=list)
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)

    def to_record_payload(self) -> Dict[str, Any]:
        """Convert state to dictionary for database storage.
        
        Returns:
            Dictionary with all fields serialized for the database.
        """
        return {
            "dataset_id": self.dataset_id,
            "current_version": self.current_version,
            "raw_path": str(self.raw_path) if self.raw_path else None,
            "curated_path": str(self.curated_path) if self.curated_path else None,
            "schema": self.schema,
            "transformation_log": self.transformation_log,
            "chat_history": self.chat_history,
        }

    @classmethod
    def from_record(cls, record: "DatasetRecord") -> "DatasetState":
        """Create a DatasetState from a database record.
        
        Args:
            record: SQLAlchemy DatasetRecord instance.
        
        Returns:
            DatasetState instance populated from the record.
        """
        return cls(
            dataset_id=record.dataset_id,
            current_version=record.current_version,
            raw_path=Path(record.raw_path) if record.raw_path else None,
            curated_path=Path(record.curated_path) if record.curated_path else None,
            schema=record.schema,
            transformation_log=record.transformation_log or [],
            chat_history=record.chat_history or [],
        )
