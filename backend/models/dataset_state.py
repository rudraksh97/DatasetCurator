from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from models.db_models import DatasetRecord


class DatasetState(BaseModel):
    """State for a dataset being curated."""
    
    dataset_id: str
    current_version: int = 0
    raw_path: Optional[Path] = None
    curated_path: Optional[Path] = None
    schema: Optional[Dict[str, Any]] = None
    quality_issues: List[Dict[str, Any]] = Field(default_factory=list)
    transformation_log: List[Dict[str, Any]] = Field(default_factory=list)
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)

    def to_record_payload(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "current_version": self.current_version,
            "raw_path": str(self.raw_path) if self.raw_path else None,
            "curated_path": str(self.curated_path) if self.curated_path else None,
            "schema": self.schema,
            "quality_issues": self.quality_issues,
            "transformation_log": self.transformation_log,
            "chat_history": self.chat_history,
        }

    @classmethod
    def from_record(cls, record: "DatasetRecord") -> "DatasetState":
        return cls(
            dataset_id=record.dataset_id,
            current_version=record.current_version,
            raw_path=Path(record.raw_path) if record.raw_path else None,
            curated_path=Path(record.curated_path) if record.curated_path else None,
            schema=record.schema,
            quality_issues=record.quality_issues or [],
            transformation_log=record.transformation_log or [],
            chat_history=record.chat_history or [],
        )
