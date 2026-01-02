from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from models.db_models import DatasetRecord


class DatasetState(BaseModel):
    dataset_id: str
    current_version: int = 0
    raw_path: Optional[Path] = None
    curated_path: Optional[Path] = None
    schema: Optional[Dict[str, Any]] = None
    quality_issues: List[Dict[str, Any]] = Field(default_factory=list)
    approved_fixes: List[Dict[str, Any]] = Field(default_factory=list)
    transformation_log: List[Dict[str, Any]] = Field(default_factory=list)
    agent_reasoning: List[Dict[str, Any]] = Field(default_factory=list)
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    dataset_card: Optional[Dict[str, Any]] = None

    def log_reasoning(self, agent: str, reasoning: str, confidence: float) -> None:
        self.agent_reasoning.append(
            {"agent": agent, "reasoning": reasoning, "confidence": confidence}
        )

    def log_transformation(self, step: str, details: Dict[str, Any]) -> None:
        self.transformation_log.append({"step": step, "details": details})

    def bump_version(self) -> int:
        self.current_version += 1
        return self.current_version

    def to_record_payload(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "current_version": self.current_version,
            "raw_path": str(self.raw_path) if self.raw_path else None,
            "curated_path": str(self.curated_path) if self.curated_path else None,
            "schema": self.schema,
            "quality_issues": self.quality_issues,
            "approved_fixes": self.approved_fixes,
            "transformation_log": self.transformation_log,
            "agent_reasoning": self.agent_reasoning,
            "chat_history": self.chat_history,
            "dataset_card": self.dataset_card,
        }

    @classmethod
    def from_record(cls, record: "DatasetRecord") -> "DatasetState":  # type: ignore[name-defined]
        return cls(
            dataset_id=record.dataset_id,
            current_version=record.current_version,
            raw_path=Path(record.raw_path) if record.raw_path else None,
            curated_path=Path(record.curated_path) if record.curated_path else None,
            schema=record.schema,
            quality_issues=record.quality_issues or [],
            approved_fixes=record.approved_fixes or [],
            transformation_log=record.transformation_log or [],
            agent_reasoning=record.agent_reasoning or [],
            chat_history=record.chat_history or [],
            dataset_card=record.dataset_card,
        )

