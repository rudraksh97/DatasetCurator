from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Integer, String, JSON, func

from db import Base


class DatasetRecord(Base):
    __tablename__ = "dataset_states"

    dataset_id = Column(String, primary_key=True, index=True)
    current_version = Column(Integer, default=0)
    raw_path = Column(String, nullable=True)
    curated_path = Column(String, nullable=True)
    schema = Column(JSON, nullable=True)
    quality_issues = Column(JSON, default=list)
    approved_fixes = Column(JSON, default=list)
    transformation_log = Column(JSON, default=list)
    agent_reasoning = Column(JSON, default=list)
    chat_history = Column(JSON, default=list)
    dataset_card = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def to_state_payload(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "current_version": self.current_version,
            "raw_path": self.raw_path,
            "curated_path": self.curated_path,
            "schema": self.schema,
            "quality_issues": self.quality_issues or [],
            "approved_fixes": self.approved_fixes or [],
            "transformation_log": self.transformation_log or [],
            "agent_reasoning": self.agent_reasoning or [],
            "chat_history": self.chat_history or [],
            "dataset_card": self.dataset_card,
        }

    @staticmethod
    def from_state(state: "DatasetState") -> "DatasetRecord":  # type: ignore  # pragma: no cover - helper
        record = DatasetRecord(
            dataset_id=state.dataset_id,
            current_version=state.current_version,
            raw_path=str(state.raw_path) if state.raw_path else None,
            curated_path=str(state.curated_path) if state.curated_path else None,
            schema=state.schema,
            quality_issues=state.quality_issues,
            approved_fixes=state.approved_fixes,
            transformation_log=state.transformation_log,
            agent_reasoning=state.agent_reasoning,
            chat_history=state.chat_history,
            dataset_card=state.dataset_card,
        )
        return record

