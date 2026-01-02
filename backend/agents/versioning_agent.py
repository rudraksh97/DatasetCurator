from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

from models.dataset_state import DatasetState


class VersioningAgent:
    def __init__(self, metadata_dir: Path = Path("storage")) -> None:
        self.metadata_dir = metadata_dir
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def create_version(self, state: DatasetState) -> DatasetState:
        new_version = state.bump_version()
        record: Dict[str, object] = {
            "version": new_version,
            "timestamp": datetime.utcnow().isoformat(),
            "curated_path": str(state.curated_path) if state.curated_path else None,
            "transformation_log": state.transformation_log[-1:] if state.transformation_log else [],
        }
        state.log_transformation("versioning", record)
        state.log_reasoning(
            "versioning",
            f"Created immutable version {new_version}",
            confidence=0.85,
        )
        return state

