from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from models.dataset_state import DatasetState


class IngestionAgent:
    def __init__(self, raw_storage: Path = Path("storage/raw")) -> None:
        self.raw_storage = raw_storage
        self.raw_storage.mkdir(parents=True, exist_ok=True)

    def ingest(self, source_path: Path, dataset_id: str) -> DatasetState:
        target_path = self.raw_storage / f"{dataset_id}{source_path.suffix}"
        shutil.copy2(source_path, target_path)
        state = DatasetState(dataset_id=dataset_id, raw_path=target_path)
        state.log_reasoning(
            "ingestion",
            f"Copied dataset to {target_path}",
            confidence=0.9,
        )
        preview = self._build_preview(target_path)
        state.schema = preview.get("schema")
        return state

    def ingest_uploaded(self, uploaded_path: Path, dataset_id: str) -> DatasetState:
        """
        Accept a path already written by API (uploaded file) and register as raw.
        """
        target_path = self.raw_storage / uploaded_path.name
        if uploaded_path != target_path:
            shutil.copy2(uploaded_path, target_path)
        state = DatasetState(dataset_id=dataset_id, raw_path=target_path)
        state.log_reasoning(
            "ingestion",
            f"Stored uploaded dataset at {target_path}",
            confidence=0.9,
        )
        preview = self._build_preview(target_path)
        state.schema = preview.get("schema")
        return state

    def _build_preview(self, path: Path) -> Dict[str, Optional[Dict[str, str]]]:
        try:
            df = pd.read_csv(path, nrows=50)
            schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
            return {"head": df.head().to_dict(orient="records"), "schema": schema}
        except Exception:
            return {"head": None, "schema": None}

