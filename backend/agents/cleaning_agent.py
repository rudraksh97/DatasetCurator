from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd

from models.dataset_state import DatasetState


class CleaningAgent:
    def __init__(self, curated_storage: Path = Path("storage/curated")) -> None:
        self.curated_storage = curated_storage
        self.curated_storage.mkdir(parents=True, exist_ok=True)

    def apply_fixes(self, state: DatasetState, approved_fixes: List[Dict]) -> DatasetState:
        if not state.raw_path:
            return state

        curated_path = self.curated_storage / f"{state.dataset_id}_v{state.current_version + 1}.csv"
        try:
            df = pd.read_csv(state.raw_path)
            # Placeholder: no-op cleaning until specific fixes are provided.
            df.to_csv(curated_path, index=False)
            state.curated_path = curated_path
            state.approved_fixes = approved_fixes
            state.log_transformation(
                "cleaning",
                {"applied_fixes": approved_fixes, "output": str(curated_path)},
            )
            state.log_reasoning(
                "cleaning",
                "Applied approved fixes (placeholder no-op copy)",
                confidence=0.6,
            )
        except Exception:
            # Fallback: still copy raw file to preserve workflow continuity
            shutil.copy2(state.raw_path, curated_path)
            state.curated_path = curated_path
            state.log_reasoning(
                "cleaning",
                "Failed to apply fixes; copied raw dataset to curated storage",
                confidence=0.1,
            )
        return state

