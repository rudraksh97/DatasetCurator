from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from models.dataset_state import DatasetState


class SchemaUnderstandingAgent:
    def analyze(self, state: DatasetState) -> DatasetState:
        if not state.raw_path:
            return state

        schema_hypotheses: List[Dict[str, str]] = []
        try:
            df = pd.read_csv(Path(state.raw_path), nrows=500)
            hypothesis = {col: str(dtype) for col, dtype in df.dtypes.items()}
            schema_hypotheses.append(hypothesis)
            state.schema = hypothesis
            state.log_reasoning(
                "schema",
                "Inferred schema from sample rows",
                confidence=0.8,
            )
        except Exception:
            state.log_reasoning(
                "schema",
                "Failed to infer schema; dataset may be unsupported",
                confidence=0.2,
            )
        return state

