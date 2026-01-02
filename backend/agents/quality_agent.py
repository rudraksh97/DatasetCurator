from __future__ import annotations

from typing import Dict, List

import pandas as pd

from models.dataset_state import DatasetState


class DataQualityAgent:
    def analyze(self, state: DatasetState) -> DatasetState:
        if not state.raw_path:
            return state

        issues: List[Dict[str, object]] = []
        try:
            df = pd.read_csv(state.raw_path, nrows=2000)
            for col in df.columns:
                missing_ratio = df[col].isna().mean()
                if missing_ratio > 0.2:
                    issues.append(
                        {
                            "column": col,
                            "issue": "high_missing_rate",
                            "severity": "medium",
                            "suggested_fix": "impute",
                            "confidence": round(min(0.95, missing_ratio + 0.3), 2),
                        }
                    )
            state.quality_issues = issues
            state.log_reasoning(
                "quality",
                "Analyzed missing values on sample rows",
                confidence=0.75,
            )
        except Exception:
            state.log_reasoning(
                "quality",
                "Quality analysis failed; dataset unreadable",
                confidence=0.2,
            )
        return state

