from __future__ import annotations

from datetime import datetime
from typing import Dict

from models.dataset_state import DatasetState


class DocumentationAgent:
    def generate(self, state: DatasetState) -> DatasetState:
        card: Dict[str, object] = {
            "dataset_id": state.dataset_id,
            "generated_at": datetime.utcnow().isoformat(),
            "schema": state.schema,
            "known_limitations": ["Automatically generated card; review required"],
            "recommended_usage": "Use curated dataset versions with documented fixes.",
        }
        state.dataset_card = card
        state.log_reasoning(
            "documentation",
            "Generated lightweight dataset card",
            confidence=0.7,
        )
        return state

