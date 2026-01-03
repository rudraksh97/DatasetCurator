from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from models.dataset_state import DatasetState
from models.db_models import DatasetRecord


class Orchestrator:
    """Simplified orchestrator for dataset processing."""

    def __init__(self, raw_storage: Path = Path("storage/raw"), curated_storage: Path = Path("storage/curated")) -> None:
        self.raw_storage = raw_storage
        self.curated_storage = curated_storage
        self.raw_storage.mkdir(parents=True, exist_ok=True)
        self.curated_storage.mkdir(parents=True, exist_ok=True)

    async def run_pipeline(
        self,
        session: AsyncSession,
        dataset_id: str,
        source_path: Path,
        approved_fixes: Optional[List[Dict]] = None,
    ) -> DatasetState:
        """Run the full pipeline: ingest, analyze schema, check quality, create curated version."""
        
        # 1. Create initial state
        state = DatasetState(dataset_id=dataset_id, raw_path=str(source_path))
        
        # 2. Read and analyze the data
        try:
            df = pd.read_csv(source_path, nrows=2000)
            
            # Infer schema
            state.schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            # Check quality issues (columns with >20% missing values)
            issues: List[Dict[str, object]] = []
            for col in df.columns:
                missing_ratio = df[col].isna().mean()
                if missing_ratio > 0.2:
                    issues.append({
                        "column": col,
                        "issue": f"{missing_ratio:.0%} missing values",
                        "severity": "high" if missing_ratio > 0.5 else "medium",
                    })
            state.quality_issues = issues
            
        except Exception as e:
            state.quality_issues = [{"column": "N/A", "issue": f"Error reading file: {str(e)}", "severity": "high"}]

        # 3. Create curated copy
        curated_path = self.curated_storage / f"{dataset_id}_v1.csv"
        try:
            df_full = pd.read_csv(source_path)
            df_full.to_csv(curated_path, index=False)
            state.curated_path = str(curated_path)
            state.current_version = 1
        except Exception:
            state.curated_path = None

        # 4. Save to database
        state = await self._upsert_state(session, state)
        return state

    async def health_report(self, session: AsyncSession, dataset_id: str) -> List[Dict[str, object]]:
        state = await self._get_state(session, dataset_id)
        return state.quality_issues

    async def _get_state(self, session: AsyncSession, dataset_id: str) -> DatasetState:
        record = await session.get(DatasetRecord, dataset_id)
        if not record:
            raise ValueError(f"Dataset {dataset_id} not found")
        return DatasetState.from_record(record)

    async def _upsert_state(self, session: AsyncSession, state: DatasetState) -> DatasetState:
        payload = state.to_record_payload()
        record = await session.get(DatasetRecord, state.dataset_id)
        if record:
            for key, value in payload.items():
                setattr(record, key, value)
        else:
            record = DatasetRecord(**payload)
            session.add(record)
        await session.commit()
        await session.refresh(record)
        return DatasetState.from_record(record)
