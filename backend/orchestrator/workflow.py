from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from agents.cleaning_agent import CleaningAgent
from agents.documentation_agent import DocumentationAgent
from agents.ingestion import IngestionAgent
from agents.quality_agent import DataQualityAgent
from agents.schema_agent import SchemaUnderstandingAgent
from agents.versioning_agent import VersioningAgent
from models.dataset_state import DatasetState
from models.db_models import DatasetRecord


class Orchestrator:
    def __init__(self) -> None:
        self.ingestion_agent = IngestionAgent()
        self.schema_agent = SchemaUnderstandingAgent()
        self.quality_agent = DataQualityAgent()
        self.cleaning_agent = CleaningAgent()
        self.documentation_agent = DocumentationAgent()
        self.versioning_agent = VersioningAgent()

    async def ingest(self, session: AsyncSession, dataset_id: str, source_path: Path) -> DatasetState:
        # source_path is expected to be an on-disk file (may be uploaded already).
        state = self.ingestion_agent.ingest_uploaded(uploaded_path=source_path, dataset_id=dataset_id)
        state = await self._upsert_state(session, state)
        return state

    async def analyze_schema(self, session: AsyncSession, dataset_id: str) -> DatasetState:
        state = await self._get_state(session, dataset_id)
        state = self.schema_agent.analyze(state)
        state = await self._upsert_state(session, state)
        return state

    async def analyze_quality(self, session: AsyncSession, dataset_id: str) -> DatasetState:
        state = await self._get_state(session, dataset_id)
        state = self.quality_agent.analyze(state)
        state = await self._upsert_state(session, state)
        return state

    async def apply_approved_fixes(
        self,
        session: AsyncSession,
        dataset_id: str,
        approved_fixes: Optional[List[Dict]] = None,
    ) -> DatasetState:
        state = await self._get_state(session, dataset_id)
        fixes = approved_fixes or state.approved_fixes
        state = self.cleaning_agent.apply_fixes(state, fixes)
        state = self.versioning_agent.create_version(state)
        state = self.documentation_agent.generate(state)
        state = await self._upsert_state(session, state)
        return state

    async def run_pipeline(
        self,
        session: AsyncSession,
        dataset_id: str,
        source_path: Path,
        approved_fixes: Optional[List[Dict]] = None,
    ) -> DatasetState:
        state = await self.ingest(session, dataset_id, source_path)
        state = await self.analyze_schema(session, dataset_id)
        state = await self.analyze_quality(session, dataset_id)
        if approved_fixes is not None:
            state = await self.apply_approved_fixes(session, dataset_id, approved_fixes)
        return state

    async def health_report(self, session: AsyncSession, dataset_id: str) -> List[Dict[str, object]]:
        state = await self._get_state(session, dataset_id)
        return state.quality_issues

    async def get_dataset_card(self, session: AsyncSession, dataset_id: str) -> Dict[str, object]:
        state = await self._get_state(session, dataset_id)
        if not state.dataset_card:
            state = self.documentation_agent.generate(state)
            state = await self._upsert_state(session, state)
        return state.dataset_card or {}

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

