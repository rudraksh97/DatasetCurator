"""Dataset repository for data persistence.

This module implements the Repository pattern for dataset state management,
abstracting SQLAlchemy details from the business logic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from models.dataset_state import DatasetState
from models.db_models import DatasetRecord

if TYPE_CHECKING:
    pass


class DatasetRepository:
    """Repository for dataset state persistence.
    
    Provides a clean interface for CRUD operations on dataset states,
    abstracting the underlying SQLAlchemy implementation.
    
    Attributes:
        _session: Database session for operations.
    """
    
    def __init__(self, session: AsyncSession):
        """Initialize the repository.
        
        Args:
            session: SQLAlchemy async session.
        """
        self._session = session
    
    async def get(self, dataset_id: str) -> DatasetState:
        """Get a dataset state by ID.
        
        Args:
            dataset_id: Unique dataset identifier.
        
        Returns:
            DatasetState instance.
        
        Raises:
            ValueError: If dataset not found.
        """
        record = await self._session.get(DatasetRecord, dataset_id)
        if not record:
            raise ValueError(f"Dataset {dataset_id} not found")
        return DatasetState.from_record(record)
    
    async def get_or_none(self, dataset_id: str) -> DatasetState | None:
        """Get a dataset state by ID, returning None if not found.
        
        Args:
            dataset_id: Unique dataset identifier.
        
        Returns:
            DatasetState instance or None.
        """
        record = await self._session.get(DatasetRecord, dataset_id)
        if not record:
            return None
        return DatasetState.from_record(record)
    
    async def save(self, state: DatasetState) -> DatasetState:
        """Save or update a dataset state.
        
        Args:
            state: Dataset state to save.
        
        Returns:
            The saved state (with any DB-generated fields).
        """
        payload = state.to_record_payload()
        record = await self._session.get(DatasetRecord, state.dataset_id)
        
        if record:
            for key, value in payload.items():
                setattr(record, key, value)
        else:
            record = DatasetRecord(**payload)
            self._session.add(record)
        
        await self._session.commit()
        await self._session.refresh(record)
        return DatasetState.from_record(record)
    
    async def delete(self, dataset_id: str) -> bool:
        """Delete a dataset state.
        
        Args:
            dataset_id: Dataset ID to delete.
        
        Returns:
            True if deleted, False if not found.
        """
        record = await self._session.get(DatasetRecord, dataset_id)
        if not record:
            return False
        
        await self._session.delete(record)
        await self._session.commit()
        return True
    
    async def exists(self, dataset_id: str) -> bool:
        """Check if a dataset exists.
        
        Args:
            dataset_id: Dataset ID to check.
        
        Returns:
            True if exists, False otherwise.
        """
        record = await self._session.get(DatasetRecord, dataset_id)
        return record is not None
