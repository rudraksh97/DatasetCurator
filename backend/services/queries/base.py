"""Base query handler and utilities.

This module provides the abstract base class for query handlers
and common utility functions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from protocols import QueryResult


def convert_to_native_type(value: Any) -> Any:
    """Convert numpy/pandas types to native Python types for JSON serialization.
    
    Args:
        value: Value to convert.
    
    Returns:
        Native Python type equivalent.
    """
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, pd.Int64Dtype)):
        return int(value)
    if isinstance(value, (np.floating, pd.Float64Dtype)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, dict):
        return {k: convert_to_native_type(v) for k, v in value.items()}
    if isinstance(value, list):
        return [convert_to_native_type(v) for v in value]
    return value


def find_column_case_insensitive(df: pd.DataFrame, column: str) -> Optional[str]:
    """Find a column by name (case-insensitive).
    
    Args:
        df: DataFrame to search.
        column: Column name to find.
    
    Returns:
        Actual column name or None if not found.
    """
    if column in df.columns:
        return column
    for c in df.columns:
        if c.lower() == column.lower():
            return c
    return None


def coerce_filter_value(df: pd.DataFrame, column: str, value: Any) -> Any:
    """Try to coerce a filter value to match column dtype.
    
    Args:
        df: DataFrame containing the column.
        column: Column name.
        value: Value to coerce.
    
    Returns:
        Coerced value or original if coercion fails.
    """
    try:
        if pd.api.types.is_integer_dtype(df[column].dtype):
            return int(value)
        elif pd.api.types.is_float_dtype(df[column].dtype):
            return float(value)
    except (ValueError, TypeError):
        pass
    return value


def apply_filter_to_df(
    df: pd.DataFrame,
    filter_column: str,
    filter_value: Any,
) -> tuple[pd.DataFrame, str, Optional[str]]:
    """Apply a filter to a DataFrame.
    
    Args:
        df: DataFrame to filter.
        filter_column: Column to filter by.
        filter_value: Value to match.
    
    Returns:
        Tuple of (filtered_df, filter_info_string, error_message).
    """
    filter_col_match = find_column_case_insensitive(df, filter_column)
    if not filter_col_match:
        return df, "", f"Filter column '{filter_column}' not found. Available: {', '.join(df.columns[:10])}"
    
    filter_value = coerce_filter_value(df, filter_col_match, filter_value)
    
    # Try exact match first
    matches = df[df[filter_col_match].astype(str).str.lower() == str(filter_value).lower()]
    
    # If no exact match, try partial match
    if len(matches) == 0:
        matches = df[df[filter_col_match].astype(str).str.lower().str.contains(str(filter_value).lower(), na=False)]
    
    if len(matches) == 0:
        return df, "", f"No rows found where {filter_col_match} matches '{filter_value}'"
    
    filter_info = f" (filtered by {filter_col_match} = '{filter_value}', {len(matches)} rows)"
    return matches, filter_info, None


class BaseQueryHandler(ABC):
    """Abstract base class for query handlers.
    
    Each handler is responsible for executing a specific type of query
    on a DataFrame. Subclasses must implement can_handle() and execute().
    """
    
    @property
    @abstractmethod
    def query_type(self) -> str:
        """The type of query this handler supports."""
        ...
    
    @property
    @abstractmethod
    def supported_functions(self) -> List[str]:
        """List of function names this handler can process."""
        ...
    
    def can_handle(self, function_name: str) -> bool:
        """Check if this handler can process the given function.
        
        Args:
            function_name: Name of the function to handle.
        
        Returns:
            True if this handler supports the function.
        """
        return function_name in self.supported_functions
    
    @abstractmethod
    def execute(
        self,
        df: pd.DataFrame,
        arguments: Dict[str, Any],
    ) -> QueryResult:
        """Execute the query on the DataFrame.
        
        Args:
            df: DataFrame to query.
            arguments: Query parameters.
        
        Returns:
            Dictionary with 'success' key and results or 'error'.
        """
        ...
    
    def _error(self, message: str) -> QueryResult:
        """Create an error result.
        
        Args:
            message: Error message.
        
        Returns:
            Error result dictionary.
        """
        return {"success": False, "error": message}
    
    def _success(self, **kwargs: Any) -> QueryResult:
        """Create a success result.
        
        Args:
            **kwargs: Result data.
        
        Returns:
            Success result dictionary.
        """
        return {"success": True, **kwargs}
