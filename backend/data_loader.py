"""Efficient data loading utilities for large datasets.

This module provides memory-efficient operations for handling large CSV files
using chunked reading and streaming. It supports:

- Smart loading with automatic large file detection
- Chunked row counting and filtering
- Streaming statistics calculation
- Reservoir sampling for random row selection

Configurable thresholds (via environment variables):
- LARGE_FILE_SIZE_MB: Files larger than this are considered large (default: 100MB)
- LARGE_ROW_COUNT: Datasets with more rows are considered large (default: 1M)
- SAMPLE_SIZE: Default sample size for exploratory queries (default: 10,000)
- CHUNK_SIZE: Processing chunk size (default: 50,000)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# Thresholds for large dataset handling (configurable via environment)
LARGE_FILE_SIZE_MB = int(os.getenv("LARGE_FILE_SIZE_MB", "100"))
LARGE_ROW_COUNT = int(os.getenv("LARGE_ROW_COUNT", "1000000"))
SAMPLE_SIZE = int(os.getenv("DATA_SAMPLE_SIZE", "10000"))
CHUNK_SIZE = int(os.getenv("DATA_CHUNK_SIZE", "50000"))


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes.
    
    Args:
        file_path: Path to the file.
    
    Returns:
        File size in MB.
    """
    return os.path.getsize(file_path) / (1024 * 1024)


def estimate_row_count(file_path: Path) -> int:
    """Quickly estimate row count without loading full file.
    
    Uses a sample of the first 1000 rows to estimate the average
    row size, then extrapolates based on total file size.
    
    Args:
        file_path: Path to CSV file.
    
    Returns:
        Estimated number of rows.
    """
    try:
        # Read first chunk to estimate
        chunk = pd.read_csv(file_path, nrows=1000)
        if len(chunk) == 0:
            return 0
        
        # Estimate based on file size
        file_size = os.path.getsize(file_path)
        avg_row_size = len(chunk.to_csv(index=False)) / len(chunk)
        estimated = int(file_size / avg_row_size)
        return estimated
    except Exception:
        return 0


def is_large_dataset(file_path: Path) -> Tuple[bool, Optional[int]]:
    """Check if dataset is large and return estimated row count.
    
    A dataset is considered large if:
    - File size exceeds LARGE_FILE_SIZE_MB (100MB), OR
    - Estimated rows exceed LARGE_ROW_COUNT (1M)
    
    Args:
        file_path: Path to CSV file.
    
    Returns:
        Tuple of (is_large, estimated_rows).
    """
    if not file_path.exists():
        return False, None
    
    size_mb = get_file_size_mb(file_path)
    estimated_rows = estimate_row_count(file_path)
    
    is_large = size_mb > LARGE_FILE_SIZE_MB or estimated_rows > LARGE_ROW_COUNT
    return is_large, estimated_rows


def load_dataframe_smart(
    file_path: Path,
    max_rows: Optional[int] = None,
    sample: bool = False,
) -> pd.DataFrame:
    """Load DataFrame with smart handling for large files.
    
    For large files:
    - If sample=True: Returns a random sample of SAMPLE_SIZE rows
    - If max_rows specified: Returns first max_rows rows
    - Otherwise: Returns first SAMPLE_SIZE rows
    
    Args:
        file_path: Path to CSV file.
        max_rows: Maximum rows to load (None = all for small files).
        sample: If True and file is large, load a random sample.
    
    Returns:
        DataFrame with loaded data.
    """
    if not file_path.exists():
        return pd.DataFrame()
    
    is_large, estimated_rows = is_large_dataset(file_path)
    
    # For large files, use sampling or limited rows
    if is_large:
        if sample:
            # Load random sample
            total_rows = estimated_rows or 1_000_000
            sample_size = min(SAMPLE_SIZE, total_rows)
            skip = sorted(
                pd.Series(range(total_rows))
                .sample(n=total_rows - sample_size, random_state=42)
                .tolist()
            )
            return pd.read_csv(file_path, skiprows=skip, nrows=sample_size)
        elif max_rows:
            return pd.read_csv(file_path, nrows=max_rows)
        else:
            # Default: load sample for large files
            return pd.read_csv(file_path, nrows=SAMPLE_SIZE)
    
    # For small files, load normally
    if max_rows:
        return pd.read_csv(file_path, nrows=max_rows)
    return pd.read_csv(file_path)


def count_rows_chunked(
    file_path: Path, 
    filter_column: Optional[str] = None, 
    filter_value: Optional[str] = None
) -> int:
    """Count rows efficiently using chunked reading.
    
    Args:
        file_path: Path to CSV file.
        filter_column: Optional column to filter by.
        filter_value: Optional value to match (case-insensitive).
    
    Returns:
        Total count of matching rows.
    """
    count = 0
    try:
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
            if filter_column and filter_value:
                # Filter chunk
                if filter_column in chunk.columns:
                    chunk = chunk[chunk[filter_column].astype(str).str.lower() == str(filter_value).lower()]
            count += len(chunk)
    except Exception as e:
        print(f"[DataLoader] Error counting rows: {e}")
    return count


def _iterate_column_chunks(file_path: Path, column: str):
    """Iterate over column chunks, dropping nulls."""
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, usecols=[column]):
        yield chunk.dropna(subset=[column])


def _calculate_mean_chunked(file_path: Path, column: str) -> Optional[float]:
    """Calculate mean using chunked reading."""
    total_sum = 0.0
    total_count = 0
    for chunk in _iterate_column_chunks(file_path, column):
        total_sum += chunk[column].sum()
        total_count += len(chunk)
    return total_sum / total_count if total_count > 0 else None


def _calculate_sum_chunked(file_path: Path, column: str) -> float:
    """Calculate sum using chunked reading."""
    total_sum = 0.0
    for chunk in _iterate_column_chunks(file_path, column):
        total_sum += chunk[column].sum()
    return total_sum


def _calculate_extremum_chunked(file_path: Path, column: str, find_max: bool) -> Optional[float]:
    """Calculate min or max using chunked reading."""
    result = None
    for chunk in _iterate_column_chunks(file_path, column):
        chunk_val = chunk[column].max() if find_max else chunk[column].min()
        if result is None:
            result = chunk_val
        elif find_max:
            result = max(result, chunk_val)
        else:
            result = min(result, chunk_val)
    return result


def calculate_stat_chunked(
    file_path: Path,
    column: str,
    stat: str = "mean",
) -> Optional[float]:
    """Calculate statistics efficiently using chunked reading.
    
    Supports streaming calculation of mean, sum, min, and max
    without loading the entire file into memory.
    
    Args:
        file_path: Path to CSV file.
        column: Column name to calculate statistics for.
        stat: Statistic type: 'mean', 'sum', 'min', or 'max'.
    
    Returns:
        Calculated statistic value, or None on error.
    """
    if stat not in ["mean", "sum", "min", "max"]:
        return None
    
    try:
        # Validate column exists and is numeric
        first_chunk = pd.read_csv(file_path, nrows=1000)
        if column not in first_chunk.columns:
            return None
        if not pd.api.types.is_numeric_dtype(first_chunk[column]):
            return None
        
        # Dispatch to appropriate calculator
        stat_calculators = {
            "mean": lambda: _calculate_mean_chunked(file_path, column),
            "sum": lambda: _calculate_sum_chunked(file_path, column),
            "min": lambda: _calculate_extremum_chunked(file_path, column, find_max=False),
            "max": lambda: _calculate_extremum_chunked(file_path, column, find_max=True),
        }
        
        return stat_calculators[stat]()
            
    except Exception as e:
        print(f"[DataLoader] Error calculating {stat}: {e}")
        return None


def group_count_chunked(
    file_path: Path,
    column: str,
) -> dict:
    """Count groups efficiently using chunked reading.
    
    Args:
        file_path: Path to CSV file.
        column: Column to group by.
    
    Returns:
        Dictionary mapping {value: count}.
    """
    counts: dict = {}
    try:
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, usecols=[column]):
            if column in chunk.columns:
                chunk_counts = chunk[column].value_counts().to_dict()
                for key, val in chunk_counts.items():
                    counts[key] = counts.get(key, 0) + val
    except Exception as e:
        print(f"[DataLoader] Error grouping: {e}")
    
    return counts


def get_random_sample_chunked(
    file_path: Path,
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    n: int = 1,
) -> pd.DataFrame:
    """Get random sample efficiently from large file using reservoir sampling.
    
    Uses reservoir sampling algorithm to select random rows without
    loading the entire file into memory.
    
    Args:
        file_path: Path to CSV file.
        filter_column: Optional column to filter by.
        filter_value: Optional value to match (case-insensitive).
        n: Number of samples to return.
    
    Returns:
        DataFrame with n random rows.
    """
    import random
    samples = []
    seen = 0
    
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
        # Apply filter if needed
        if filter_column and filter_value:
            if filter_column in chunk.columns:
                chunk = chunk[chunk[filter_column].astype(str).str.lower() == str(filter_value).lower()]
        
        if len(chunk) == 0:
            continue
        
        # Reservoir sampling: randomly replace samples as we see more data
        for _, row in chunk.iterrows():
            seen += 1
            if len(samples) < n:
                samples.append(row)
            else:
                # Randomly replace with probability n/seen
                j = random.randint(0, seen - 1)
                if j < n:
                    samples[j] = row
    
    if samples:
        return pd.DataFrame(samples)
    return pd.DataFrame()
