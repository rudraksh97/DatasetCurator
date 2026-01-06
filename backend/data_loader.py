"""Efficient data loading utilities for large datasets.

This module provides memory-efficient operations for handling large CSV files
using chunked reading and streaming. It supports:

- Smart loading with automatic large file detection
- Chunked row counting and filtering
- Streaming statistics calculation
- Reservoir sampling for random row selection

Configuration is loaded from the centralized config module.
"""
from __future__ import annotations

import io
from typing import Optional, Tuple

import pandas as pd

from config import settings
from services.storage import get_storage


async def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes.
    
    Args:
        file_path: Path or key to the file.
    
    Returns:
        File size in MB.
    """
    storage = get_storage()
    try:
        stats = await storage.get_file_stats(str(file_path))
        return stats["size"] / (1024 * 1024)
    except Exception:
        return 0.0


async def estimate_row_count(file_path: str) -> int:
    """Quickly estimate row count.
    
    Uses a sample of the first 1000 rows to estimate the average
    row size, then extrapolates based on total file size.
    
    Args:
        file_path: Path or key to CSV file.
    
    Returns:
        Estimated number of rows.
    """
    storage = get_storage()
    try:
        # Read header + small sample
        # Optimized: Fetch only the first 1MB using read_head (Range request)
        content = await storage.read_head(str(file_path), 1024 * 1024)
        
        chunk = pd.read_csv(io.BytesIO(content), nrows=1000)
        if len(chunk) == 0:
            return 0
        
        size_mb = await get_file_size_mb(file_path)
        file_size_bytes = size_mb * 1024 * 1024
        
        # Approximate
        avg_row_size = len(chunk.to_csv(index=False)) / len(chunk)
        if avg_row_size == 0: return 0
        
        estimated = int(file_size_bytes / avg_row_size)
        return estimated
    except Exception:
        return 0


async def is_large_dataset(file_path: str) -> Tuple[bool, Optional[int]]:
    """Check if dataset is large and return estimated row count.
    
    Args:
        file_path: Path or key to CSV file.
    
    Returns:
        Tuple of (is_large, estimated_rows).
    """
    storage = get_storage()
    path_str = str(file_path)
    
    if not await storage.exists(path_str):
        return False, None
    
    size_mb = await get_file_size_mb(path_str)
    
    # Only estimate rows if size is borderline, otherwise trust size
    if size_mb > settings.data_loader.large_file_size_mb:
        return True, None
        
    estimated_rows = await estimate_row_count(path_str)
    
    is_large = (
        size_mb > settings.data_loader.large_file_size_mb or 
        estimated_rows > settings.data_loader.large_row_count
    )
    return is_large, estimated_rows


async def load_dataframe_smart(
    file_path: str,
    max_rows: Optional[int] = None,
    sample: bool = False,
) -> pd.DataFrame:
    """Load DataFrame with smart handling for large files.
    
    Args:
        file_path: Path or key to CSV file.
        max_rows: Maximum rows to load.
        sample: If True and file is large, load a random sample.
    
    Returns:
        DataFrame.
    """
    storage = get_storage()
    path_str = str(file_path)
    
    if not await storage.exists(path_str):
        return pd.DataFrame()
    
    is_large, estimated_rows = await is_large_dataset(path_str)
    sample_size = settings.data_loader.sample_size
    
    content = await storage.read_file(path_str)
    file_obj = io.BytesIO(content)
    
    if is_large:
        if sample:
            total_rows = estimated_rows or 1_000_000
            actual_sample_size = min(sample_size, total_rows)
            # Reservoir sampling or skip logic
            # Since we have full content in memory (due to simple S3Storage), 
            # we can just read generic sample
            skip = sorted(
                pd.Series(range(total_rows))
                .sample(n=max(0, total_rows - actual_sample_size), random_state=42)
                .tolist()
            )
            # Be careful with skip indices on partial reads, but here we read from bytes
            return pd.read_csv(file_obj, skiprows=skip, nrows=actual_sample_size)
        elif max_rows:
            return pd.read_csv(file_obj, nrows=max_rows)
        else:
            return pd.read_csv(file_obj, nrows=sample_size)
    
    if max_rows:
        return pd.read_csv(file_obj, nrows=max_rows)
        
    return pd.read_csv(file_obj)


async def count_rows_chunked(
    file_path: str, 
    filter_column: Optional[str] = None, 
    filter_value: Optional[str] = None,
) -> int:
    """Count rows efficiently using chunked reading."""
    storage = get_storage()
    path_str = str(file_path)
    
    if not await storage.exists(path_str):
        return 0

    count = 0
    chunk_size = settings.data_loader.chunk_size
    
    try:
        content = await storage.read_file(path_str)
        file_obj = io.BytesIO(content)
        
        for chunk in pd.read_csv(file_obj, chunksize=chunk_size):
            if filter_column and filter_value:
                if filter_column in chunk.columns:
                    chunk = chunk[chunk[filter_column].astype(str).str.lower() == str(filter_value).lower()]
            count += len(chunk)
    except Exception as e:
        print(f"[DataLoader] Error counting rows: {e}")
    return count


async def _iterate_column_chunks(file_path: str, column: str):
    """Iterate over column chunks, dropping nulls."""
    storage = get_storage()
    path_str = str(file_path)
    chunk_size = settings.data_loader.chunk_size
    
    content = await storage.read_file(path_str)
    file_obj = io.BytesIO(content)
    
    for chunk in pd.read_csv(file_obj, chunksize=chunk_size, usecols=[column]):
        yield chunk.dropna(subset=[column])


async def calculate_stat_chunked(
    file_path: str,
    column: str,
    stat: str = "mean",
) -> Optional[float]:
    """Calculate statistics efficiently using chunked reading."""
    storage = get_storage()
    path_str = str(file_path)
    
    if stat not in ["mean", "sum", "min", "max"]:
        return None
    
    if not await storage.exists(path_str):
        return None
    
    try:
        # For simple byte-stream, we can reuse logic, but we need to iterate async generator
        # Implementing manually for clarity since we can't easily pass async generator to dict lambda
        
        total_sum = 0.0
        total_count = 0
        extremum = None
        
        found_col = False
        
        async for chunk in _iterate_column_chunks(path_str, column):
            found_col = True
            if stat == "mean":
                total_sum += chunk[column].sum()
                total_count += len(chunk)
            elif stat == "sum":
                total_sum += chunk[column].sum()
            elif stat == "min":
                val = chunk[column].min()
                extremum = val if extremum is None else min(extremum, val)
            elif stat == "max":
                val = chunk[column].max()
                extremum = val if extremum is None else max(extremum, val)
                
        if not found_col:
            # Maybe column check failed inside iterator or file empty
            return None

        if stat == "mean":
            return total_sum / total_count if total_count > 0 else None
        elif stat == "sum":
            return total_sum
        else:
            return extremum
            
    except Exception as e:
        print(f"[DataLoader] Error calculating {stat}: {e}")
        return None


async def group_count_chunked(file_path: str, column: str) -> dict:
    """Count groups efficiently using chunked reading."""
    storage = get_storage()
    path_str = str(file_path)
    
    counts: dict = {}
    chunk_size = settings.data_loader.chunk_size
    
    try:
        content = await storage.read_file(path_str)
        file_obj = io.BytesIO(content)
        
        for chunk in pd.read_csv(file_obj, chunksize=chunk_size, usecols=[column]):
            if column in chunk.columns:
                chunk_counts = chunk[column].value_counts().to_dict()
                for key, val in chunk_counts.items():
                    counts[key] = counts.get(key, 0) + val
    except Exception as e:
        print(f"[DataLoader] Error grouping: {e}")
    
    return counts


async def get_random_sample_chunked(
    file_path: str,
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
    n: int = 1,
) -> pd.DataFrame:
    """Get random sample efficiently using reservoir sampling."""
    import random
    storage = get_storage()
    path_str = str(file_path)
    
    samples = []
    seen = 0
    chunk_size = settings.data_loader.chunk_size
    
    try:
        content = await storage.read_file(path_str)
        file_obj = io.BytesIO(content)
    
        for chunk in pd.read_csv(file_obj, chunksize=chunk_size):
            if filter_column and filter_value:
                if filter_column in chunk.columns:
                    chunk = chunk[chunk[filter_column].astype(str).str.lower() == str(filter_value).lower()]
            
            if len(chunk) == 0:
                continue
            
            for _, row in chunk.iterrows():
                seen += 1
                if len(samples) < n:
                    samples.append(row)
                else:
                    j = random.randint(0, seen - 1)
                    if j < n:
                        samples[j] = row
    except Exception:
        pass
    
    if samples:
        return pd.DataFrame(samples)
    return pd.DataFrame()
