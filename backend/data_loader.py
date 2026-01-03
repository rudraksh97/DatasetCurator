"""Efficient data loading for large datasets."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# Thresholds for large dataset handling
LARGE_FILE_SIZE_MB = 100  # 100MB
LARGE_ROW_COUNT = 1_000_000  # 1M rows
SAMPLE_SIZE = 10_000  # Sample size for exploratory queries
CHUNK_SIZE = 50_000  # Chunk size for processing


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)


def estimate_row_count(file_path: Path) -> int:
    """Quickly estimate row count without loading full file."""
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
    """Check if dataset is large and return estimated row count."""
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
    
    Args:
        file_path: Path to CSV file
        max_rows: Maximum rows to load (None = all)
        sample: If True and file is large, load a random sample
    
    Returns:
        DataFrame
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


def count_rows_chunked(file_path: Path, filter_column: Optional[str] = None, filter_value: Optional[str] = None) -> int:
    """Count rows efficiently using chunked reading.
    
    Args:
        file_path: Path to CSV
        filter_column: Optional column to filter by
        filter_value: Optional value to match
    
    Returns:
        Total count
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


def calculate_stat_chunked(
    file_path: Path,
    column: str,
    stat: str = "mean",
) -> Optional[float]:
    """Calculate statistics efficiently using chunked reading.
    
    Args:
        file_path: Path to CSV
        column: Column name
        stat: 'mean', 'sum', 'min', 'max'
    
    Returns:
        Statistic value or None
    """
    if stat not in ["mean", "sum", "min", "max"]:
        return None
    
    try:
        # Read first chunk to check column exists and type
        first_chunk = pd.read_csv(file_path, nrows=1000)
        if column not in first_chunk.columns:
            return None
        
        if not pd.api.types.is_numeric_dtype(first_chunk[column]):
            return None
        
        # Calculate using chunks
        if stat == "mean":
            total_sum = 0.0
            total_count = 0
            for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, usecols=[column]):
                chunk = chunk.dropna(subset=[column])
                total_sum += chunk[column].sum()
                total_count += len(chunk)
            return total_sum / total_count if total_count > 0 else None
        
        elif stat == "sum":
            total_sum = 0.0
            for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, usecols=[column]):
                chunk = chunk.dropna(subset=[column])
                total_sum += chunk[column].sum()
            return total_sum
        
        elif stat == "min":
            min_val = None
            for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, usecols=[column]):
                chunk = chunk.dropna(subset=[column])
                chunk_min = chunk[column].min()
                if min_val is None or chunk_min < min_val:
                    min_val = chunk_min
            return min_val
        
        elif stat == "max":
            max_val = None
            for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, usecols=[column]):
                chunk = chunk.dropna(subset=[column])
                chunk_max = chunk[column].max()
                if max_val is None or chunk_max > max_val:
                    max_val = chunk_max
            return max_val
            
    except Exception as e:
        print(f"[DataLoader] Error calculating {stat}: {e}")
    
    return None


def group_count_chunked(
    file_path: Path,
    column: str,
) -> dict:
    """Count groups efficiently using chunked reading.
    
    Args:
        file_path: Path to CSV
        column: Column to group by
    
    Returns:
        Dictionary of {value: count}
    """
    counts = {}
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
    """Get random sample efficiently from large file.
    
    Args:
        file_path: Path to CSV
        filter_column: Optional column to filter by
        filter_value: Optional value to match
        n: Number of samples
    
    Returns:
        DataFrame with samples
    """
    try:
        # For large files, use reservoir sampling approach
        # Load chunks and randomly sample from them
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
        
    except Exception as e:
        print(f"[DataLoader] Error sampling: {e}")
        # Fallback: load sample and filter
        df = load_dataframe_smart(file_path, max_rows=SAMPLE_SIZE, sample=True)
        if filter_column and filter_value and filter_column in df.columns:
            df = df[df[filter_column].astype(str).str.lower() == str(filter_value).lower()]
        return df.sample(n=min(n, len(df))) if len(df) > 0 else pd.DataFrame()
