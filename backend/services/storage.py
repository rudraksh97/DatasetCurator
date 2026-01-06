"""Storage abstraction layer for file operations.

This module provides a unified interface for file storage with two implementations:
- LocalStorage: Uses local filesystem (for development)
- S3Storage: Uses AWS S3 (for production on Render)

Usage:
    from services.storage import get_storage
    storage = get_storage()
    await storage.write_file("path/to/file.csv", content)
"""
from __future__ import annotations

import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import pandas as pd


@dataclass
class FileStats:
    """File statistics."""
    size_bytes: int
    modified_at: datetime
    exists: bool


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    async def read_head(self, path: str, n_bytes: int = 1024) -> bytes:
        """Read the first n bytes of a file."""
        pass

    async def read_csv(self, path: str) -> pd.DataFrame:
        """Read CSV file into DataFrame."""
        content = await self.read_file(path)
        return pd.read_csv(io.BytesIO(content))
    
    async def write_csv(self, path: str, df: pd.DataFrame, index: bool = False) -> None:
        """Write DataFrame to CSV file."""
        buffer = io.BytesIO()
        df.to_csv(buffer, index=index)
        await self.write_file(path, buffer.getvalue())


class LocalStorage(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, base_path: Path):
        """Initialize local storage with base path."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to base path."""
        return self.base_path / path
    
    async def read_file(self, path: str) -> bytes:
        """Read file content as bytes."""
        file_path = self._resolve_path(path)
        async with aiofiles.open(file_path, "rb") as f:
            return await f.read()

    async def read_head(self, path: str, n_bytes: int = 1024) -> bytes:
        """Read the first n bytes of a file."""
        file_path = self._resolve_path(path)
        if not file_path.exists():
            return b""
        async with aiofiles.open(file_path, "rb") as f:
            return await f.read(n_bytes)
    
    async def write_file(self, path: str, content: bytes) -> None:
        """Write content to file."""
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
    
    async def delete_file(self, path: str) -> bool:
        """Delete a file."""
        file_path = self._resolve_path(path)
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    async def exists(self, path: str) -> bool:
        """Check if file exists."""
        return self._resolve_path(path).exists()
    
    async def list_files(self, prefix: str) -> List[str]:
        """List files with given prefix."""
        base = self._resolve_path(prefix)
        if not base.exists():
            return []
        
        if base.is_file():
            return [prefix]
        
        files = []
        for item in base.rglob("*"):
            if item.is_file():
                files.append(str(item.relative_to(self.base_path)))
        return files
    
    async def get_file_stats(self, path: str) -> Optional[FileStats]:
        """Get file statistics."""
        file_path = self._resolve_path(path)
        if not file_path.exists():
            return None
        
        stat = file_path.stat()
        return FileStats(
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            exists=True,
        )
    
    def get_local_path(self, path: str) -> Optional[Path]:
        """Get local filesystem path."""
        return self._resolve_path(path)


class S3Storage(StorageBackend):
    """AWS S3 storage backend."""
    
    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        prefix: str = "",
    ):
        """Initialize S3 storage.
        
        Args:
            bucket: S3 bucket name.
            region: AWS region.
            access_key_id: AWS access key (uses env var if not provided).
            secret_access_key: AWS secret key (uses env var if not provided).
            prefix: Optional prefix for all paths.
        """
        import boto3
        
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        
        session_kwargs: Dict[str, Any] = {"region_name": region}
        if access_key_id and secret_access_key:
            session_kwargs["aws_access_key_id"] = access_key_id
            session_kwargs["aws_secret_access_key"] = secret_access_key
        
        self._session = boto3.Session(**session_kwargs)
        self._s3 = self._session.client("s3")
    
    def _get_key(self, path: str) -> str:
        """Get S3 key from path."""
        path = str(path).lstrip("/")
        if self.prefix:
            return f"{self.prefix}/{path}"
        return path

    def _read_file_sync(self, key: str) -> bytes:
        """Sync read file from S3."""
        response = self._s3.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def _read_head_sync(self, key: str, n_bytes: int) -> bytes:
        """Sync read head from S3 using Range."""
        # Range is inclusive, so bytes=0-{n_bytes-1}
        response = self._s3.get_object(
            Bucket=self.bucket, 
            Key=key, 
            Range=f"bytes=0-{n_bytes-1}"
        )
        return response["Body"].read()

    def _write_file_sync(self, key: str, content: bytes) -> None:
        """Sync write file to S3."""
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=content)

    def _delete_file_sync(self, key: str) -> bool:
        """Sync delete file from S3."""
        try:
            self._s3.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def _exists_sync(self, key: str) -> bool:
        """Sync check existence in S3."""
        from botocore.exceptions import ClientError
        
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            # 404 means file doesn't exist, which is expected
            if e.response.get('Error', {}).get('Code') == '404':
                return False
            # Log other errors but still return False
            print(f"[S3] Error checking existence for {key}: {e}")
            return False
        except Exception as e:
            print(f"[S3] Unexpected error checking existence for {key}: {e}")
            return False
            
    def _get_file_stats_sync(self, key: str) -> Optional[FileStats]:
        """Sync get stats from S3."""
        try:
            response = self._s3.head_object(Bucket=self.bucket, Key=key)
            return FileStats(
                size_bytes=response["ContentLength"],
                modified_at=response["LastModified"],
                exists=True,
            )
        except Exception:
            return None
    
    async def read_file(self, path: str) -> bytes:
        """Read file content from S3 (non-blocking)."""
        import asyncio
        key = self._get_key(path)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._read_file_sync, key)

    async def read_head(self, path: str, n_bytes: int = 1024) -> bytes:
        """Read first n bytes from S3 (non-blocking)."""
        import asyncio
        key = self._get_key(path)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._read_head_sync, key, n_bytes)
    
    async def write_file(self, path: str, content: bytes) -> None:
        """Write content to S3 (non-blocking)."""
        import asyncio
        key = self._get_key(path)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_file_sync, key, content)
    
    async def delete_file(self, path: str) -> bool:
        """Delete a file from S3 (non-blocking)."""
        import asyncio
        key = self._get_key(path)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._delete_file_sync, key)
    
    async def exists(self, path: str) -> bool:
        """Check if file exists in S3 (non-blocking)."""
        import asyncio
        key = self._get_key(path)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._exists_sync, key)
    
    async def list_files(self, prefix: str) -> List[str]:
        """List files in S3 with given prefix (non-blocking possible but keeping simple for now).
           Pagination involves multiple calls so wrapping strictly is harder. 
           Should ideally wrap, but list_files is less critical for latency than read.
        """
        # Leaving sync for now or wrapping entire pagination? 
        # Pagination returns iterator.
        full_prefix = self._get_key(prefix)
        paginator = self._s3.get_paginator("list_objects_v2")
        
        files = []
        # This part is still blocking. For completeness, should move.
        # But paginator yields pages. 
        # For simplicity in this step, I'll execute sync logic in executor.
        import asyncio
        loop = asyncio.get_running_loop()
        
        def _list_sync():
            f = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if self.prefix and key.startswith(self.prefix + "/"):
                        key = key[len(self.prefix) + 1:]
                    f.append(key)
            return f

        return await loop.run_in_executor(None, _list_sync)
    
    async def get_file_stats(self, path: str) -> Optional[FileStats]:
        """Get file statistics from S3 (non-blocking)."""
        import asyncio
        key = self._get_key(path)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_file_stats_sync, key)
    
    def get_local_path(self, path: str) -> Optional[Path]:
        """S3 doesn't have local paths."""
        return None
    
    def generate_presigned_url(self, path: str, expires_in: int = 3600) -> str:
        """Generate a presigned URL for file download.
        
        Args:
            path: File path.
            expires_in: URL expiration in seconds.
        
        Returns:
            Presigned URL string.
        """
        key = self._get_key(path)
        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )


# Storage singleton
_storage_instance: Optional[StorageBackend] = None


def get_storage() -> StorageBackend:
    """Get the configured storage backend singleton.
    
    Returns:
        StorageBackend instance based on configuration.
    """
    global _storage_instance
    
    if _storage_instance is None:
        backend = os.getenv("STORAGE_BACKEND", "local").lower()
        
        if backend == "s3":
            bucket = os.getenv("S3_BUCKET")
            if not bucket:
                raise ValueError("S3_BUCKET environment variable is required for S3 storage")
            
            _storage_instance = S3Storage(
                bucket=bucket,
                region=os.getenv("S3_REGION", "us-east-1"),
                access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
        else:
            # Local storage - use project storage directory
            base_path = Path(os.getenv("LOCAL_STORAGE_PATH", "storage"))
            _storage_instance = LocalStorage(base_path)
    
    return _storage_instance


def reset_storage() -> None:
    """Reset storage singleton (useful for testing)."""
    global _storage_instance
    _storage_instance = None
