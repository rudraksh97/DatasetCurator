"""Centralized application configuration.

This module provides a single source of truth for all configurable values,
loaded from environment variables with sensible defaults.

Usage:
    from config import settings
    print(settings.llm.model)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _get_env(key: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.getenv(key, default)


def _get_env_int(key: str, default: int) -> int:
    """Get integer environment variable with default."""
    return int(os.getenv(key, str(default)))


def _get_env_float(key: str, default: float) -> float:
    """Get float environment variable with default."""
    return float(os.getenv(key, str(default)))


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean environment variable with default."""
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


def _get_env_list(key: str, default: str, separator: str = ",") -> List[str]:
    """Get list environment variable with default."""
    value = os.getenv(key, default)
    return [item.strip() for item in value.split(separator) if item.strip()]


@dataclass(frozen=True)
class LLMSettings:
    """LLM provider configuration."""
    api_key: str = field(default_factory=lambda: _get_env("OPENROUTER_API_KEY"))
    base_url: str = field(default_factory=lambda: _get_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    default_model: str = field(default_factory=lambda: _get_env("DEFAULT_LLM_MODEL", "mistralai/mistral-7b-instruct:free"))
    default_temperature: float = 0.7
    default_max_tokens: int = 2048
    max_tool_iterations: int = field(default_factory=lambda: _get_env_int("MAX_TOOL_ITERATIONS", 10))


@dataclass(frozen=True)
class EmbeddingSettings:
    """Embedding model configuration."""
    model: str = field(default_factory=lambda: _get_env("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    dimension: int = field(default_factory=lambda: _get_env_int("EMBEDDING_DIM", 384))


@dataclass(frozen=True)
class DataLoaderSettings:
    """Data loading configuration for handling large files."""
    large_file_size_mb: int = field(default_factory=lambda: _get_env_int("LARGE_FILE_SIZE_MB", 100))
    large_row_count: int = field(default_factory=lambda: _get_env_int("LARGE_ROW_COUNT", 1_000_000))
    sample_size: int = field(default_factory=lambda: _get_env_int("DATA_SAMPLE_SIZE", 10_000))
    chunk_size: int = field(default_factory=lambda: _get_env_int("DATA_CHUNK_SIZE", 50_000))
    search_result_limit: int = field(default_factory=lambda: _get_env_int("SEARCH_RESULT_LIMIT", 5))


@dataclass(frozen=True)
class StorageSettings:
    """File storage configuration.
    
    Supports two backends:
    - local: Uses local filesystem (for development)
    - s3: Uses AWS S3 (for production on Render)
    """
    backend: str = field(default_factory=lambda: _get_env("STORAGE_BACKEND", "local"))
    
    # Local storage paths (used when backend="local")
    raw_path: Path = field(default_factory=lambda: Path(_get_env("RAW_STORAGE_PATH", "storage/raw")))
    curated_path: Path = field(default_factory=lambda: Path(_get_env("CURATED_STORAGE_PATH", "storage/curated")))
    
    # S3 configuration (used when backend="s3")
    s3_bucket: str = field(default_factory=lambda: _get_env("S3_BUCKET", ""))
    s3_region: str = field(default_factory=lambda: _get_env("S3_REGION", "us-east-1"))
    aws_access_key_id: str = field(default_factory=lambda: _get_env("AWS_ACCESS_KEY_ID", ""))
    aws_secret_access_key: str = field(default_factory=lambda: _get_env("AWS_SECRET_ACCESS_KEY", ""))
    
    # Prefixes for raw and curated data in S3
    raw_prefix: str = "raw"
    curated_prefix: str = "curated"
    
    def __post_init__(self):
        """Ensure storage directories exist for local backend."""
        if self.backend == "local":
            object.__setattr__(self, 'raw_path', Path(self.raw_path))
            object.__setattr__(self, 'curated_path', Path(self.curated_path))
            self.raw_path.mkdir(parents=True, exist_ok=True)
            self.curated_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_s3(self) -> bool:
        """Check if using S3 backend."""
        return self.backend.lower() == "s3"
    
    def validate_s3(self) -> None:
        """Validate S3 configuration.
        
        Raises:
            ValueError: If S3 is configured but bucket is missing.
        """
        if self.is_s3 and not self.s3_bucket:
            raise ValueError(
                "S3_BUCKET environment variable is required when STORAGE_BACKEND=s3"
            )


@dataclass(frozen=True)
class WorkflowSettings:
    """Workflow execution configuration."""
    require_approval: bool = field(default_factory=lambda: _get_env_bool("REQUIRE_APPROVAL_FOR_DESTRUCTIVE_OPS", False))
    approval_row_threshold: int = field(default_factory=lambda: _get_env_int("APPROVAL_ROW_THRESHOLD", 1000))
    default_max_retries: int = 1


@dataclass(frozen=True)
class CORSSettings:
    """CORS configuration."""
    allowed_origins: List[str] = field(
        default_factory=lambda: _get_env_list("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    )


@dataclass(frozen=True)
class DatabaseSettings:
    """Database configuration."""
    url: str = field(default_factory=lambda: _get_env("DATABASE_URL"))
    
    def validate(self) -> None:
        """Validate database configuration.
        
        Raises:
            ValueError: If DATABASE_URL is not set.
        """
        if not self.url:
            raise ValueError(
                "DATABASE_URL environment variable is required. "
                "Example: postgresql+asyncpg://user:password@localhost:5432/dbname"
            )
    
    @property
    def is_configured(self) -> bool:
        """Check if database is configured without raising."""
        return bool(self.url)


@dataclass(frozen=True)
class Settings:
    """Application settings container."""
    llm: LLMSettings = field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    data_loader: DataLoaderSettings = field(default_factory=DataLoaderSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    workflow: WorkflowSettings = field(default_factory=WorkflowSettings)
    cors: CORSSettings = field(default_factory=CORSSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()


# Convenience alias for direct import
settings = get_settings()
