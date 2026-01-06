"""Tests for storage backends.

Tests both LocalStorage and S3Storage implementations.
"""
import io
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from services.storage import LocalStorage, S3Storage, get_storage, reset_storage, FileStats


class TestLocalStorage:
    """Tests for LocalStorage backend."""
    
    @pytest.fixture
    def storage(self, tmp_path):
        """Create a LocalStorage instance with temp directory."""
        return LocalStorage(tmp_path)
    
    @pytest.fixture
    def sample_csv_content(self):
        """Sample CSV content for testing."""
        return b"name,age,city\nAlice,30,NYC\nBob,25,LA\n"
    
    @pytest.mark.asyncio
    async def test_write_and_read_file(self, storage, sample_csv_content):
        """Test writing and reading a file."""
        path = "test/data.csv"
        
        await storage.write_file(path, sample_csv_content)
        content = await storage.read_file(path)
        
        assert content == sample_csv_content
    
    @pytest.mark.asyncio
    async def test_exists(self, storage, sample_csv_content):
        """Test file existence check."""
        path = "test/data.csv"
        
        assert not await storage.exists(path)
        await storage.write_file(path, sample_csv_content)
        assert await storage.exists(path)
    
    @pytest.mark.asyncio
    async def test_delete_file(self, storage, sample_csv_content):
        """Test file deletion."""
        path = "test/data.csv"
        
        await storage.write_file(path, sample_csv_content)
        assert await storage.exists(path)
        
        result = await storage.delete_file(path)
        assert result is True
        assert not await storage.exists(path)
        
        # Delete non-existent file
        result = await storage.delete_file(path)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_list_files(self, storage, sample_csv_content):
        """Test listing files with prefix."""
        await storage.write_file("raw/file1.csv", sample_csv_content)
        await storage.write_file("raw/file2.csv", sample_csv_content)
        await storage.write_file("curated/file3.csv", sample_csv_content)
        
        raw_files = await storage.list_files("raw")
        assert len(raw_files) == 2
        assert any("file1.csv" in f for f in raw_files)
        assert any("file2.csv" in f for f in raw_files)
    
    @pytest.mark.asyncio
    async def test_get_file_stats(self, storage, sample_csv_content):
        """Test getting file statistics."""
        path = "test/data.csv"
        
        # Non-existent file
        stats = await storage.get_file_stats(path)
        assert stats is None
        
        # Existing file
        await storage.write_file(path, sample_csv_content)
        stats = await storage.get_file_stats(path)
        
        assert stats is not None
        assert stats.exists is True
        assert stats.size_bytes == len(sample_csv_content)
    
    @pytest.mark.asyncio
    async def test_read_csv(self, storage, sample_csv_content):
        """Test reading CSV into DataFrame."""
        path = "test/data.csv"
        await storage.write_file(path, sample_csv_content)
        
        df = await storage.read_csv(path)
        
        assert len(df) == 2
        assert list(df.columns) == ["name", "age", "city"]
        assert df.iloc[0]["name"] == "Alice"
    
    @pytest.mark.asyncio
    async def test_write_csv(self, storage):
        """Test writing DataFrame to CSV."""
        path = "test/output.csv"
        df = pd.DataFrame({
            "name": ["Charlie", "Diana"],
            "score": [95, 88]
        })
        
        await storage.write_csv(path, df)
        
        # Read it back
        result_df = await storage.read_csv(path)
        assert len(result_df) == 2
        assert result_df.iloc[0]["name"] == "Charlie"
    
    def test_get_local_path(self, storage):
        """Test getting local path."""
        path = storage.get_local_path("test/data.csv")
        assert path is not None
        assert isinstance(path, Path)


class TestS3Storage:
    """Tests for S3Storage backend (mocked)."""
    
    @pytest.fixture
    def mock_s3_client(self):
        """Create a mocked S3 client."""
        with patch("boto3.Session") as mock_session:
            mock_client = MagicMock()
            mock_session.return_value.client.return_value = mock_client
            yield mock_client
    
    @pytest.fixture
    def storage(self, mock_s3_client):
        """Create an S3Storage instance with mocked client."""
        return S3Storage(
            bucket="test-bucket",
            region="us-east-1",
            access_key_id="test-key",
            secret_access_key="test-secret",
        )
    
    @pytest.mark.asyncio
    async def test_write_file(self, storage, mock_s3_client):
        """Test writing file to S3."""
        content = b"test content"
        await storage.write_file("test/file.txt", content)
        
        mock_s3_client.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="test/file.txt",
            Body=content,
        )
    
    @pytest.mark.asyncio
    async def test_read_file(self, storage, mock_s3_client):
        """Test reading file from S3."""
        expected_content = b"test content"
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=expected_content))
        }
        
        content = await storage.read_file("test/file.txt")
        
        assert content == expected_content
        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="test/file.txt",
        )
    
    def test_get_local_path_returns_none(self, storage):
        """Test that S3 storage returns None for local path."""
        assert storage.get_local_path("test/file.txt") is None
    
    def test_generate_presigned_url(self, storage, mock_s3_client):
        """Test generating presigned URL."""
        mock_s3_client.generate_presigned_url.return_value = "https://s3.example.com/signed-url"
        
        url = storage.generate_presigned_url("test/file.txt", expires_in=300)
        
        assert "signed-url" in url
        mock_s3_client.generate_presigned_url.assert_called_once()


class TestStorageFactory:
    """Tests for storage factory function."""
    
    def setup_method(self):
        """Reset storage singleton before each test."""
        reset_storage()
    
    def teardown_method(self):
        """Reset storage singleton after each test."""
        reset_storage()
    
    def test_get_storage_default_local(self, tmp_path, monkeypatch):
        """Test default local storage."""
        monkeypatch.setenv("STORAGE_BACKEND", "local")
        monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
        
        storage = get_storage()
        
        assert isinstance(storage, LocalStorage)
    
    def test_get_storage_s3(self, monkeypatch):
        """Test S3 storage creation."""
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.setenv("S3_BUCKET", "test-bucket")
        monkeypatch.setenv("S3_REGION", "us-west-2")
        
        with patch("boto3.Session"):
            storage = get_storage()
            assert isinstance(storage, S3Storage)
    
    def test_get_storage_s3_missing_bucket(self, monkeypatch):
        """Test S3 storage fails without bucket."""
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.delenv("S3_BUCKET", raising=False)
        
        with pytest.raises(ValueError, match="S3_BUCKET"):
            get_storage()
