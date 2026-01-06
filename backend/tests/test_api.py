"""API integration tests."""
from pathlib import Path

from fastapi.testclient import TestClient


DATA_PATH = Path(__file__).resolve().parent.parent.parent / "examples" / "Data.csv"


def test_upload_and_preview(client: TestClient):
    """Test uploading a dataset and getting preview."""
    dataset_id = "test_upload"

    with DATA_PATH.open("rb") as f:
        resp = client.post(
            "/upload",
            data={"dataset_id": dataset_id},
            files={"file": ("Data.csv", f, "text/csv")},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dataset_id"] == dataset_id
    assert "preview" in body
    assert "row_count" in body
    assert "column_count" in body


def test_preview_endpoint(client: TestClient):
    """Test getting preview for existing dataset."""
    dataset_id = "test_preview"

    # First upload
    with DATA_PATH.open("rb") as f:
        client.post(
            "/upload",
            data={"dataset_id": dataset_id},
            files={"file": ("Data.csv", f, "text/csv")},
        )

    # Get preview
    resp = client.get(f"/preview/{dataset_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_id"] == dataset_id
    assert "preview" in body


def test_download_file(client: TestClient):
    """Test downloading processed file."""
    dataset_id = "test_download"

    # Upload first
    with DATA_PATH.open("rb") as f:
        client.post(
            "/upload",
            data={"dataset_id": dataset_id},
            files={"file": ("Data.csv", f, "text/csv")},
        )

    # Download file
    resp = client.get(f"/download/{dataset_id}/file")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")


def test_chat_no_data(client: TestClient):
    """Test chat when no data is loaded."""
    resp = client.post(
        "/chat/empty_dataset",
        json={"content": "hello"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "No data" in body["assistant_message"] or "Upload" in body["assistant_message"]


def test_preview_not_found(client: TestClient):
    """Test preview for non-existent dataset."""
    resp = client.get("/preview/nonexistent")
    assert resp.status_code == 404
