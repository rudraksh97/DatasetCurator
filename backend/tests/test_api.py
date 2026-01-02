from pathlib import Path

from fastapi.testclient import TestClient


DATA_PATH = Path(__file__).resolve().parent.parent / "Data.csv"


def test_api_end_to_end(client: TestClient):
    dataset_id = "integration1"

    with DATA_PATH.open("rb") as f:
        resp = client.post(
            "/upload",
            data={"dataset_id": dataset_id},
            files={"file": ("Data.csv", f, "text/csv")},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dataset_id"] == dataset_id

    # Health after upload
    resp = client.get(f"/health/{dataset_id}")
    assert resp.status_code == 200
    assert "issues" in resp.json()

    # Download endpoints should be ready post-upload (auto-clean)
    resp = client.get(f"/download/{dataset_id}")
    assert resp.status_code == 200
    assert "curated_path" in resp.json()

    resp = client.get(f"/download/{dataset_id}/file")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
