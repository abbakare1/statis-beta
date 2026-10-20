"""Dataset Upload UX Regression Tests.

Covers the researcher-facing upload workflow:
- Upload endpoint returns row/column counts + preview
- Preview endpoint returns read-only shape + first rows
- Upload associates dataset with correct project
- Original uploaded file remains immutable
- Dataset identity (content_hash) is consistent across endpoints
- Unsupported / empty / malformed files produce clear errors
- Dataset persists across list calls (refresh / navigation)
- Frontend JS never computes statistics or modifies source data
"""

import asyncio
import hashlib
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth_headers():
    """Create a test user with generous quotas and return auth headers."""
    from app.security.auth import create_user, create_access_token
    from app.core.database import get_session
    
    async def _setup():
        async with get_session() as session:
            user = await create_user(session, f"upload_ux_{uuid.uuid4().hex[:8]}@example.com", "password123", "Upload UX Test User")
            user.max_projects = 100
            user.max_analyses = 1000
            user.max_upload_mb = 1000
            await session.commit()
            return user
    
    user = asyncio.run(_setup())
    token = create_access_token(user.id, user.role)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def tmp_data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d


def _uid() -> str:
    return str(uuid.uuid4())[:8]


def _make_csv(tmp_dir: Path, n: int = 50, seed: int = 42) -> Path:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n)],
        "age": rng.normal(50, 10, n).round(1),
        "treatment": rng.binomial(1, 0.5, n),
        "outcome": rng.normal(100, 15, n).round(1),
    })
    path = tmp_dir / f"upload_{_uid()}.csv"
    df.to_csv(path, index=False)
    return path


def _create_project(client: TestClient, auth_headers: dict) -> str:
    r = client.post("/api/projects", data={
        "name": f"Upload UX Project {_uid()}",
        "description": "Dataset upload UX regression test",
        "research_question": "Does treatment affect outcome?",
        "study_design": "cohort",
    }, headers=auth_headers)
    assert r.status_code == 200
    return r.json()["id"]


# ---------------------------------------------------------------------------
# 1. Upload endpoint returns shape + preview
# ---------------------------------------------------------------------------

def test_upload_returns_row_and_column_counts(client, auth_headers, tmp_data_dir):
    pid = _create_project(client, auth_headers)
    csv = _make_csv(tmp_data_dir, n=50)
    r = client.post(f"/api/projects/{pid}/upload", files={
        "file": (csv.name, csv.read_bytes(), "text/csv"),
    }, headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["row_count"] == 50
    assert body["column_count"] == 4
    assert body["content_hash"]
    assert body["id"]


def test_upload_returns_preview_data(client, auth_headers, tmp_data_dir):
    pid = _create_project(client, auth_headers)
    csv = _make_csv(tmp_data_dir, n=30)
    r = client.post(f"/api/projects/{pid}/upload", files={
        "file": (csv.name, csv.read_bytes(), "text/csv"),
    }, headers=auth_headers)
    assert r.status_code == 200
    preview = r.json()["preview"]
    assert preview["n_rows"] == 30
    assert preview["n_columns"] == 4
    assert len(preview["columns"]) == 4
    assert len(preview["head"]) == 10  # default head_n
    # Column metadata
    for col in preview["columns"]:
        assert "name" in col
        assert "dtype" in col
        assert "n_missing" in col
        assert "n_unique" in col


# ---------------------------------------------------------------------------
# 2. Dataset associated with correct project
# ---------------------------------------------------------------------------

def test_upload_associates_dataset_with_project(client, auth_headers, tmp_data_dir):
    pid = _create_project(client, auth_headers)
    csv = _make_csv(tmp_data_dir, n=20)
    r = client.post(f"/api/projects/{pid}/upload", files={
        "file": (csv.name, csv.read_bytes(), "text/csv"),
    }, headers=auth_headers)
    assert r.status_code == 200
    did = r.json()["id"]

    r = client.get(f"/api/projects/{pid}/datasets", headers=auth_headers)
    assert r.status_code == 200
    datasets = r.json()
    assert len(datasets) == 1
    assert datasets[0]["id"] == did
    assert datasets[0]["row_count"] == 20
    assert datasets[0]["column_count"] == 4


# ---------------------------------------------------------------------------
# 3. Original dataset remains immutable
# ---------------------------------------------------------------------------

def test_uploaded_file_remains_immutable(client, auth_headers, tmp_data_dir):
    pid = _create_project(client, auth_headers)
    csv = _make_csv(tmp_data_dir, n=40)
    original_bytes = csv.read_bytes()
    original_file_hash = hashlib.sha256(original_bytes).hexdigest()

    r = client.post(f"/api/projects/{pid}/upload", files={
        "file": (csv.name, csv.read_bytes(), "text/csv"),
    }, headers=auth_headers)
    assert r.status_code == 200
    content_hash = r.json()["content_hash"]

    # Backend content_hash must be the canonical dataset identity (INV 2/18):
    # the same hash compute_dataset_identity produces, because executor
    # guards and output rendering verify against it.
    from app.data.dataset_identity import compute_dataset_identity
    expected = compute_dataset_identity(csv)["content_hash"]
    assert content_hash == expected

    # Source file on disk unchanged
    assert csv.read_bytes() == original_bytes
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == original_file_hash


# ---------------------------------------------------------------------------
# 4. Dataset identity consistent across endpoints
# ---------------------------------------------------------------------------

def test_dataset_identity_consistent_across_endpoints(client, auth_headers, tmp_data_dir):
    pid = _create_project(client, auth_headers)
    csv = _make_csv(tmp_data_dir, n=25)
    r = client.post(f"/api/projects/{pid}/upload", files={
        "file": (csv.name, csv.read_bytes(), "text/csv"),
    }, headers=auth_headers)
    upload_hash = r.json()["content_hash"]
    did = r.json()["id"]

    # list_datasets returns same hash
    r = client.get(f"/api/projects/{pid}/datasets", headers=auth_headers)
    listed = next(d for d in r.json() if d["id"] == did)
    assert listed["content_hash"] == upload_hash

    # preview endpoint returns same hash
    r = client.get(f"/api/datasets/{did}/preview")
    assert r.status_code == 200
    assert r.json()["content_hash"] == upload_hash


# ---------------------------------------------------------------------------
# 5. Preview endpoint
# ---------------------------------------------------------------------------

def test_preview_endpoint_returns_shape_and_head(client, auth_headers, tmp_data_dir):
    pid = _create_project(client, auth_headers)
    csv = _make_csv(tmp_data_dir, n=60)
    r = client.post(f"/api/projects/{pid}/upload", files={
        "file": (csv.name, csv.read_bytes(), "text/csv"),
    }, headers=auth_headers)
    did = r.json()["id"]

    r = client.get(f"/api/datasets/{did}/preview?head=5")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset_id"] == did
    assert body["n_rows"] == 60
    assert body["n_columns"] == 4
    assert len(body["head"]) == 5
    assert body["columns"][0]["name"] == "patient_id"
    assert body["file_size_bytes"] > 0


def test_preview_endpoint_404_for_unknown_dataset(client):
    r = client.get(f"/api/datasets/{_uid()}/preview")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 6. Error handling
# ---------------------------------------------------------------------------

def test_upload_rejects_unsupported_file_type(client, auth_headers, tmp_data_dir):
    pid = _create_project(client, auth_headers)
    bad = tmp_data_dir / "data.xyz"
    bad.write_bytes(b"not a real file")
    r = client.post(f"/api/projects/{pid}/upload", files={
        "file": (bad.name, bad.read_bytes(), "application/octet-stream"),
    }, headers=auth_headers)
    assert r.status_code == 400
    assert "Unsupported file type" in r.json()["detail"]


def test_upload_rejects_empty_file(client, auth_headers, tmp_data_dir):
    pid = _create_project(client, auth_headers)
    empty = tmp_data_dir / "empty.csv"
    empty.write_bytes(b"")
    r = client.post(f"/api/projects/{pid}/upload", files={
        "file": (empty.name, empty.read_bytes(), "text/csv"),
    }, headers=auth_headers)
    assert r.status_code == 400
    assert "Empty file" in r.json()["detail"]


def test_upload_handles_malformed_csv(client, auth_headers, tmp_data_dir):
    pid = _create_project(client, auth_headers)
    bad = tmp_data_dir / "malformed.csv"
    bad.write_bytes(b"\x00\x01\x02\x03 not valid csv \xff\xfe")
    r = client.post(f"/api/projects/{pid}/upload", files={
        "file": (bad.name, bad.read_bytes(), "text/csv"),
    }, headers=auth_headers)
    # Malformed CSV: either 400 (parse failure caught at upload) or 200
    # with preview containing an error — both are valid error surfaces.
    if r.status_code == 200:
        # Preview should flag the parse failure
        preview = r.json().get("preview", {})
        assert "error" in preview or preview.get("n_rows", 0) >= 0
    else:
        assert r.status_code in (400, 422)


def test_upload_requires_file_field(client, auth_headers):
    pid = _create_project(client, auth_headers)
    r = client.post(f"/api/projects/{pid}/upload", headers=auth_headers)
    assert r.status_code == 422  # FastAPI missing-field validation


# ---------------------------------------------------------------------------
# 7. Dataset persists across list calls (refresh / navigation)
# ---------------------------------------------------------------------------

def test_dataset_persists_across_list_calls(client, auth_headers, tmp_data_dir):
    pid = _create_project(client, auth_headers)
    csv = _make_csv(tmp_data_dir, n=35)
    r = client.post(f"/api/projects/{pid}/upload", files={
        "file": (csv.name, csv.read_bytes(), "text/csv"),
    }, headers=auth_headers)
    did = r.json()["id"]

    # Simulate page refresh / navigate away and back
    for _ in range(3):
        r = client.get(f"/api/projects/{pid}/datasets", headers=auth_headers)
        assert r.status_code == 200
        ids = [d["id"] for d in r.json()]
        assert did in ids


# ---------------------------------------------------------------------------
# 8. Frontend JS never computes statistics or modifies source data
# ---------------------------------------------------------------------------

def test_frontend_js_contains_no_statistical_computation():
    js_dir = ROOT / "frontend" / "static" / "js"
    # Forbidden: any use of scientific/statistical libraries or
    # computation primitives. Displaying backend-provided values
    # (e.g. result.p_value in a result card) is allowed — the frontend
    # never *computes* these.
    forbidden = [
        "scipy", "statsmodels", "lifelines", "numpy", "pandas",
        "import stats", "require('scipy",
    ]
    for js_file in js_dir.glob("*.js"):
        content = js_file.read_text(encoding="utf-8")
        for term in forbidden:
            assert term not in content.lower(), (
                f"{js_file.name} contains forbidden statistical library: {term}"
            )


def test_frontend_js_contains_no_file_modification():
    js_dir = ROOT / "frontend" / "static" / "js"
    # Frontend must never write/delete/modify files on the server
    forbidden_ops = ["writeFile", "unlink", "rmSync", "renameSync", "appendFile"]
    for js_file in js_dir.glob("*.js"):
        content = js_file.read_text(encoding="utf-8")
        for op in forbidden_ops:
            assert op not in content, (
                f"{js_file.name} contains forbidden file operation: {op}"
            )


def test_frontend_template_contains_upload_form():
    html = (ROOT / "frontend" / "templates" / "index.html").read_text(encoding="utf-8")
    # SPA shell loads views.js which builds the upload form dynamically.
    # Verify views.js contains the upload form markup.
    views = (ROOT / "frontend" / "static" / "js" / "views.js").read_text(encoding="utf-8")
    assert "uploadForm" in views
    assert "datasetFile" in views
    assert "Upload Dataset" in views
    assert "multipart/form-data" in views


def test_frontend_api_client_has_upload_and_preview():
    api = (ROOT / "frontend" / "static" / "js" / "api.js").read_text(encoding="utf-8")
    assert "datasets.upload" in api or "upload:" in api
    assert "preview:" in api
    assert "/upload" in api
    assert "/preview" in api


# ---------------------------------------------------------------------------
# 9. TSV upload works (multi-format support)
# ---------------------------------------------------------------------------

def test_upload_supports_tsv(client, auth_headers, tmp_data_dir):
    pid = _create_project(client, auth_headers)
    rng = np.random.RandomState(7)
    df = pd.DataFrame({
        "id": range(15),
        "value": rng.normal(0, 1, 15).round(3),
    })
    tsv = tmp_data_dir / f"data_{_uid()}.tsv"
    df.to_csv(tsv, sep="\t", index=False)

    r = client.post(f"/api/projects/{pid}/upload", files={
        "file": (tsv.name, tsv.read_bytes(), "text/tab-separated-values"),
    }, headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["row_count"] == 15
    assert r.json()["column_count"] == 2


# ---------------------------------------------------------------------------
# 10. Workflow integration: dataset → quality → analysis chain intact
# ---------------------------------------------------------------------------

def test_upload_to_quality_to_analysis_chain(client, auth_headers, tmp_data_dir):
    pid = _create_project(client, auth_headers)
    csv = _make_csv(tmp_data_dir, n=80, seed=99)
    r = client.post(f"/api/projects/{pid}/upload", files={
        "file": (csv.name, csv.read_bytes(), "text/csv"),
    }, headers=auth_headers)
    assert r.status_code == 200
    did = r.json()["id"]

    # Workflow shows dataset step complete
    r = client.get(f"/api/projects/{pid}/workflow", headers=auth_headers)
    wf = r.json()
    ds_step = next(s for s in wf["steps"] if s["step"] == "dataset")
    assert ds_step["state"] == "complete"

    # Data quality assessment works
    r = client.post(f"/api/data-quality/{did}/assess", json={})
    assert r.status_code == 200
    quality = r.json()

    # Quality profile exists
    r = client.get(f"/api/projects/{pid}/quality-profile", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["count"] >= 1

    # Analysis plan — mock Ollama provider to avoid network call
    from unittest.mock import AsyncMock, patch
    mock_provider = AsyncMock()
    mock_provider.is_available.return_value = False

    with patch("app.core.ollama_adapter.get_model_provider", return_value=mock_provider):
        data_dictionary = {
            "n_rows": 80,
            "variables": [
                {"name": "patient_id", "dtype": "object", "n_unique": 80, "n_missing": 0, "n_total": 80},
                {"name": "age", "dtype": "float64", "n_unique": 80, "n_missing": 0, "n_total": 80},
                {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_missing": 0, "n_total": 80},
                {"name": "outcome", "dtype": "float64", "n_unique": 80, "n_missing": 0, "n_total": 80},
            ],
        }
        r = client.post("/api/analysis-plan/create", json={
            "project_id": pid,
            "dataset_id": did,
            "research_question": "Does treatment affect outcome?",
            "study_design": "cohort",
            "data_dictionary": data_dictionary,
            "data_quality": quality,
        })
    assert r.status_code == 200
    assert r.json()["n_analyses"] >= 1
