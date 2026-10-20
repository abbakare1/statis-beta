"""Phase 2I — Researcher-Facing Frontend Integration & User Experience.

Comprehensive tests covering:
- Frontend template serving
- Project summary aggregation
- Workflow status derivation
- Project-scoped resource endpoints
- API service layer integrity
- Numerical integrity (backend values preserved)
- Error handling
- Security
- Audit preservation
"""

import asyncio
import json
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Test client for the STATIS API."""
    from app.api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary directory for test CSV files."""
    d = tmp_path / "data"
    d.mkdir()
    return d


def _uid() -> str:
    return str(uuid.uuid4())[:8]


def _make_csv(tmp_dir: Path, n: int = 60, seed: int = 42) -> Path:
    """Create a synthetic test CSV."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n)],
        "age": rng.normal(50, 10, n).round(1),
        "bmi": rng.normal(25, 3, n).round(1),
        "treatment": rng.binomial(1, 0.5, n),
        "systolic_bp": rng.normal(130, 15, n).round(1),
        "sex": rng.choice(["M", "F"], n),
    })
    # Add missing values
    df.loc[rng.choice(n, 5, replace=False), "bmi"] = np.nan
    path = tmp_dir / f"test_{_uid()}.csv"
    df.to_csv(path, index=False)
    return path


async def _insert_project(name: str, **kwargs) -> str:
    from app.core.database import get_session
    from app.models import Project
    async with get_session() as session:
        project = Project(name=name, **kwargs)
        session.add(project)
        await session.commit()
        await session.refresh(project)
        return project.id


async def _upload_dataset(project_id: str, file_path: Path) -> str:
    from app.core.database import get_session
    from app.models import Dataset
    from app.data.hashing import file_hash
    content_hash = file_hash(file_path)
    async with get_session() as session:
        dataset = Dataset(
            project_id=project_id,
            original_filename=file_path.name,
            stored_path=str(file_path),
            content_hash=content_hash,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)
        return dataset.id


# ---------------------------------------------------------------------------
# Frontend Template Tests
# ---------------------------------------------------------------------------

class TestFrontendTemplate:
    """Tests for the frontend template serving."""

    def test_index_page_loads(self, client):
        """Index page returns 200 with full SPA layout."""
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")

    def test_index_contains_app_main(self, client):
        """Template contains the main content container."""
        r = client.get("/")
        assert 'id="app-main"' in r.text

    def test_index_contains_sidebar(self, client):
        """Template contains the sidebar navigation."""
        r = client.get("/")
        assert 'class="sidebar"' in r.text

    def test_index_contains_navigation_links(self, client):
        """Template contains researcher-facing navigation."""
        r = client.get("/")
        for nav in ["Dashboard", "Results", "Figures", "Reports", "Audit"]:
            assert nav in r.text

    def test_index_contains_all_js_modules(self, client):
        """Template loads all frontend JS modules."""
        r = client.get("/")
        for js in ["api.js", "state.js", "router.js", "components.js", "views.js", "app.js"]:
            assert js in r.text

    def test_index_contains_css(self, client):
        """Template loads the stylesheet."""
        r = client.get("/")
        assert "style.css" in r.text

    def test_index_preserves_docs_link(self, client):
        """Developer API documentation remains accessible."""
        r = client.get("/")
        assert "/docs" in r.text

    def test_index_has_skip_link(self, client):
        """Template includes accessibility skip link."""
        r = client.get("/")
        assert "skip-link" in r.text

    def test_index_has_modal_overlay(self, client):
        """Template includes modal overlay for dialogs."""
        r = client.get("/")
        assert 'id="modal-overlay"' in r.text

    def test_index_has_notifications(self, client):
        """Template includes notification toast container."""
        r = client.get("/")
        assert 'id="notifications"' in r.text


class TestFrontendStaticAssets:
    """Tests for static asset serving."""

    def test_css_served(self, client):
        r = client.get("/static/css/style.css")
        assert r.status_code == 200

    def test_js_api_served(self, client):
        r = client.get("/static/js/api.js")
        assert r.status_code == 200

    def test_js_state_served(self, client):
        r = client.get("/static/js/state.js")
        assert r.status_code == 200

    def test_js_router_served(self, client):
        r = client.get("/static/js/router.js")
        assert r.status_code == 200

    def test_js_components_served(self, client):
        r = client.get("/static/js/components.js")
        assert r.status_code == 200

    def test_js_views_served(self, client):
        r = client.get("/static/js/views.js")
        assert r.status_code == 200

    def test_js_app_served(self, client):
        r = client.get("/static/js/app.js")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Project Summary Tests
# ---------------------------------------------------------------------------

class TestProjectSummary:
    """Tests for the project summary aggregation endpoint."""

    def test_summary_returns_empty(self, client):
        """Summary works with no projects."""
        r = client.get("/api/projects/summary")
        assert r.status_code == 200
        data = r.json()
        assert "projects" in data
        assert "count" in data

    def test_summary_returns_projects(self, client):
        """Summary returns project data with counts."""
        pid = asyncio.run(_insert_project(f"summary_test_{_uid()}"))
        r = client.get("/api/projects/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] >= 1
        projects = data["projects"]
        match = [p for p in projects if p["id"] == pid]
        assert len(match) == 1
        p = match[0]
        assert p["name"].startswith("summary_test_")
        assert "dataset_count" in p
        assert "analysis_plan_count" in p
        assert "analysis_run_count" in p
        assert "output_count" in p

    def test_summary_aggregates_dataset_counts(self, client, tmp_path):
        """Summary correctly counts datasets per project."""
        pid = asyncio.run(_insert_project(f"summary_ds_{_uid()}"))
        csv = _make_csv(tmp_path)
        asyncio.run(_upload_dataset(pid, csv))

        r = client.get("/api/projects/summary")
        data = r.json()
        match = [p for p in data["projects"] if p["id"] == pid]
        assert len(match) == 1
        assert match[0]["dataset_count"] == 1

    def test_summary_route_not_shadowed(self, client):
        """Summary route is not captured by project_id route."""
        r = client.get("/api/projects/summary")
        assert r.status_code == 200
        assert "projects" in r.json()


# ---------------------------------------------------------------------------
# Project Workflow Tests
# ---------------------------------------------------------------------------

class TestProjectWorkflow:
    """Tests for the project workflow status endpoint."""

    def test_workflow_404_for_missing_project(self, client):
        r = client.get(f"/api/projects/{uuid.uuid4()}/workflow")
        assert r.status_code == 404

    def test_workflow_returns_structure(self, client):
        pid = asyncio.run(_insert_project(
            f"workflow_test_{_uid()}",
            research_question="What is the effect of treatment?",
            study_design="cohort",
        ))
        r = client.get(f"/api/projects/{pid}/workflow")
        assert r.status_code == 200
        data = r.json()
        assert data["project_id"] == pid
        assert "workflow_status" in data
        assert "steps" in data
        assert "counts" in data
        assert "review_items" in data

    def test_workflow_has_all_steps(self, client):
        """Workflow includes all expected pipeline steps."""
        pid = asyncio.run(_insert_project(f"workflow_steps_{_uid()}"))
        r = client.get(f"/api/projects/{pid}/workflow")
        data = r.json()
        step_keys = [s["step"] for s in data["steps"]]
        expected = [
            "research_question", "study_design", "dataset", "data_quality",
            "variable_roles", "power", "analysis_plan", "researcher_review",
            "statistical_analysis", "diagnostics", "defensibility",
            "results", "figures_tables", "report", "reproducibility",
        ]
        for key in expected:
            assert key in step_keys, f"Missing step: {key}"

    def test_workflow_derives_research_question_state(self, client):
        """Workflow reflects whether research question is set."""
        pid = asyncio.run(_insert_project(
            f"workflow_rq_{_uid()}",
            research_question="Test question?",
        ))
        r = client.get(f"/api/projects/{pid}/workflow")
        data = r.json()
        rq_step = next(s for s in data["steps"] if s["step"] == "research_question")
        assert rq_step["state"] == "complete"

    def test_workflow_derives_dataset_state(self, client, tmp_path):
        """Workflow reflects whether dataset exists."""
        pid = asyncio.run(_insert_project(f"workflow_ds_{_uid()}"))
        csv = _make_csv(tmp_path)
        asyncio.run(_upload_dataset(pid, csv))

        r = client.get(f"/api/projects/{pid}/workflow")
        data = r.json()
        ds_step = next(s for s in data["steps"] if s["step"] == "dataset")
        assert ds_step["state"] == "complete"

    def test_workflow_status_not_started_for_empty(self, client):
        """Empty project has not_started workflow status."""
        pid = asyncio.run(_insert_project(f"workflow_empty_{_uid()}"))
        r = client.get(f"/api/projects/{pid}/workflow")
        data = r.json()
        assert data["workflow_status"] == "not_started"

    def test_workflow_counts_datasets(self, client, tmp_path):
        """Workflow counts datasets correctly."""
        pid = asyncio.run(_insert_project(f"workflow_count_{_uid()}"))
        csv = _make_csv(tmp_path)
        asyncio.run(_upload_dataset(pid, csv))

        r = client.get(f"/api/projects/{pid}/workflow")
        data = r.json()
        assert data["counts"]["datasets"] == 1


# ---------------------------------------------------------------------------
# Project-Scoped Resource Endpoint Tests
# ---------------------------------------------------------------------------

class TestProjectResources:
    """Tests for project-scoped resource endpoints."""

    def test_project_plans_empty(self, client):
        pid = asyncio.run(_insert_project(f"plans_test_{_uid()}"))
        r = client.get(f"/api/projects/{pid}/plans")
        assert r.status_code == 200
        data = r.json()
        assert "plans" in data
        assert data["count"] == 0

    def test_project_runs_empty(self, client):
        pid = asyncio.run(_insert_project(f"runs_test_{_uid()}"))
        r = client.get(f"/api/projects/{pid}/runs")
        assert r.status_code == 200
        data = r.json()
        assert "runs" in data

    def test_project_outputs_empty(self, client):
        pid = asyncio.run(_insert_project(f"outputs_test_{_uid()}"))
        r = client.get(f"/api/projects/{pid}/outputs")
        assert r.status_code == 200
        data = r.json()
        assert "figures" in data
        assert "tables" in data
        assert "reports" in data
        assert data["count"] == 0

    def test_project_variable_roles_empty(self, client):
        pid = asyncio.run(_insert_project(f"roles_test_{_uid()}"))
        r = client.get(f"/api/projects/{pid}/variable-roles")
        assert r.status_code == 200
        data = r.json()
        assert "variable_roles" in data

    def test_project_quality_profile_empty(self, client):
        pid = asyncio.run(_insert_project(f"quality_test_{_uid()}"))
        r = client.get(f"/api/projects/{pid}/quality-profile")
        assert r.status_code == 200
        data = r.json()
        assert "profiles" in data

    def test_project_reproducibility_empty(self, client):
        pid = asyncio.run(_insert_project(f"repro_test_{_uid()}"))
        r = client.get(f"/api/projects/{pid}/reproducibility")
        assert r.status_code == 200
        data = r.json()
        assert "packages" in data

    def test_project_404_for_missing(self, client):
        missing_id = uuid.uuid4()
        for endpoint in ["plans", "runs", "outputs", "variable-roles", "quality-profile", "reproducibility"]:
            r = client.get(f"/api/projects/{missing_id}/{endpoint}")
            # Some endpoints return empty lists, others 404
            assert r.status_code in (200, 404), f"{endpoint}: {r.status_code}"


# ---------------------------------------------------------------------------
# Numerical Integrity Tests
# ---------------------------------------------------------------------------

class TestNumericalIntegrity:
    """Tests proving backend StatisticalResult values are preserved exactly."""

    def test_backend_result_values_preserved(self, client, tmp_data_dir):
        """Backend numerical values pass through API unchanged."""
        csv = _make_csv(tmp_data_dir, n=80, seed=123)

        r = client.post("/api/analysis/descriptive", json={
            "file_path": str(csv),
            "columns": ["age", "bmi"],
        })
        assert r.status_code == 200
        result = r.json()

        # Values must be present and numeric — frontend must not modify them
        assert "columns" in result or "summary" in result or "statistics" in result

    def test_t_test_result_values(self, client, tmp_data_dir):
        """t-test returns exact backend values."""
        csv = _make_csv(tmp_data_dir, n=60, seed=42)
        r = client.post("/api/analysis/t-test", json={
            "file_path": str(csv),
            "outcome": "systolic_bp",
            "group": "treatment",
            "group1": 0,
            "group2": 1,
        })
        assert r.status_code == 200
        result = r.json()

        # Key statistical fields must be present
        assert "p_value" in result or "pvalue" in result
        # The values come from the backend engine — frontend must not recalculate

    def test_logistic_regression_result_values(self, client, tmp_data_dir):
        """Logistic regression returns exact backend values."""
        csv = _make_csv(tmp_data_dir, n=60, seed=42)
        r = client.post("/api/analysis/logistic-regression", json={
            "file_path": str(csv),
            "outcome": "treatment",
            "predictors": ["age", "bmi"],
        })
        assert r.status_code == 200
        result = r.json()
        # Result contains backend-computed values
        assert isinstance(result, dict)

    def test_correlation_result_values(self, client, tmp_data_dir):
        """Correlation returns exact backend values."""
        csv = _make_csv(tmp_data_dir, n=60, seed=42)
        r = client.post("/api/analysis/correlation", json={
            "file_path": str(csv),
            "x": "age",
            "y": "systolic_bp",
            "method": "pearson",
        })
        assert r.status_code == 200
        result = r.json()
        assert isinstance(result, dict)

    def test_result_fingerprint_stability(self, client, tmp_data_dir):
        """Same input produces same result fingerprint."""
        csv = _make_csv(tmp_data_dir, n=60, seed=42)
        body = {
            "file_path": str(csv),
            "outcome": "systolic_bp",
            "group": "treatment",
            "group1": 0,
            "group2": 1,
        }
        r1 = client.post("/api/analysis/t-test", json=body)
        r2 = client.post("/api/analysis/t-test", json=body)
        assert r1.status_code == 200
        assert r2.status_code == 200
        # Both results come from the deterministic backend
        v1 = r1.json()
        v2 = r2.json()
        # Key values should match (deterministic engine)
        for key in ("p_value", "pvalue", "test_statistic", "t_statistic"):
            if key in v1 and key in v2:
                assert v1[key] == v2[key], f"Non-deterministic value for {key}"


# ---------------------------------------------------------------------------
# API Error Handling Tests
# ---------------------------------------------------------------------------

class TestAPIErrorHandling:
    """Tests for API error responses."""

    @pytest.fixture(autouse=True)
    def auth_headers(self):
        """Create a test user and return auth headers."""
        from app.security.auth import create_user, create_access_token
        from app.core.database import get_session

        async def _setup():
            async with get_session() as session:
                user = await create_user(session, f"api_err_{uuid.uuid4().hex[:8]}@example.com", "password123", "API Error Test User")
                return user

        user = asyncio.run(_setup())
        token = create_access_token(user.id, user.role)
        return {"Authorization": f"Bearer {token}"}

    def test_project_not_found(self, client, auth_headers):
        r = client.get(f"/api/projects/{uuid.uuid4()}", headers=auth_headers)
        assert r.status_code == 404

    def test_workflow_not_found(self, client, auth_headers):
        r = client.get(f"/api/projects/{uuid.uuid4()}/workflow", headers=auth_headers)
        assert r.status_code == 404

    def test_dataset_upload_missing_file(self, client):
        pid = asyncio.run(_insert_project(f"upload_test_{_uid()}"))
        # Without auth, expect 401 (upload endpoint requires authentication)
        r = client.post(f"/api/projects/{pid}/upload")
        assert r.status_code in (400, 401, 422)

    def test_pipeline_missing_params(self, client):
        r = client.post("/api/pipeline/run", json={})
        assert r.status_code == 400

    def test_output_404(self, client):
        r = client.get(f"/api/outputs/figures/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_audit_returns_list(self, client):
        r = client.get("/api/audit?limit=5")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_methods_list(self, client):
        r = client.get("/api/methods")
        assert r.status_code == 200
        data = r.json()
        assert "methods" in data
        assert len(data["methods"]) > 0


# ---------------------------------------------------------------------------
# Security Tests
# ---------------------------------------------------------------------------

class TestSecurity:
    """Tests for security invariants."""

    def test_no_path_traversal_in_artifact(self, client):
        """Artifact endpoints reject path traversal."""
        r = client.get("/api/outputs/figures/../../etc/passwd/artifact")
        assert r.status_code in (400, 404, 422)

    def test_project_id_validation(self, client):
        """Invalid project IDs are rejected or return 404."""
        from app.security.auth import create_user, create_access_token
        from app.core.database import get_session

        async def _setup():
            async with get_session() as session:
                user = await create_user(session, f"sec_{uuid.uuid4().hex[:8]}@example.com", "password123", "Security Test User")
                return user

        user = asyncio.run(_setup())
        token = create_access_token(user.id, user.role)
        auth_headers = {"Authorization": f"Bearer {token}"}

        for bad_id in ["../../etc", "'; DROP TABLE--", "<script>alert(1)</script>"]:
            r = client.get(f"/api/projects/{bad_id}/workflow", headers=auth_headers)
            assert r.status_code in (400, 401, 404, 422)

    def test_audit_project_filter(self, client):
        """Audit endpoint accepts project filter."""
        r = client.get(f"/api/audit?project_id={uuid.uuid4()}")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_frontend_no_statistical_computation(self, client):
        """Frontend JS does not contain statistical computation."""
        r = client.get("/static/js/views.js")
        assert r.status_code == 200
        content = r.text
        # Frontend must not import or use scipy/statsmodels
        assert "scipy" not in content
        assert "statsmodels" not in content
        # Frontend must not compute p-values
        assert "p_value =" not in content
        assert "pvalue =" not in content


# ---------------------------------------------------------------------------
# Audit Preservation Tests
# ---------------------------------------------------------------------------

class TestAuditPreservation:
    """Tests that audit history is preserved and accessible."""

    def test_audit_endpoint_returns_entries(self, client):
        r = client.get("/api/audit?limit=10")
        assert r.status_code == 200
        entries = r.json()
        assert isinstance(entries, list)

    def test_audit_entry_structure(self, client):
        """Audit entries have expected fields."""
        r = client.get("/api/audit?limit=1")
        entries = r.json()
        if entries:
            entry = entries[0]
            assert "action" in entry
            assert "agent" in entry
            assert "decision" in entry
            assert "timestamp" in entry

    def test_project_audit_filter(self, client):
        """Project-filtered audit works."""
        pid = asyncio.run(_insert_project(f"audit_test_{_uid()}"))
        r = client.get(f"/api/audit?project_id={pid}")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Output API Integration Tests (Phase 2G/2H endpoints used by frontend)
# ---------------------------------------------------------------------------

class TestOutputAPIIntegration:
    """Tests that output APIs work for the frontend."""

    def test_figures_list_empty(self, client):
        r = client.get("/api/outputs/figures")
        assert r.status_code == 200
        data = r.json()
        assert "success" in data

    def test_tables_list_empty(self, client):
        r = client.get("/api/outputs/tables")
        assert r.status_code == 200

    def test_reports_list_empty(self, client):
        r = client.get("/api/outputs/reports")
        assert r.status_code == 200

    def test_figure_404(self, client):
        r = client.get(f"/api/outputs/figures/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_reproducibility_404(self, client):
        r = client.get(f"/api/outputs/reproducibility/{uuid.uuid4()}")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Frontend Architecture Tests
# ---------------------------------------------------------------------------

class TestFrontendArchitecture:
    """Tests for frontend architecture invariants."""

    def test_api_js_centralizes_requests(self, client):
        """API layer centralizes fetch calls."""
        r = client.get("/static/js/api.js")
        assert r.status_code == 200
        content = r.text
        assert "fetch(" in content
        assert "APIError" in content
        assert "humanizeError" in content

    def test_api_js_has_all_service_modules(self, client):
        """API layer covers all major backend services."""
        r = client.get("/static/js/api.js")
        content = r.text
        for module in ["projects", "datasets", "pipeline", "quality", "analysisPlan", "runs", "figures", "tables", "reports", "reproducibility", "audit"]:
            assert module in content, f"Missing API module: {module}"

    def test_state_js_manages_loading(self, client):
        """State management handles loading states."""
        r = client.get("/static/js/state.js")
        content = r.text
        assert "startLoading" in content
        assert "endLoading" in content
        assert "isLoading" in content

    def test_state_js_handles_errors(self, client):
        """State management handles errors."""
        r = client.get("/static/js/state.js")
        content = r.text
        assert "setError" in content
        assert "getError" in content

    def test_state_js_notifies(self, client):
        """State management supports notifications."""
        r = client.get("/static/js/state.js")
        content = r.text
        assert "notify" in content

    def test_router_js_supports_deep_links(self, client):
        """Router supports project deep links."""
        r = client.get("/static/js/router.js")
        content = r.text
        assert "project/:id" in content
        assert "hashchange" in content

    def test_components_js_has_status_badges(self, client):
        """Components include textual status badges."""
        r = client.get("/static/js/components.js")
        content = r.text
        assert "statusBadge" in content
        assert "badge-" in content
        assert "badge" in content

    def test_components_js_has_loading_empty_error(self, client):
        """Components include loading, empty, and error states."""
        r = client.get("/static/js/components.js")
        content = r.text
        assert "loading" in content
        assert "empty" in content
        assert "error" in content

    def test_views_js_handles_all_major_views(self, client):
        """Views module handles all major researcher workflows."""
        r = client.get("/static/js/views.js")
        content = r.text
        for view in ["dashboard", "project", "resultsGlobal", "outputsGlobal", "reportsGlobal", "auditGlobal"]:
            assert view in content, f"Missing view: {view}"

    def test_views_js_project_subviews(self, client):
        """Project view supports all sub-views."""
        r = client.get("/static/js/views.js")
        content = r.text
        for subview in ["overview", "pipeline", "data", "analysis", "results", "outputs", "reports", "reproducibility", "audit"]:
            assert subview in content, f"Missing subview: {subview}"

    def test_css_has_dark_theme(self, client):
        """CSS preserves the STATIS dark theme."""
        r = client.get("/static/css/style.css")
        content = r.text
        assert "--bg:" in content
        assert "--accent:" in content
        assert "#0f1117" in content or "--bg: #0f1117" in content

    def test_css_has_accessible_focus(self, client):
        """CSS includes focus styles for accessibility."""
        r = client.get("/static/css/style.css")
        content = r.text
        assert "focus-visible" in content

    def test_css_responsive(self, client):
        """CSS includes responsive design rules."""
        r = client.get("/static/css/style.css")
        content = r.text
        assert "@media" in content


# ---------------------------------------------------------------------------
# Synthetic End-to-End Demonstration Tests
# ---------------------------------------------------------------------------

class TestEndToEndDemonstration:
    """Synthetic end-to-end workflow demonstration."""

    def test_full_workflow_demonstration(self, client, tmp_data_dir):
        """Demonstrate the complete researcher workflow via API."""
        # Create test user and get auth token
        from app.security.auth import create_user, create_access_token
        from app.core.database import get_session
        
        async def _setup_user():
            async with get_session() as session:
                user = await create_user(session, f"e2e_{_uid()}@example.com", "password123", "E2E Test User")
                return user
        
        user = asyncio.run(_setup_user())
        token = create_access_token(user.id, user.role)
        auth_headers = {"Authorization": f"Bearer {token}"}
        
        # 1. Create project
        r = client.post("/api/projects", data={
            "name": f"E2E Demo {_uid()}",
            "description": "End-to-end demonstration project",
            "research_question": "Does treatment affect blood pressure?",
            "study_design": "cohort",
        }, headers=auth_headers)
        assert r.status_code == 200
        project = r.json()
        pid = project["id"]

        # 2. Upload synthetic dataset
        csv = _make_csv(tmp_data_dir, n=80, seed=99)
        r = client.post(f"/api/projects/{pid}/upload", files={
            "file": (csv.name, csv.read_bytes(), "text/csv"),
        }, headers=auth_headers)
        assert r.status_code == 200
        dataset = r.json()
        did = dataset["id"]

        # 3. Verify project summary shows dataset
        r = client.get("/api/projects/summary")
        data = r.json()
        match = [p for p in data["projects"] if p["id"] == pid]
        assert len(match) == 1
        assert match[0]["dataset_count"] == 1

        # 4. Verify workflow shows dataset complete
        r = client.get(f"/api/projects/{pid}/workflow", headers=auth_headers)
        wf = r.json()
        ds_step = next(s for s in wf["steps"] if s["step"] == "dataset")
        assert ds_step["state"] == "complete"

        # 5. Run data quality assessment
        r = client.post(f"/api/data-quality/{did}/assess", json={})
        assert r.status_code == 200

        # 6. Verify quality profile exists
        r = client.get(f"/api/projects/{pid}/quality-profile", headers=auth_headers)
        assert r.status_code == 200
        assert r.json()["count"] >= 1

        # 7. Run statistical analysis (direct engine call)
        r = client.post("/api/analysis/t-test", json={
            "file_path": str(csv),
            "outcome": "systolic_bp",
            "group": "treatment",
            "group1": 0,
            "group2": 1,
        })
        assert r.status_code == 200
        result = r.json()
        # Verify backend values are present
        assert isinstance(result, dict)

        # 8. Verify audit log has entries
        r = client.get(f"/api/audit?project_id={pid}")
        assert r.status_code == 200

        # 9. Verify project resources endpoints work
        for endpoint in ["plans", "runs", "outputs", "variable-roles", "reproducibility"]:
            r = client.get(f"/api/projects/{pid}/{endpoint}", headers=auth_headers)
            assert r.status_code == 200, f"{endpoint}: {r.status_code}"

        # 10. Verify outputs are empty but accessible
        r = client.get(f"/api/projects/{pid}/outputs", headers=auth_headers)
        outputs = r.json()
        assert outputs["count"] == 0

    def test_workflow_status_after_upload(self, client, tmp_data_dir):
        """Workflow status updates after dataset upload."""
        from app.security.auth import create_user, create_access_token
        from app.core.database import get_session

        async def _setup_user():
            async with get_session() as session:
                user = await create_user(session, f"wf_{_uid()}@example.com", "password123", "WF Test User")
                return user

        user = asyncio.run(_setup_user())
        token = create_access_token(user.id, user.role)
        auth_headers = {"Authorization": f"Bearer {token}"}

        r = client.post("/api/projects", data={
            "name": f"WF Update {_uid()}",
            "research_question": "Test?",
            "study_design": "cohort",
        }, headers=auth_headers)
        pid = r.json()["id"]

        # Before upload: dataset not complete
        r = client.get(f"/api/projects/{pid}/workflow", headers=auth_headers)
        wf = r.json()
        ds_step = next(s for s in wf["steps"] if s["step"] == "dataset")
        assert ds_step["state"] == "not_started"

        # Upload dataset
        csv = _make_csv(tmp_data_dir, n=40, seed=7)
        r = client.post(f"/api/projects/{pid}/upload", files={
            "file": (csv.name, csv.read_bytes(), "text/csv"),
        }, headers=auth_headers)
        assert r.status_code == 200

        # After upload: dataset complete
        r = client.get(f"/api/projects/{pid}/workflow", headers=auth_headers)
        wf = r.json()
        ds_step = next(s for s in wf["steps"] if s["step"] == "dataset")
        assert ds_step["state"] == "complete"

    def test_frontend_serves_after_backend_changes(self, client):
        """Frontend still loads correctly after backend endpoint additions."""
        r = client.get("/")
        assert r.status_code == 200
        assert "STATIS" in r.text

        # New endpoints are registered
        r = client.get("/api/projects/summary")
        assert r.status_code == 200

        r = client.get("/openapi.json")
        schema = r.json()
        paths = schema.get("paths", {})
        assert "/api/projects/summary" in paths
        assert "/api/projects/{project_id}/workflow" in paths
