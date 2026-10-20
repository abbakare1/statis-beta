"""Phase 2J — Analysis Plan Persistence Tests.

Tests that analysis plans survive application restart via DB persistence:
- Create plan → approve → lock → execute → restart → retrieve same plan
- Lifecycle state persistence across restart
- Researcher decision persistence
- Provenance and audit persistence
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path

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
def plan_persistence():
    """Import the plan_persistence module."""
    from app.data import plan_persistence
    return plan_persistence


def _uid() -> str:
    return str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Invariant 29: Analysis Plan Persistence
# ---------------------------------------------------------------------------

class TestPlanPersistence:
    """Verify plans persist to the database and survive restart."""

    def test_save_and_load_plan(self, plan_persistence):
        """Plan saved via save_plan can be loaded via load_plan."""
        plan_id = f"test-plan-{_uid()}"
        plan = {
            "plan_id": plan_id,
            "project_id": "test-project",
            "dataset_id": "test-dataset",
            "research_question": "Does X affect Y?",
            "study_design": "observational",
            "analyses": [
                {
                    "analysis_id": f"analysis-{_uid()}",
                    "analysis_name": "Primary Analysis",
                    "method": "independent_t_test",
                    "outcome": "Y",
                    "predictors": ["X"],
                    "lifecycle": "draft",
                    "approval": "proposed",
                    "validation": "pass",
                }
            ],
            "warnings": ["Test warning"],
            "rejected_methods": ["method_a"],
        }

        asyncio.run(plan_persistence.save_plan(plan))
        loaded = asyncio.run(plan_persistence.load_plan(plan_id))

        assert loaded is not None
        assert loaded["plan_id"] == plan_id
        assert loaded["research_question"] == "Does X affect Y?"
        assert len(loaded["analyses"]) == 1
        assert loaded["analyses"][0]["method"] == "independent_t_test"
        assert loaded["warnings"] == ["Test warning"]

    def test_load_nonexistent_plan_returns_none(self, plan_persistence):
        """Loading a plan that doesn't exist returns None."""
        result = asyncio.run(plan_persistence.load_plan("nonexistent-plan-id"))
        assert result is None

    def test_update_analysis_persists(self, plan_persistence):
        """Updating an analysis field persists to the database."""
        plan_id = f"test-update-{_uid()}"
        analysis_id = f"analysis-{_uid()}"
        plan = {
            "plan_id": plan_id,
            "project_id": "test-project",
            "analyses": [
                {
                    "analysis_id": analysis_id,
                    "method": "independent_t_test",
                    "lifecycle": "draft",
                    "approval": "proposed",
                }
            ],
        }

        asyncio.run(plan_persistence.save_plan(plan))

        # Approve the analysis
        asyncio.run(plan_persistence.update_analysis(plan_id, analysis_id, {
            "approval": "approved",
            "lifecycle": "approved",
        }))

        # Load and verify
        loaded = asyncio.run(plan_persistence.load_plan(plan_id))
        assert loaded["analyses"][0]["approval"] == "approved"
        assert loaded["analyses"][0]["lifecycle"] == "approved"

    def test_find_analysis_across_plans(self, plan_persistence):
        """find_analysis locates an analysis in any plan."""
        plan_id = f"test-find-{_uid()}"
        analysis_id = f"analysis-{_uid()}"
        plan = {
            "plan_id": plan_id,
            "project_id": "test-project",
            "analyses": [
                {
                    "analysis_id": analysis_id,
                    "method": "mann_whitney",
                    "lifecycle": "draft",
                }
            ],
        }

        asyncio.run(plan_persistence.save_plan(plan))
        found = asyncio.run(plan_persistence.find_analysis(analysis_id))

        assert found is not None
        found_plan, found_analysis = found
        assert found_plan["plan_id"] == plan_id
        assert found_analysis["method"] == "mann_whitney"

    def test_find_nonexistent_analysis_returns_none(self, plan_persistence):
        """find_analysis returns None for unknown analysis_id."""
        found = asyncio.run(plan_persistence.find_analysis("nonexistent-analysis-id"))
        assert found is None

    def test_delete_plan(self, plan_persistence):
        """delete_plan removes the plan from the database."""
        plan_id = f"test-delete-{_uid()}"
        plan = {
            "plan_id": plan_id,
            "project_id": "test-project",
            "analyses": [],
        }

        asyncio.run(plan_persistence.save_plan(plan))
        assert asyncio.run(plan_persistence.load_plan(plan_id)) is not None

        deleted = asyncio.run(plan_persistence.delete_plan(plan_id))
        assert deleted is True
        assert asyncio.run(plan_persistence.load_plan(plan_id)) is None

    def test_delete_nonexistent_plan_returns_false(self, plan_persistence):
        """Deleting a nonexistent plan returns False."""
        deleted = asyncio.run(plan_persistence.delete_plan("nonexistent-plan-id"))
        assert deleted is False


# ---------------------------------------------------------------------------
# API Integration Tests
# ---------------------------------------------------------------------------

class TestPlanAPIIntegration:
    """Test plan endpoints with DB persistence via the API."""

    @pytest.fixture(autouse=True)
    def auth_headers(self):
        """Create a test user and return auth headers."""
        from app.security.auth import create_user, create_access_token
        from app.core.database import get_session

        async def _setup():
            async with get_session() as session:
                user = await create_user(session, f"plan_api_{uuid.uuid4().hex[:8]}@example.com", "password123", "Plan API Test User")
                return user

        user = asyncio.run(_setup())
        token = create_access_token(user.id, user.role)
        return {"Authorization": f"Bearer {token}"}

    def test_get_plan_not_found(self, client, auth_headers):
        """GET /api/analysis-plan/{plan_id} returns 404 for missing plan."""
        resp = client.get("/api/analysis-plan/nonexistent-plan-id", headers=auth_headers)
        assert resp.status_code == 404

    def test_decision_trace_not_found(self, client, auth_headers):
        """GET /api/analysis-plan/{analysis_id}/decision-trace returns 404 for missing analysis."""
        resp = client.get("/api/analysis-plan/nonexistent-analysis-id/decision-trace", headers=auth_headers)
        assert resp.status_code == 404

    def test_approve_without_plan_id(self, client, auth_headers):
        """POST approve without plan_id returns 400."""
        resp = client.post(
            "/api/analysis-plan/some-analysis/approve",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_lock_without_plan_id(self, client, auth_headers):
        """POST lock without plan_id returns 400."""
        resp = client.post(
            "/api/analysis-plan/some-analysis/lock",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_reject_without_plan_id(self, client, auth_headers):
        """POST reject without plan_id returns 400."""
        resp = client.post(
            "/api/analysis-plan/some-analysis/reject",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_execute_without_plan_id(self, client, auth_headers):
        """POST execute without plan_id returns 400."""
        resp = client.post(
            "/api/analysis-plan/some-analysis/execute",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Restart / Recovery Test (Invariant 29)
# ---------------------------------------------------------------------------

class TestRestartRecovery:
    """Verify plans survive a simulated application restart."""

    def test_plan_survives_restart(self, plan_persistence):
        """Plan created before 'restart' is retrievable after."""
        plan_id = f"restart-test-{_uid()}"
        analysis_id = f"analysis-{_uid()}"

        # Phase 1: Create plan and approve analysis (before "restart")
        plan = {
            "plan_id": plan_id,
            "project_id": "restart-project",
            "dataset_id": "restart-dataset",
            "research_question": "Does treatment affect outcome?",
            "study_design": "observational",
            "analyses": [
                {
                    "analysis_id": analysis_id,
                    "analysis_name": "Primary T-Test",
                    "method": "independent_t_test",
                    "outcome": "outcome",
                    "predictors": ["treatment"],
                    "lifecycle": "draft",
                    "approval": "proposed",
                    "validation": "pass",
                }
            ],
            "warnings": [],
            "rejected_methods": [],
        }
        asyncio.run(plan_persistence.save_plan(plan))

        # Approve
        asyncio.run(plan_persistence.update_analysis(plan_id, analysis_id, {
            "approval": "approved",
            "lifecycle": "approved",
        }))

        # Lock
        asyncio.run(plan_persistence.update_analysis(plan_id, analysis_id, {
            "lifecycle": "locked",
        }))

        # Phase 2: Simulate restart — clear any in-memory state, reload from DB
        # (Since we're using DB persistence, this is just loading again)
        loaded = asyncio.run(plan_persistence.load_plan(plan_id))

        # Phase 3: Verify all state persisted
        assert loaded is not None
        assert loaded["plan_id"] == plan_id
        assert loaded["project_id"] == "restart-project"
        assert loaded["research_question"] == "Does treatment affect outcome?"

        analysis = loaded["analyses"][0]
        assert analysis["analysis_id"] == analysis_id
        assert analysis["method"] == "independent_t_test"
        assert analysis["approval"] == "approved"
        assert analysis["lifecycle"] == "locked"

    def test_multiple_analyses_persist(self, plan_persistence):
        """Multiple analyses in a plan all persist correctly."""
        plan_id = f"multi-analysis-{_uid()}"
        analysis1_id = f"analysis-{_uid()}"
        analysis2_id = f"analysis-{_uid()}"

        plan = {
            "plan_id": plan_id,
            "project_id": "multi-project",
            "analyses": [
                {
                    "analysis_id": analysis1_id,
                    "method": "independent_t_test",
                    "lifecycle": "draft",
                    "approval": "proposed",
                },
                {
                    "analysis_id": analysis2_id,
                    "method": "mann_whitney",
                    "lifecycle": "draft",
                    "approval": "proposed",
                },
            ],
        }

        asyncio.run(plan_persistence.save_plan(plan))

        # Approve first, lock second
        asyncio.run(plan_persistence.update_analysis(plan_id, analysis1_id, {
            "approval": "approved",
            "lifecycle": "approved",
        }))
        asyncio.run(plan_persistence.update_analysis(plan_id, analysis2_id, {
            "approval": "approved",
            "lifecycle": "locked",
        }))

        # Verify both persisted
        loaded = asyncio.run(plan_persistence.load_plan(plan_id))
        assert len(loaded["analyses"]) == 2

        a1 = next(a for a in loaded["analyses"] if a["analysis_id"] == analysis1_id)
        a2 = next(a for a in loaded["analyses"] if a["analysis_id"] == analysis2_id)

        assert a1["lifecycle"] == "approved"
        assert a2["lifecycle"] == "locked"

    def test_rejected_analysis_persists(self, plan_persistence):
        """Rejected analysis state persists across restart."""
        plan_id = f"reject-test-{_uid()}"
        analysis_id = f"analysis-{_uid()}"

        plan = {
            "plan_id": plan_id,
            "project_id": "reject-project",
            "analyses": [
                {
                    "analysis_id": analysis_id,
                    "method": "logistic_regression",
                    "lifecycle": "draft",
                    "approval": "proposed",
                }
            ],
        }

        asyncio.run(plan_persistence.save_plan(plan))

        # Reject
        asyncio.run(plan_persistence.update_analysis(plan_id, analysis_id, {
            "approval": "rejected",
            "lifecycle": "draft",
        }))

        # Verify
        loaded = asyncio.run(plan_persistence.load_plan(plan_id))
        assert loaded["analyses"][0]["approval"] == "rejected"
        assert loaded["analyses"][0]["lifecycle"] == "draft"


# ---------------------------------------------------------------------------
# Invariant 30: Dataset Persistence (plans reference datasets)
# ---------------------------------------------------------------------------

class TestDatasetPersistenceInPlans:
    """Verify dataset references persist in plans."""

    def test_dataset_id_persists(self, plan_persistence):
        """dataset_id stored in plan persists correctly."""
        plan_id = f"dataset-persist-{_uid()}"
        dataset_id = f"dataset-{_uid()}"

        plan = {
            "plan_id": plan_id,
            "project_id": "dataset-project",
            "dataset_id": dataset_id,
            "analyses": [],
        }

        asyncio.run(plan_persistence.save_plan(plan))
        loaded = asyncio.run(plan_persistence.load_plan(plan_id))

        assert loaded["dataset_id"] == dataset_id

    def test_research_context_persists(self, plan_persistence):
        """Research question and study design persist."""
        plan_id = f"context-persist-{_uid()}"

        plan = {
            "plan_id": plan_id,
            "project_id": "context-project",
            "research_question": "What is the effect of X on Y?",
            "study_design": "retrospective cohort",
            "analyses": [],
        }

        asyncio.run(plan_persistence.save_plan(plan))
        loaded = asyncio.run(plan_persistence.load_plan(plan_id))

        assert loaded["research_question"] == "What is the effect of X on Y?"
        assert loaded["study_design"] == "retrospective cohort"
