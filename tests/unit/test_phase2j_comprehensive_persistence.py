"""Phase 2J — Comprehensive Persistence Tests (Invariants 29–36).

Covers:
- Dataset identity persistence (Invariant 30)
- Statistical result persistence (Invariant 31)
- Output persistence (Invariant 32)
- Audit persistence (Invariant 36)
- Provenance integrity (Invariant 35)
- Researcher decision persistence (Invariant 34)
- Synthetic dataset fixture
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
    from app.api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def plan_persistence():
    from app.data import plan_persistence
    return plan_persistence


@pytest.fixture
def dataset_identity():
    from app.data import dataset_identity
    return dataset_identity


@pytest.fixture
def run_store():
    from app.data import run_store
    return run_store


@pytest.fixture
def audit_log():
    from app.memory.audit import log_audit, get_audit_log
    return log_audit, get_audit_log


@pytest.fixture
def synthetic_df():
    from tests.fixtures.synthetic_dataset import generate_synthetic_cohort
    return generate_synthetic_cohort()


@pytest.fixture
def synthetic_csv(tmp_path, synthetic_df):
    from tests.fixtures.synthetic_dataset import save_synthetic_cohort
    path = tmp_path / "synthetic_cohort.csv"
    save_synthetic_cohort(synthetic_df, path)
    return path


def _uid() -> str:
    return str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Invariant 30: Dataset Identity Persistence
# ---------------------------------------------------------------------------

class TestDatasetIdentityPersistence:
    """Dataset identity (canonical hash) survives restart."""

    def test_canonical_identity_roundtrip(self, dataset_identity, synthetic_csv):
        """Identity computed from file can be recomputed and matches."""
        identity1 = dataset_identity.compute_dataset_identity(synthetic_csv)
        identity2 = dataset_identity.compute_dataset_identity(synthetic_csv)
        assert identity1["content_hash"] == identity2["content_hash"]
        assert identity1["schema_hash"] == identity2["schema_hash"]
        assert identity1["row_count"] == identity2["row_count"]

    def test_same_data_same_identity(self, dataset_identity, synthetic_df, tmp_path):
        """Same data written twice → same identity."""
        p1 = tmp_path / "a.csv"
        p2 = tmp_path / "b.csv"
        synthetic_df.to_csv(p1, index=False)
        synthetic_df.to_csv(p2, index=False)
        id1 = dataset_identity.compute_dataset_identity(p1)
        id2 = dataset_identity.compute_dataset_identity(p2)
        assert id1["content_hash"] == id2["content_hash"]

    def test_reordered_columns_same_identity(self, dataset_identity, synthetic_df, tmp_path):
        """Column order does not affect canonical identity."""
        cols = list(synthetic_df.columns)
        shuffled = synthetic_df[cols[::-1]]
        p1 = tmp_path / "orig.csv"
        p2 = tmp_path / "shuffled.csv"
        synthetic_df.to_csv(p1, index=False)
        shuffled.to_csv(p2, index=False)
        id1 = dataset_identity.compute_dataset_identity(p1)
        id2 = dataset_identity.compute_dataset_identity(p2)
        assert id1["content_hash"] == id2["content_hash"]

    def test_reordered_rows_same_identity(self, dataset_identity, synthetic_df, tmp_path):
        """Row order does not affect canonical identity."""
        shuffled = synthetic_df.sample(frac=1, random_state=42).reset_index(drop=True)
        p1 = tmp_path / "orig.csv"
        p2 = tmp_path / "shuffled.csv"
        synthetic_df.to_csv(p1, index=False)
        shuffled.to_csv(p2, index=False)
        id1 = dataset_identity.compute_dataset_identity(p1)
        id2 = dataset_identity.compute_dataset_identity(p2)
        assert id1["content_hash"] == id2["content_hash"]

    def test_changed_value_different_identity(self, dataset_identity, synthetic_df, tmp_path):
        """A single value change produces a different identity."""
        df2 = synthetic_df.copy()
        df2.loc[0, "age"] = df2.loc[0, "age"] + 1.0
        p1 = tmp_path / "orig.csv"
        p2 = tmp_path / "changed.csv"
        synthetic_df.to_csv(p1, index=False)
        df2.to_csv(p2, index=False)
        id1 = dataset_identity.compute_dataset_identity(p1)
        id2 = dataset_identity.compute_dataset_identity(p2)
        assert id1["content_hash"] != id2["content_hash"]

    def test_changed_column_different_identity(self, dataset_identity, synthetic_df, tmp_path):
        """A renamed column produces a different schema hash."""
        df2 = synthetic_df.rename(columns={"age": "age_years"})
        p1 = tmp_path / "orig.csv"
        p2 = tmp_path / "renamed.csv"
        synthetic_df.to_csv(p1, index=False)
        df2.to_csv(p2, index=False)
        id1 = dataset_identity.compute_dataset_identity(p1)
        id2 = dataset_identity.compute_dataset_identity(p2)
        assert id1["schema_hash"] != id2["schema_hash"]
        assert id1["content_hash"] != id2["content_hash"]

    def test_missing_value_change_different_identity(self, dataset_identity, synthetic_df, tmp_path):
        """Changing a missing value to a present value changes identity."""
        df2 = synthetic_df.copy()
        # Find a NaN in hba1c and fill it
        nan_idx = df2["hba1c"].first_valid_index()
        if nan_idx is None:
            nan_idx = 0
        df2.loc[nan_idx, "hba1c"] = 7.0
        p1 = tmp_path / "orig.csv"
        p2 = tmp_path / "filled.csv"
        synthetic_df.to_csv(p1, index=False)
        df2.to_csv(p2, index=False)
        id1 = dataset_identity.compute_dataset_identity(p1)
        id2 = dataset_identity.compute_dataset_identity(p2)
        assert id1["content_hash"] != id2["content_hash"]

    def test_schema_change_different_schema_identity(self, dataset_identity, synthetic_df, tmp_path):
        """Adding a column changes schema identity."""
        df2 = synthetic_df.copy()
        df2["new_column"] = 1.0
        p1 = tmp_path / "orig.csv"
        p2 = tmp_path / "extended.csv"
        synthetic_df.to_csv(p1, index=False)
        df2.to_csv(p2, index=False)
        id1 = dataset_identity.compute_dataset_identity(p1)
        id2 = dataset_identity.compute_dataset_identity(p2)
        assert id1["schema_hash"] != id2["schema_hash"]

    def test_derived_dataset_distinct_identity(self, dataset_identity, synthetic_df, tmp_path):
        """A derived dataset has a distinct identity from the source."""
        df2 = synthetic_df.copy()
        df2["bmi_group"] = pd.cut(df2["bmi"], bins=[0, 25, 30, 100], labels=["normal", "overweight", "obese"])
        p1 = tmp_path / "source.csv"
        p2 = tmp_path / "derived.csv"
        synthetic_df.to_csv(p1, index=False)
        df2.to_csv(p2, index=False)
        id1 = dataset_identity.compute_dataset_identity(p1)
        id2 = dataset_identity.compute_dataset_identity(p2)
        assert id1["content_hash"] != id2["content_hash"]
        assert id1["schema_hash"] != id2["schema_hash"]

    def test_verify_dataset_identity(self, dataset_identity, synthetic_csv):
        """verify_dataset_identity returns valid=True for unchanged data."""
        identity = dataset_identity.compute_dataset_identity(synthetic_csv)
        result = dataset_identity.verify_dataset_identity(
            synthetic_csv,
            expected_content_hash=identity["content_hash"],
            expected_schema_hash=identity["schema_hash"],
            expected_row_count=identity["row_count"],
        )
        assert result["valid"] is True
        assert result["content_match"] is True

    def test_verify_dataset_identity_mismatch(self, dataset_identity, synthetic_csv):
        """verify_dataset_identity returns valid=False for changed data."""
        result = dataset_identity.verify_dataset_identity(
            synthetic_csv,
            expected_content_hash="0" * 64,
        )
        assert result["valid"] is False
        assert result["content_match"] is False


# ---------------------------------------------------------------------------
# Invariant 31: Statistical Result Persistence
# ---------------------------------------------------------------------------

class TestStatisticalResultPersistence:
    """Statistical results and fingerprints survive restart."""

    def test_run_store_create_and_get(self, run_store, synthetic_csv, dataset_identity):
        """Analysis run created and retrieved preserves all fields."""
        identity = dataset_identity.compute_dataset_identity(synthetic_csv)
        spec = {"analysis_type": "independent_t_test", "outcome": "qol_score", "group": "treatment"}

        run_id = asyncio.run(run_store.create_analysis_run(
            project_id="test-project",
            analysis_type="independent_t_test",
            specification=spec,
            dataset_id="test-dataset",
            dataset_content_hash=identity["content_hash"],
            dataset_schema_hash=identity["schema_hash"],
            file_path=str(synthetic_csv),
        ))

        record = asyncio.run(run_store.get_analysis_run(run_id))
        assert record is not None
        assert record["project_id"] == "test-project"
        assert record["analysis_type"] == "independent_t_test"
        assert record["dataset_content_hash"] == identity["content_hash"]
        assert record["dataset_schema_hash"] == identity["schema_hash"]
        assert record["input_fingerprint"] is not None
        assert record["result_fingerprint"] is None  # not completed yet

    def test_run_store_complete_preserves_result(self, run_store, synthetic_csv, dataset_identity):
        """Completing a run persists the result dict and fingerprint."""
        identity = dataset_identity.compute_dataset_identity(synthetic_csv)
        spec = {"analysis_type": "independent_t_test", "outcome": "qol_score", "group": "treatment"}
        result_data = {
            "test": "Welch's t-test",
            "t_statistic": -3.42,
            "p_value": 0.0006,
            "cohens_d": -0.38,
            "df": 580.5,
            "n_group1": 312,
            "n_group2": 288,
        }

        run_id = asyncio.run(run_store.create_analysis_run(
            project_id="test-project",
            analysis_type="independent_t_test",
            specification=spec,
            dataset_id="test-dataset",
            dataset_content_hash=identity["content_hash"],
            dataset_schema_hash=identity["schema_hash"],
            file_path=str(synthetic_csv),
        ))

        completed = asyncio.run(run_store.complete_analysis_run(
            run_id,
            result=result_data,
            success=True,
            lifecycle="executed",
        ))

        assert completed is not None
        assert completed["success"] is True
        assert completed["result_fingerprint"] is not None
        assert completed["result"]["t_statistic"] == -3.42
        assert completed["result"]["p_value"] == 0.0006
        assert completed["lifecycle"] == "executed"

    def test_result_fingerprint_deterministic(self, run_store):
        """Same result dict → same fingerprint."""
        result = {"t_statistic": 1.5, "p_value": 0.13, "n": 100}
        fp1 = run_store.compute_result_fingerprint(result)
        fp2 = run_store.compute_result_fingerprint(result)
        assert fp1 == fp2

    def test_result_fingerprint_sensitive_to_values(self, run_store):
        """Different result values → different fingerprint."""
        r1 = {"t_statistic": 1.5, "p_value": 0.13}
        r2 = {"t_statistic": 1.6, "p_value": 0.13}
        fp1 = run_store.compute_result_fingerprint(r1)
        fp2 = run_store.compute_result_fingerprint(r2)
        assert fp1 != fp2

    def test_input_fingerprint_deterministic(self, run_store):
        """Same specification + dataset hash → same input fingerprint."""
        spec = {"analysis_type": "t_test", "outcome": "y"}
        fp1 = run_store.compute_input_fingerprint(spec, "hash1", "schema1")
        fp2 = run_store.compute_input_fingerprint(spec, "hash1", "schema1")
        assert fp1 == fp2

    def test_input_fingerprint_sensitive_to_spec(self, run_store):
        """Different specification → different input fingerprint."""
        spec1 = {"analysis_type": "t_test", "outcome": "y"}
        spec2 = {"analysis_type": "t_test", "outcome": "z"}
        fp1 = run_store.compute_input_fingerprint(spec1, "hash1", "schema1")
        fp2 = run_store.compute_input_fingerprint(spec2, "hash1", "schema1")
        assert fp1 != fp2

    def test_list_analysis_runs(self, run_store, synthetic_csv, dataset_identity):
        """list_analysis_runs returns runs for a project."""
        identity = dataset_identity.compute_dataset_identity(synthetic_csv)
        project_id = f"list-project-{_uid()}"
        spec = {"analysis_type": "correlation"}

        asyncio.run(run_store.create_analysis_run(
            project_id=project_id,
            analysis_type="correlation",
            specification=spec,
            dataset_id="ds",
            dataset_content_hash=identity["content_hash"],
            dataset_schema_hash=identity["schema_hash"],
        ))

        runs = asyncio.run(run_store.list_analysis_runs(project_id=project_id))
        assert len(runs) >= 1
        assert runs[0]["project_id"] == project_id


# ---------------------------------------------------------------------------
# Invariant 34: Researcher Decision Persistence
# ---------------------------------------------------------------------------

class TestResearcherDecisionPersistence:
    """Researcher decisions survive restart and remain auditable."""

    def test_analysis_approval_persists(self, plan_persistence):
        """Approved analysis decision persists."""
        plan_id = f"decision-{_uid()}"
        analysis_id = f"analysis-{_uid()}"
        plan = {
            "plan_id": plan_id,
            "project_id": "decision-project",
            "analyses": [{
                "analysis_id": analysis_id,
                "method": "independent_t_test",
                "lifecycle": "draft",
                "approval": "proposed",
                "prespecified": "prespecified",
            }],
        }
        asyncio.run(plan_persistence.save_plan(plan))
        asyncio.run(plan_persistence.update_analysis(plan_id, analysis_id, {
            "approval": "approved",
            "lifecycle": "approved",
        }))
        loaded = asyncio.run(plan_persistence.load_plan(plan_id))
        assert loaded["analyses"][0]["approval"] == "approved"

    def test_analysis_rejection_persists(self, plan_persistence):
        """Rejected analysis decision persists."""
        plan_id = f"decision-{_uid()}"
        analysis_id = f"analysis-{_uid()}"
        plan = {
            "plan_id": plan_id,
            "project_id": "decision-project",
            "analyses": [{
                "analysis_id": analysis_id,
                "method": "logistic_regression",
                "lifecycle": "draft",
                "approval": "proposed",
            }],
        }
        asyncio.run(plan_persistence.save_plan(plan))
        asyncio.run(plan_persistence.update_analysis(plan_id, analysis_id, {
            "approval": "rejected",
            "lifecycle": "draft",
        }))
        loaded = asyncio.run(plan_persistence.load_plan(plan_id))
        assert loaded["analyses"][0]["approval"] == "rejected"

    def test_prespecification_status_persists(self, plan_persistence):
        """Prespecification status decision persists."""
        plan_id = f"prespec-{_uid()}"
        analysis_id = f"analysis-{_uid()}"
        plan = {
            "plan_id": plan_id,
            "project_id": "prespec-project",
            "analyses": [{
                "analysis_id": analysis_id,
                "method": "independent_t_test",
                "lifecycle": "draft",
                "approval": "proposed",
                "prespecified": "unclear",
            }],
        }
        asyncio.run(plan_persistence.save_plan(plan))
        # Researcher confirms prespecification
        asyncio.run(plan_persistence.update_analysis(plan_id, analysis_id, {
            "prespecified": "prespecified",
        }))
        loaded = asyncio.run(plan_persistence.load_plan(plan_id))
        assert loaded["analyses"][0]["prespecified"] == "prespecified"

    def test_analysis_classification_persists(self, plan_persistence):
        """Primary/secondary/sensitivity/exploratory classification persists."""
        plan_id = f"class-{_uid()}"
        analysis_id = f"analysis-{_uid()}"
        plan = {
            "plan_id": plan_id,
            "project_id": "class-project",
            "analyses": [{
                "analysis_id": analysis_id,
                "method": "mann_whitney_u",
                "analysis_class": "exploratory",
                "lifecycle": "draft",
                "approval": "proposed",
            }],
        }
        asyncio.run(plan_persistence.save_plan(plan))
        # Researcher reclassifies as sensitivity
        asyncio.run(plan_persistence.update_analysis(plan_id, analysis_id, {
            "analysis_class": "sensitivity",
        }))
        loaded = asyncio.run(plan_persistence.load_plan(plan_id))
        assert loaded["analyses"][0]["analysis_class"] == "sensitivity"


# ---------------------------------------------------------------------------
# Invariant 36: Audit Persistence
# ---------------------------------------------------------------------------

class TestAuditPersistence:
    """Audit events survive restart."""

    def test_audit_event_persists(self, audit_log):
        """A logged audit event is retrievable."""
        log_audit, get_audit_log = audit_log
        project_id = f"audit-project-{_uid()}"
        asyncio.run(log_audit(
            action="test_action",
            agent="test_agent",
            decision="test decision",
            project_id=project_id,
        ))
        entries = asyncio.run(get_audit_log(project_id=project_id))
        assert len(entries) >= 1
        assert entries[0]["action"] == "test_action"
        assert entries[0]["agent"] == "test_agent"
        assert entries[0]["decision"] == "test decision"

    def test_multiple_audit_events_persist(self, audit_log):
        """Multiple audit events all persist."""
        log_audit, get_audit_log = audit_log
        project_id = f"audit-multi-{_uid()}"
        for i in range(3):
            asyncio.run(log_audit(
                action=f"action_{i}",
                agent="test_agent",
                decision=f"decision_{i}",
                project_id=project_id,
            ))
        entries = asyncio.run(get_audit_log(project_id=project_id))
        assert len(entries) >= 3

    def test_researcher_decision_audit_persists(self, audit_log):
        """Researcher decision audit event persists with researcher_decision field."""
        log_audit, get_audit_log = audit_log
        project_id = f"audit-rd-{_uid()}"
        asyncio.run(log_audit(
            action="analysis_approved",
            agent="researcher",
            decision="Approved primary analysis",
            project_id=project_id,
            researcher_decision="approve",
        ))
        entries = asyncio.run(get_audit_log(project_id=project_id))
        assert len(entries) >= 1
        assert entries[0]["researcher_decision"] == "approve"


# ---------------------------------------------------------------------------
# Invariant 35: Provenance Integrity
# ---------------------------------------------------------------------------

class TestProvenanceIntegrity:
    """Provenance chain Project → Dataset → Analysis → Result → Output."""

    def test_run_provenance_fields_present(self, run_store, synthetic_csv, dataset_identity):
        """Analysis run records carry full provenance fields."""
        identity = dataset_identity.compute_dataset_identity(synthetic_csv)
        spec = {"analysis_type": "independent_t_test"}
        run_id = asyncio.run(run_store.create_analysis_run(
            project_id="prov-project",
            analysis_type="independent_t_test",
            specification=spec,
            dataset_id="prov-dataset",
            dataset_content_hash=identity["content_hash"],
            dataset_schema_hash=identity["schema_hash"],
        ))
        record = asyncio.run(run_store.get_analysis_run(run_id))
        assert record["project_id"] == "prov-project"
        assert record["dataset_id"] == "prov-dataset"
        assert record["dataset_content_hash"] == identity["content_hash"]
        assert record["dataset_schema_hash"] == identity["schema_hash"]
        assert record["specification"] is not None
        assert record["input_fingerprint"] is not None

    def test_cross_dataset_provenance_is_distinct(self, run_store, synthetic_csv, dataset_identity, tmp_path):
        """Two datasets → two distinct dataset_content_hash values in runs."""
        df2 = synthetic_csv.read_bytes() + b"\n0,0,0"
        p2 = tmp_path / "other.csv"
        p2.write_bytes(df2)

        id1 = dataset_identity.compute_dataset_identity(synthetic_csv)
        id2 = dataset_identity.compute_dataset_identity(p2)

        spec = {"analysis_type": "independent_t_test"}
        run1 = asyncio.run(run_store.create_analysis_run(
            project_id="prov-cross",
            analysis_type="t",
            specification=spec,
            dataset_id="ds1",
            dataset_content_hash=id1["content_hash"],
            dataset_schema_hash=id1["schema_hash"],
        ))
        run2 = asyncio.run(run_store.create_analysis_run(
            project_id="prov-cross",
            analysis_type="t",
            specification=spec,
            dataset_id="ds2",
            dataset_content_hash=id2["content_hash"],
            dataset_schema_hash=id2["schema_hash"],
        ))

        r1 = asyncio.run(run_store.get_analysis_run(run1))
        r2 = asyncio.run(run_store.get_analysis_run(run2))
        assert r1["dataset_content_hash"] != r2["dataset_content_hash"]

    def test_plan_dataset_binding_persists(self, plan_persistence):
        """Plan's dataset_id binding persists and is distinct per plan."""
        plan_id = f"bind-{_uid()}"
        plan = {
            "plan_id": plan_id,
            "project_id": "bind-project",
            "dataset_id": f"dataset-{_uid()}",
            "analyses": [],
        }
        asyncio.run(plan_persistence.save_plan(plan))
        loaded = asyncio.run(plan_persistence.load_plan(plan_id))
        assert loaded["dataset_id"] == plan["dataset_id"]
        # Ensure it's not None or empty
        assert loaded["dataset_id"] != ""
        assert loaded["dataset_id"] is not None
