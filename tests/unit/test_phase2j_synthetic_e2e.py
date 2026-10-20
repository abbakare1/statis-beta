"""Phase 2J — Full Synthetic E2E Workflow (Invariant 42, Steps 7).

Complete workflow using the deterministic synthetic cohort:
  1. Generate synthetic dataset
  2. Save to CSV
  3. Compute dataset identity (canonical hash)
  4. Create analysis plan
  5. Researcher approves plan
  6. Execute analyses (t-test, logistic, survival)
  7. Record runs with fingerprints
  8. Compute defensibility
  9. Log audit entries
  10. Simulate application restart
  11. Verify full recovery of plan, runs, results, audit, identity

All values are deterministic (seed=42). No LLM required.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.database import init_db, close_db
from app.data import plan_persistence
from app.data.run_store import (
    create_analysis_run, complete_analysis_run, get_analysis_run,
    compute_result_fingerprint,
)
from app.data.dataset_identity import compute_dataset_identity
from app.memory.audit import log_audit, get_audit_log
from app.statistics.engine import (
    independent_t_test, logistic_regression, linear_regression,
    mann_whitney_u, correlation_test,
)
from app.statistics.survival import kaplan_meier, cox_proportional_hazards, log_rank
from app.statistics.defensibility import assess_defensibility
from app.statistics.result import ExecutionStatus, SanityStatus
from app.statistics.diagnostics import DiagnosticDiagnosticStatus
from tests.fixtures.synthetic_dataset import (
    generate_synthetic_cohort, save_synthetic_cohort, GROUND_TRUTH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _reinit_db():
    await close_db()
    await init_db()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_csv(tmp_path):
    """Generate and save the synthetic cohort; return its CSV path."""
    df = generate_synthetic_cohort()
    path = tmp_path / "synthetic_cohort.csv"
    save_synthetic_cohort(df, path)
    return path


# ===========================================================================
# STEP 7: Full Synthetic E2E Workflow
# ===========================================================================

class TestSyntheticE2EWorkflow:
    """End-to-end workflow on the deterministic synthetic cohort."""

    async def test_full_workflow_survives_restart(self, synthetic_csv):
        """Complete workflow: generate → identity → plan → approve → execute → audit → restart → recover."""

        await init_db()

        # ----------------------------------------------------------------
        # Step 1: Dataset identity
        # ----------------------------------------------------------------
        ds_identity = compute_dataset_identity(synthetic_csv)
        assert ds_identity["row_count"] == GROUND_TRUTH["n_observations"]
        assert ds_identity["content_hash"]
        assert ds_identity["schema_hash"]

        # ----------------------------------------------------------------
        # Step 2: Create analysis plan
        # ----------------------------------------------------------------
        plan_id = "synthetic-e2e-plan"
        project_id = "synthetic-e2e-project"
        analyses = [
            {
                "analysis_id": "synthetic-analysis-ttest",
                "method": "independent_t_test",
                "outcome": "qol_score",
                "predictors": ["treatment"],
                "lifecycle": "draft",
                "approval": "proposed",
            },
            {
                "analysis_id": "synthetic-analysis-logistic",
                "method": "logistic_regression",
                "outcome": "readmission_30d",
                "predictors": ["treatment", "heart_failure", "age"],
                "lifecycle": "draft",
                "approval": "proposed",
            },
            {
                "analysis_id": "synthetic-analysis-cox",
                "method": "cox_proportional_hazards",
                "outcome": "time_to_event",
                "predictors": ["treatment"],
                "lifecycle": "draft",
                "approval": "proposed",
            },
            {
                "analysis_id": "synthetic-analysis-paired",
                "method": "paired_t_test",
                "outcome": "sbp_followup",
                "predictors": ["sbp_baseline"],
                "lifecycle": "draft",
                "approval": "proposed",
            },
        ]
        plan = {
            "plan_id": plan_id,
            "project_id": project_id,
            "dataset_id": "synthetic-e2e-dataset",
            "research_question": "Does Treatment_B affect quality of life, readmission, and survival?",
            "study_design": "observational",
            "analyses": analyses,
            "warnings": [],
            "rejected_methods": [],
        }
        await plan_persistence.save_plan(plan)

        # ----------------------------------------------------------------
        # Step 3: Researcher approves all analyses
        # ----------------------------------------------------------------
        for a in analyses:
            await plan_persistence.update_analysis(plan_id, a["analysis_id"], {
                "approval": "approved",
                "lifecycle": "approved",
            })

        # ----------------------------------------------------------------
        # Step 4: Execute t-test (qol_score by treatment)
        # ----------------------------------------------------------------
        ttest_result = independent_t_test(
            synthetic_csv, outcome="qol_score", group="treatment",
            group1=0, group2=1,
        )
        assert "error" not in ttest_result
        assert ttest_result["n_group1"] > 0
        assert ttest_result["n_group2"] > 0
        assert ttest_result["p_value"] is not None

        # Record run
        run_id_ttest = await create_analysis_run(
            project_id=project_id,
            analysis_type="independent_t_test",
            specification={
                "analysis_method": "independent_t_test",
                "primary_outcome": "qol_score",
                "predictors": ["treatment"],
            },
            dataset_id="synthetic-e2e-dataset",
            dataset_content_hash=ds_identity["content_hash"],
            dataset_schema_hash=ds_identity["schema_hash"],
            file_path=str(synthetic_csv),
        )
        await complete_analysis_run(run_id_ttest, result=ttest_result)

        # ----------------------------------------------------------------
        # Step 5: Execute logistic regression (readmission)
        # ----------------------------------------------------------------
        logistic_result = logistic_regression(
            synthetic_csv, outcome="readmission_30d",
            predictors=["treatment", "heart_failure", "age"],
        )
        assert "error" not in logistic_result
        assert "parameters" in logistic_result
        assert "treatment" in logistic_result["parameters"]
        assert "heart_failure" in logistic_result["parameters"]

        run_id_logistic = await create_analysis_run(
            project_id=project_id,
            analysis_type="logistic_regression",
            specification={
                "analysis_method": "logistic_regression",
                "primary_outcome": "readmission_30d",
                "predictors": ["treatment", "heart_failure", "age"],
            },
            dataset_id="synthetic-e2e-dataset",
            dataset_content_hash=ds_identity["content_hash"],
            dataset_schema_hash=ds_identity["schema_hash"],
            file_path=str(synthetic_csv),
        )
        await complete_analysis_run(run_id_logistic, result=logistic_result)

        # ----------------------------------------------------------------
        # Step 6: Execute Cox PH (survival)
        # ----------------------------------------------------------------
        cox_result = cox_proportional_hazards(
            synthetic_csv, time_col="time_to_event",
            event_col="event_observed", covariates=["treatment"],
        )
        assert "error" not in cox_result
        assert "hazard_ratios" in cox_result
        assert "treatment" in cox_result["hazard_ratios"]

        run_id_cox = await create_analysis_run(
            project_id=project_id,
            analysis_type="cox_proportional_hazards",
            specification={
                "analysis_method": "cox_proportional_hazards",
                "primary_outcome": "time_to_event",
                "predictors": ["treatment"],
            },
            dataset_id="synthetic-e2e-dataset",
            dataset_content_hash=ds_identity["content_hash"],
            dataset_schema_hash=ds_identity["schema_hash"],
            file_path=str(synthetic_csv),
        )
        await complete_analysis_run(run_id_cox, result=cox_result)

        # ----------------------------------------------------------------
        # Step 7: Execute paired t-test (SBP baseline vs follow-up)
        # ----------------------------------------------------------------
        from app.statistics.engine import paired_t_test
        paired_result = paired_t_test(
            synthetic_csv, outcome1="sbp_followup", outcome2="sbp_baseline",
        )
        assert "error" not in paired_result
        assert paired_result["n_pairs"] > 0

        run_id_paired = await create_analysis_run(
            project_id=project_id,
            analysis_type="paired_t_test",
            specification={
                "analysis_method": "paired_t_test",
                "primary_outcome": "sbp_followup",
                "predictors": ["sbp_baseline"],
            },
            dataset_id="synthetic-e2e-dataset",
            dataset_content_hash=ds_identity["content_hash"],
            dataset_schema_hash=ds_identity["schema_hash"],
            file_path=str(synthetic_csv),
        )
        await complete_analysis_run(run_id_paired, result=paired_result)

        # ----------------------------------------------------------------
        # Step 8: Defensibility assessment
        # ----------------------------------------------------------------
        # t-test: normal data, sufficient n → defensible
        ttest_def = assess_defensibility(
            ExecutionStatus.SUCCESS, SanityStatus.PASS,
            DiagnosticDiagnosticStatus.PASS,
        )
        assert ttest_def.value == "defensible"

        # Cox: check survival metadata
        n_events = int(cox_result.get("n_events", 0))
        n_total = int(cox_result.get("n_total", 0))
        n_censored = n_total - n_events if n_total else 0
        cox_def = assess_defensibility(
            ExecutionStatus.SUCCESS, SanityStatus.PASS,
            DiagnosticDiagnosticStatus.PASS,
            survival_metadata={
                "method_id": "cox_proportional_hazards",
                "n_events": n_events,
                "n_censored": n_censored,
                "n_total": n_total,
            },
        )
        # Should be defensible or review_required depending on event count
        assert cox_def.value in ("defensible", "defensible_with_caveats", "review_required")

        # ----------------------------------------------------------------
        # Step 9: Log audit entries
        # ----------------------------------------------------------------
        await log_audit(
            action="analysis_executed",
            agent="statistical_executor",
            decision=f"t-test completed: p={ttest_result['p_value']:.6f}",
            project_id=project_id,
        )
        await log_audit(
            action="analysis_executed",
            agent="statistical_executor",
            decision=f"logistic completed: treatment OR={logistic_result['parameters']['treatment']['odds_ratio']:.4f}",
            project_id=project_id,
        )
        await log_audit(
            action="analysis_executed",
            agent="statistical_executor",
            decision=f"cox completed: treatment HR={cox_result['hazard_ratios']['treatment']:.4f}",
            project_id=project_id,
        )
        await log_audit(
            action="researcher_decision",
            agent="researcher",
            decision="approved_all_analyses",
            project_id=project_id,
            researcher_decision="approved",
        )

        # ----------------------------------------------------------------
        # Step 10: SIMULATE RESTART
        # ----------------------------------------------------------------
        await _reinit_db()

        # ----------------------------------------------------------------
        # Step 11: VERIFY FULL RECOVERY
        # ----------------------------------------------------------------

        # Plan recovered
        loaded_plan = await plan_persistence.load_plan(plan_id)
        assert loaded_plan is not None
        assert loaded_plan["plan_id"] == plan_id
        assert len(loaded_plan["analyses"]) == 4

        # All approvals recovered
        for a in analyses:
            result_tuple = await plan_persistence.find_analysis(a["analysis_id"])
            assert result_tuple is not None, f"Analysis {a['analysis_id']} lost after restart"
            _, analysis = result_tuple
            assert analysis["approval"] == "approved"
            assert analysis["lifecycle"] == "approved"

        # All runs and results recovered
        for run_id, expected_result in [
            (run_id_ttest, ttest_result),
            (run_id_logistic, logistic_result),
            (run_id_cox, cox_result),
            (run_id_paired, paired_result),
        ]:
            loaded_run = await get_analysis_run(run_id)
            assert loaded_run is not None, f"Run {run_id} lost after restart"
            # Core numerical values identical
            for key in expected_result:
                if isinstance(expected_result[key], (int, float, str, bool, type(None))):
                    assert loaded_run["result"].get(key) == expected_result[key], \
                        f"Run {run_id} key {key} mismatch after restart"

        # Fingerprints recovered and match recomputation
        for run_id, expected_result in [
            (run_id_ttest, ttest_result),
            (run_id_logistic, logistic_result),
            (run_id_cox, cox_result),
            (run_id_paired, paired_result),
        ]:
            loaded_run = await get_analysis_run(run_id)
            recomputed = compute_result_fingerprint(expected_result)
            assert loaded_run["result_fingerprint"] == recomputed, \
                f"Run {run_id} fingerprint mismatch after restart"

        # Audit recovered
        entries = await get_audit_log(project_id=project_id)
        assert len(entries) >= 4
        actions = [e["action"] for e in entries]
        assert "analysis_executed" in actions
        assert "researcher_decision" in actions

        # Dataset identity stable (recomputed from file after restart)
        ds_identity_after = compute_dataset_identity(synthetic_csv)
        assert ds_identity_after["content_hash"] == ds_identity["content_hash"]
        assert ds_identity_after["schema_hash"] == ds_identity["schema_hash"]

    async def test_synthetic_dataset_deterministic(self, tmp_path):
        """Two generations with same seed produce identical data and identity."""
        df1 = generate_synthetic_cohort(seed=42)
        df2 = generate_synthetic_cohort(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

        p1 = tmp_path / "det1.csv"
        p2 = tmp_path / "det2.csv"
        save_synthetic_cohort(df1, p1)
        save_synthetic_cohort(df2, p2)
        id1 = compute_dataset_identity(p1)
        id2 = compute_dataset_identity(p2)
        assert id1["content_hash"] == id2["content_hash"]

    def test_synthetic_ground_truth_shapes(self, synthetic_csv):
        """Synthetic cohort matches ground-truth structural expectations."""
        df = pd.read_csv(synthetic_csv)
        assert len(df) == GROUND_TRUTH["n_observations"]
        assert set(GROUND_TRUTH["survival_groups"]) <= set(df["treatment_label"].unique())
        # Identifiers present
        assert "patient_id" in df.columns
        assert "mrn" in df.columns
        # Missing values injected
        assert df["hba1c"].isna().sum() > 0
        assert df["qol_score"].isna().sum() > 0
        # Survival data present
        assert df["time_to_event"].notna().all()
        assert set(df["event_observed"].unique()) <= {0, 1}
        assert df["event_observed"].sum() > 0  # at least some events
        # Paired measurements present
        assert "sbp_baseline" in df.columns
        assert "sbp_followup" in df.columns

    def test_analyses_produce_expected_directions(self, synthetic_csv):
        """Analyses recover ground-truth effect directions from synthetic data."""
        # Treatment effect on QoL is negative (-3.5 in ground truth)
        ttest = independent_t_test(
            synthetic_csv, outcome="qol_score", group="treatment",
            group1=0, group2=1,
        )
        # treatment=1 (B) should have lower QoL than treatment=0 (A)
        assert ttest["mean_group1"] > ttest["mean_group2"]

        # Treatment effect on SBP follow-up: B reduces SBP more
        from app.statistics.engine import paired_t_test
        paired = paired_t_test(synthetic_csv, outcome1="sbp_followup", outcome2="sbp_baseline")
        # Mean difference should be negative (SBP decreased)
        assert paired["mean_difference"] < 0

        # Cox: Treatment_B should have HR < 1 (reduces hazard)
        cox = cox_proportional_hazards(
            synthetic_csv, time_col="time_to_event",
            event_col="event_observed", covariates=["treatment"],
        )
        assert cox["hazard_ratios"]["treatment"] < 1.0

    def test_dataset_identity_order_invariant(self, tmp_path):
        """Dataset identity is invariant to row order."""
        df = generate_synthetic_cohort()
        p1 = tmp_path / "order1.csv"
        p2 = tmp_path / "order2.csv"
        save_synthetic_cohort(df, p1)
        # Shuffle rows
        df_shuffled = df.sample(frac=1, random_state=99).reset_index(drop=True)
        save_synthetic_cohort(df_shuffled, p2)
        id1 = compute_dataset_identity(p1)
        id2 = compute_dataset_identity(p2)
        assert id1["content_hash"] == id2["content_hash"]

    def test_different_data_different_identity(self, tmp_path):
        """Different datasets produce different identity hashes."""
        df1 = generate_synthetic_cohort(seed=42)
        df2 = generate_synthetic_cohort(seed=99)
        p1 = tmp_path / "d42.csv"
        p2 = tmp_path / "d99.csv"
        save_synthetic_cohort(df1, p1)
        save_synthetic_cohort(df2, p2)
        id1 = compute_dataset_identity(p1)
        id2 = compute_dataset_identity(p2)
        assert id1["content_hash"] != id2["content_hash"]


# ===========================================================================
# Cross-Method Consistency on Synthetic Data
# ===========================================================================

class TestSyntheticCrossMethodConsistency:
    """All methods on the same synthetic dataset produce stable, JSON-safe results."""

    def test_all_methods_json_roundtrip(self, synthetic_csv):
        """Every engine method on synthetic data survives JSON round-trip."""
        methods = [
            ("ttest", lambda: independent_t_test(
                synthetic_csv, outcome="qol_score", group="treatment", group1=0, group2=1)),
            ("logistic", lambda: logistic_regression(
                synthetic_csv, outcome="readmission_30d",
                predictors=["treatment", "heart_failure"])),
            ("linear", lambda: linear_regression(
                synthetic_csv, outcome="qol_score", predictors=["treatment", "age"])),
            ("mannwhitney", lambda: mann_whitney_u(
                synthetic_csv, outcome="qol_score", group="treatment", group1=0, group2=1)),
            ("cox", lambda: cox_proportional_hazards(
                synthetic_csv, time_col="time_to_event",
                event_col="event_observed", covariates=["treatment"])),
        ]
        for name, fn in methods:
            r1 = fn()
            assert "error" not in r1, f"{name} returned error: {r1.get('error')}"
            r2 = json.loads(json.dumps(r1, default=str))
            # Spot-check: first float/int key matches
            for k, v in r1.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    assert r2[k] == v, f"{name}.{k} mismatch after roundtrip"
                    break

    def test_survival_event_rate_reasonable(self, synthetic_csv):
        """Synthetic survival data has a reasonable event rate (>10%)."""
        df = pd.read_csv(synthetic_csv)
        event_rate = df["event_observed"].mean()
        assert 0.10 < event_rate < 0.90, f"Event rate {event_rate:.3f} outside expected range"

    def test_missing_data_present(self, synthetic_csv):
        """Synthetic cohort has missing values as documented."""
        df = pd.read_csv(synthetic_csv)
        assert df["hba1c"].isna().sum() > 0
        assert df["cholesterol"].isna().sum() > 0
        assert df["creatinine"].isna().sum() > 0
        assert df["qol_score"].isna().sum() > 0
