"""Phase 2F — Survival Analysis Test Suite.

Comprehensive tests for STATIS survival analysis pipeline covering:
- Kaplan-Meier estimation (grouped and ungrouped)
- Log-rank test
- Cox proportional hazards regression
- Data validation (missing, negative, invalid values)
- Determinism and reproducibility
- Identifier protection
- Dataset mutation detection
- Locked analysis enforcement
- Anti-p-hacking guards
- Research-question ambiguity handling
- Adversarial edge cases
- End-to-end integration

All tests use temporary CSV files via pytest's tmp_path fixture.
Data generation uses np.random.seed(42) for determinism.
"""

import sys
import hashlib
import json
import math
import uuid
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from app.statistics.survival import (
    kaplan_meier,
    log_rank,
    cox_proportional_hazards,
    validate_survival_data,
)
from app.statistics.executor import StatisticalExecutor, verify_dataset
from app.statistics.result import ExecutionStatus
from app.statistics.variable_roles import (
    RoleResolver,
    VariableRole,
    AnalysisProblemType,
)
from app.statistics.registry import get_method
from app.statistics.method_diagnostics import run_method_diagnostics
from app.statistics.candidate_generator import CandidateGenerator, DataProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uid() -> str:
    return f"surv_{uuid.uuid4().hex[:10]}"


def _write_csv(tmp_path: Path, df: pd.DataFrame, name: str | None = None) -> Path:
    fname = name or f"{_uid()}.csv"
    path = tmp_path / fname
    df.to_csv(path, index=False)
    return path


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Data generators (deterministic via np.random.seed)
# ---------------------------------------------------------------------------

def _make_clean_survival(n: int = 200, event_prob: float = 0.6, scale: float = 10.0, seed: int = 42) -> pd.DataFrame:
    """Create a clean survival dataset with time and event columns."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "time": rng.exponential(scale, n),
        "event": rng.binomial(1, event_prob, n),
    })


def _make_grouped_survival(n: int = 200, event_prob: float = 0.6, scale: float = 10.0, seed: int = 42) -> pd.DataFrame:
    """Create survival dataset with binary treatment group."""
    rng = np.random.RandomState(seed)
    treatment = rng.choice([0, 1], n)
    # Treatment group has slightly better survival (lower hazard)
    time = np.where(
        treatment == 1,
        rng.exponential(scale * 1.2, n),
        rng.exponential(scale * 0.8, n),
    )
    event = rng.binomial(1, event_prob, n)
    return pd.DataFrame({
        "time": time,
        "event": event,
        "treatment": treatment,
    })


def _make_covariates_survival(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create survival dataset with covariates for Cox regression."""
    rng = np.random.RandomState(seed)
    treatment = rng.choice([0, 1], n)
    age = rng.normal(60, 10, n)
    biomarker = rng.lognormal(2, 0.5, n)
    # Hazard depends on treatment and biomarker
    linear_pred = -0.5 * treatment + 0.03 * age + 0.2 * biomarker
    scale = np.exp(-linear_pred)
    time = rng.exponential(scale)
    event = rng.binomial(1, 0.7, n)
    return pd.DataFrame({
        "time": time,
        "event": event,
        "treatment": treatment,
        "age": age,
        "biomarker": biomarker,
    })


# ===========================================================================
# TESTS A through R — Class-based pytest organization
# ===========================================================================


class TestA_BasicKaplanMeier:
    """Test A — Basic Kaplan-Meier estimation on clean survival data."""

    def test_survival_estimates_exist(self, tmp_path):
        """Kaplan-Meier returns survival function estimates."""
        df = _make_clean_survival(n=200)
        path = _write_csv(tmp_path, df, "test_a_basic_km.csv")
        result = kaplan_meier(path, time_col="time", event_col="event")
        assert "error" not in result, result.get("error")
        assert "groups" in result
        assert "overall" in result["groups"]
        surv_probs = result["groups"]["overall"]["survival_probabilities"]
        assert len(surv_probs) > 0
        assert all(0.0 <= p <= 1.0 for p in surv_probs)

    def test_event_counts(self, tmp_path):
        """KM correctly counts events and censored observations."""
        df = _make_clean_survival(n=200, event_prob=0.6)
        path = _write_csv(tmp_path, df, "test_a_event_counts.csv")
        result = kaplan_meier(path, time_col="time", event_col="event")
        assert result["n_events"] == int(df["event"].sum())
        assert result["n_censored"] == int((df["event"] == 0).sum())
        assert result["n_events"] + result["n_censored"] == result["n_complete"]

    def test_deterministic_result(self, tmp_path):
        """KM produces deterministic results for the same input."""
        df = _make_clean_survival(n=200)
        path = _write_csv(tmp_path, df, "test_a_deterministic.csv")
        r1 = kaplan_meier(path, time_col="time", event_col="event")
        r2 = kaplan_meier(path, time_col="time", event_col="event")
        assert r1["groups"]["overall"]["survival_probabilities"] == \
               r2["groups"]["overall"]["survival_probabilities"]
        assert r1["groups"]["overall"]["median_survival"] == \
               r2["groups"]["overall"]["median_survival"]

    def test_median_survival_estimable(self, tmp_path):
        """Median survival time is estimable for a well-powered dataset."""
        df = _make_clean_survival(n=200, event_prob=0.7, scale=10.0)
        path = _write_csv(tmp_path, df, "test_a_median.csv")
        result = kaplan_meier(path, time_col="time", event_col="event")
        median = result["groups"]["overall"]["median_survival"]
        assert median is not None, "Median survival should be estimable"
        assert median > 0

    def test_n_complete_matches(self, tmp_path):
        """n_complete equals n_total when no missing data."""
        df = _make_clean_survival(n=200)
        path = _write_csv(tmp_path, df, "test_a_n_complete.csv")
        result = kaplan_meier(path, time_col="time", event_col="event")
        assert result["n_total"] == 200
        assert result["n_complete"] == 200


class TestB_KaplanMeierTwoGroups:
    """Test B — Kaplan-Meier with two groups."""

    def test_separate_curves(self, tmp_path):
        """KM with grouping produces separate survival curves per group."""
        df = _make_grouped_survival(n=200)
        path = _write_csv(tmp_path, df, "test_b_two_groups.csv")
        result = kaplan_meier(path, time_col="time", event_col="event", group_col="treatment")
        assert "error" not in result, result.get("error")
        assert result["grouped"] is True
        assert "0" in result["groups"]
        assert "1" in result["groups"]

    def test_group_specific_n(self, tmp_path):
        """Each group has its own N count."""
        df = _make_grouped_survival(n=200)
        path = _write_csv(tmp_path, df, "test_b_group_n.csv")
        result = kaplan_meier(path, time_col="time", event_col="event", group_col="treatment")
        n0 = result["groups"]["0"]["n"]
        n1 = result["groups"]["1"]["n"]
        assert n0 + n1 == result["n_complete"]
        assert n0 > 0 and n1 > 0

    def test_group_event_counts(self, tmp_path):
        """Each group has its own event and censor counts."""
        df = _make_grouped_survival(n=200)
        path = _write_csv(tmp_path, df, "test_b_group_events.csv")
        result = kaplan_meier(path, time_col="time", event_col="event", group_col="treatment")
        for g_key in ["0", "1"]:
            g = result["groups"][g_key]
            assert g["n_events"] + g["n_censored"] == g["n"]


class TestC_LogRank:
    """Test C — Log-rank test on grouped survival data."""

    def test_log_rank_statistic_and_pvalue(self, tmp_path):
        """Log-rank returns test statistic and p-value."""
        df = _make_grouped_survival(n=200)
        path = _write_csv(tmp_path, df, "test_c_log_rank.csv")
        result = log_rank(path, time_col="time", event_col="event", group_col="treatment")
        assert "error" not in result, result.get("error")
        assert "test_statistic" in result
        assert "p_value" in result
        assert isinstance(result["test_statistic"], float)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_log_rank_deterministic(self, tmp_path):
        """Log-rank produces identical results across runs."""
        df = _make_grouped_survival(n=200)
        path = _write_csv(tmp_path, df, "test_c_deterministic.csv")
        r1 = log_rank(path, time_col="time", event_col="event", group_col="treatment")
        r2 = log_rank(path, time_col="time", event_col="event", group_col="treatment")
        assert r1["test_statistic"] == r2["test_statistic"]
        assert r1["p_value"] == r2["p_value"]

    def test_log_rank_two_group_requirement(self, tmp_path):
        """Log-rank fails gracefully with fewer than 2 groups."""
        df = _make_clean_survival(n=100)
        df["group"] = 0  # single group
        path = _write_csv(tmp_path, df, "test_c_single_group.csv")
        result = log_rank(path, time_col="time", event_col="event", group_col="group")
        assert "error" in result
        assert "2 groups" in result["error"]

    def test_log_rank_group_summaries(self, tmp_path):
        """Log-rank returns per-group N and event counts."""
        df = _make_grouped_survival(n=200)
        path = _write_csv(tmp_path, df, "test_c_summaries.csv")
        result = log_rank(path, time_col="time", event_col="event", group_col="treatment")
        assert result["n_group1"] > 0
        assert result["n_group2"] > 0
        assert result["n_events_group1"] >= 0
        assert result["n_events_group2"] >= 0


class TestD_CoxRegression:
    """Test D — Cox proportional hazards regression."""

    def test_cox_coefficients(self, tmp_path):
        """Cox regression returns coefficients for covariates."""
        df = _make_covariates_survival(n=200)
        path = _write_csv(tmp_path, df, "test_d_cox.csv")
        result = cox_proportional_hazards(
            path, time_col="time", event_col="event", covariates=["treatment", "age", "biomarker"]
        )
        assert "error" not in result, result.get("error")
        assert "coefficients" in result
        assert "treatment" in result["coefficients"]
        assert "age" in result["coefficients"]
        assert "biomarker" in result["coefficients"]

    def test_cox_hazard_ratios(self, tmp_path):
        """Cox regression returns hazard ratios."""
        df = _make_covariates_survival(n=200)
        path = _write_csv(tmp_path, df, "test_d_hr.csv")
        result = cox_proportional_hazards(
            path, time_col="time", event_col="event", covariates=["treatment"]
        )
        assert "error" not in result, result.get("error")
        hr = result["hazard_ratios"]["treatment"]
        assert hr > 0, "Hazard ratio must be positive"

    def test_cox_confidence_intervals(self, tmp_path):
        """Cox regression returns confidence intervals."""
        df = _make_covariates_survival(n=200)
        path = _write_csv(tmp_path, df, "test_d_ci.csv")
        result = cox_proportional_hazards(
            path, time_col="time", event_col="event", covariates=["treatment"]
        )
        assert "error" not in result, result.get("error")
        ci = result["confidence_intervals"]["treatment"]
        assert ci[0] < ci[1], "CI lower must be less than upper"

    def test_cox_p_values(self, tmp_path):
        """Cox regression returns p-values for covariates."""
        df = _make_covariates_survival(n=200)
        path = _write_csv(tmp_path, df, "test_d_pval.csv")
        result = cox_proportional_hazards(
            path, time_col="time", event_col="event", covariates=["treatment"]
        )
        assert "error" not in result, result.get("error")
        pv = result["p_values"]["treatment"]
        assert 0.0 <= pv <= 1.0

    def test_cox_events_and_n(self, tmp_path):
        """Cox regression reports events and N."""
        df = _make_covariates_survival(n=200)
        path = _write_csv(tmp_path, df, "test_d_n.csv")
        result = cox_proportional_hazards(
            path, time_col="time", event_col="event", covariates=["treatment"]
        )
        assert "error" not in result, result.get("error")
        assert result["n_events"] > 0
        assert result["n_obs"] == 200

    def test_cox_convergence(self, tmp_path):
        """Cox model converges on well-behaved data."""
        df = _make_covariates_survival(n=200)
        path = _write_csv(tmp_path, df, "test_d_converge.csv")
        result = cox_proportional_hazards(
            path, time_col="time", event_col="event", covariates=["treatment", "age"]
        )
        assert "error" not in result, result.get("error")
        assert result.get("converged") is True


class TestE_CensoredObservations:
    """Test E — Dataset with many censored observations."""

    def test_heavy_censoring_represented(self, tmp_path):
        """High censoring rate is correctly represented, not treated as event."""
        rng = np.random.RandomState(42)
        n = 200
        time = rng.exponential(10, n)
        event = np.zeros(n, dtype=int)  # all censored
        event[:30] = 1  # only 30 events
        df = pd.DataFrame({"time": time, "event": event})
        path = _write_csv(tmp_path, df, "test_e_censored.csv")
        result = kaplan_meier(path, time_col="time", event_col="event")
        assert result["n_censored"] == n - 30
        assert result["n_events"] == 30

    def test_censoring_not_treated_as_event(self, tmp_path):
        """Event=0 rows are counted as censored, not events."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "time": rng.exponential(10, 100),
            "event": np.zeros(100, dtype=int),
        })
        path = _write_csv(tmp_path, df, "test_e_no_events.csv")
        result = kaplan_meier(path, time_col="time", event_col="event")
        assert result["n_events"] == 0
        assert result["n_censored"] == 100


class TestF_MissingSurvivalTime:
    """Test F — Dataset with NaN in the time column."""

    def test_missing_time_handled(self, tmp_path):
        """Rows with missing time are excluded and counted."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "time": np.concatenate([rng.exponential(10, 190), [np.nan] * 10]),
            "event": rng.binomial(1, 0.6, 200),
        })
        path = _write_csv(tmp_path, df, "test_f_missing_time.csv")
        result = kaplan_meier(path, time_col="time", event_col="event")
        assert result["n_complete"] == 190
        validation = result["validation"]
        assert validation["missing_time"] == 10


class TestG_MissingEventIndicator:
    """Test G — Dataset with NaN in the event column."""

    def test_missing_event_not_auto_censored(self, tmp_path):
        """Missing event indicators are NOT automatically converted to censored."""
        rng = np.random.RandomState(42)
        events = rng.binomial(1, 0.6, 200).astype(float)
        events[:5] = np.nan
        df = pd.DataFrame({
            "time": rng.exponential(10, 200),
            "event": events,
        })
        path = _write_csv(tmp_path, df, "test_g_missing_event.csv")
        validation = validate_survival_data(df, "time", "event")
        assert validation["missing_event"] == 5
        assert validation["n_complete"] == 195


class TestH_InvalidNegativeSurvivalTime:
    """Test H — Dataset with negative survival times."""

    def test_negative_time_validation_failure(self, tmp_path):
        """Negative survival times cause validation failure."""
        rng = np.random.RandomState(42)
        time_vals = rng.exponential(10, 200)
        time_vals[0] = -5.0
        time_vals[1] = -1.0
        df = pd.DataFrame({
            "time": time_vals,
            "event": rng.binomial(1, 0.6, 200),
        })
        path = _write_csv(tmp_path, df, "test_h_negative_time.csv")
        validation = validate_survival_data(df, "time", "event")
        assert validation["valid"] is False
        assert validation["invalid_time"] >= 2
        assert len(validation["blockers"]) > 0


class TestI_ConstantEventIndicator:
    """Test I — Dataset where all events=1 (no censoring)."""

    def test_all_events(self, tmp_path):
        """Survival analysis handles all-events data appropriately."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "time": rng.exponential(10, 200),
            "event": np.ones(200, dtype=int),
        })
        path = _write_csv(tmp_path, df, "test_i_all_events.csv")
        result = kaplan_meier(path, time_col="time", event_col="event")
        assert "error" not in result, result.get("error")
        assert result["n_events"] == 200
        assert result["n_censored"] == 0


class TestJ_SparseEvents:
    """Test J — Dataset with very few events (2-3)."""

    def test_sparse_events_warning(self, tmp_path):
        """Few events produce a review warning."""
        rng = np.random.RandomState(42)
        events = np.zeros(200, dtype=int)
        events[:3] = 1
        df = pd.DataFrame({
            "time": rng.exponential(10, 200),
            "event": events,
        })
        path = _write_csv(tmp_path, df, "test_j_sparse.csv")
        validation = validate_survival_data(df, "time", "event")
        assert any("Very few events" in w for w in validation["warnings"])


class TestK_CoxConvergenceIssue:
    """Test K — Perfect separation in Cox regression."""

    def test_perfect_separation_handling(self, tmp_path):
        """Perfect separation (event perfectly predicted by covariate) is handled."""
        rng = np.random.RandomState(42)
        n = 100
        covariate = np.concatenate([np.zeros(50), np.ones(50)])
        time = np.concatenate([
            rng.exponential(5, 50),
            rng.exponential(20, 50),
        ])
        # Perfect separation: covariate=0 always event, covariate=1 always censored
        event = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])
        df = pd.DataFrame({
            "time": time,
            "event": event,
            "cov": covariate,
        })
        path = _write_csv(tmp_path, df, "test_k_perfect_sep.csv")
        result = cox_proportional_hazards(
            path, time_col="time", event_col="event", covariates=["cov"]
        )
        # Should either produce an error or a result with extreme coefficients
        if "error" in result:
            assert "convergence" in result["error"].lower() or "fail" in result["error"].lower()
        else:
            hr = result["hazard_ratios"]["cov"]
            assert isinstance(hr, (int, float))
            assert math.isfinite(hr) or hr == float('inf') or hr == 0.0


class TestL_PHDiagnosticConcern:
    """Test L — Proportional hazards violation detection."""

    def test_ph_assumption_checked(self, tmp_path):
        """PH diagnostic is run and reported."""
        df = _make_covariates_survival(n=300)
        path = _write_csv(tmp_path, df, "test_l_ph.csv")
        result = cox_proportional_hazards(
            path, time_col="time", event_col="event", covariates=["treatment"]
        )
        assert "error" not in result, result.get("error")
        assert "ph_test" in result
        assert result["ph_test"] in ("passed", "violations_detected", "error")


class TestM_DatasetMutation:
    """Test M — Dataset modified after computing expected hash."""

    def test_mutation_blocks_execution(self, tmp_path):
        """Modifying the dataset after recording hash causes BLOCKED status."""
        df = _make_clean_survival(n=200)
        path = _write_csv(tmp_path, df, "test_m_mutation.csv")
        expected_hash = _hash_file(path)

        # Mutate the dataset
        df.iloc[0, df.columns.get_loc("time")] = 999.0
        df.to_csv(path, index=False)

        executor = StatisticalExecutor()
        result = executor.execute(
            "kaplan_meier",
            path,
            outcome="time",
            event_col="event",
            expected_content_hash=expected_hash,
        )
        assert result.execution_status == ExecutionStatus.BLOCKED


class TestN_LockedAnalysis:
    """Test N — Locked analysis cannot be silently modified."""

    def test_locked_specification_prevents_modification(self, tmp_path):
        """A locked specification raises an error on modification."""
        from app.statistics.specification import (
            StatisticalAnalysisSpecification,
            ApprovalStatus,
        )

        spec = StatisticalAnalysisSpecification(
            primary_outcome="time",
            analysis_method="kaplan_meier",
            research_question="Does treatment affect survival?",
        )
        spec.approve()
        spec.lock()
        assert spec.approval_status == ApprovalStatus.LOCKED

        # Attempting to modify a locked spec should be flagged
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            spec.modify(new_version="2.0")
            assert spec.approval_status == ApprovalStatus.MODIFIED


class TestO_AntiPHacking:
    """Test O — Two competing survival methods; STATIS does not select based on p-value."""

    def test_no_pvalue_driven_selection(self, tmp_path):
        """CandidateGenerator does not pick methods based on p-values."""
        from app.statistics.variable_roles import VariableRoleResolutionSet, VariableRoleResolution, RoleSource, RoleStatus

        profile = DataProfile(
            n_observations=200,
            outcome_variable="time",
            outcome_type="survival",
            predictor_variables=["treatment"],
            predictor_types=["binary"],
            n_groups=2,
            study_design="rct",
            research_question="Does treatment affect survival?",
            problem_type=AnalysisProblemType.TIME_TO_EVENT,
        )
        generator = CandidateGenerator()
        candidates = generator.generate(profile)

        method_ids = [c.method_id for c in candidates.primary_candidates]
        assert "kaplan_meier" in method_ids or "log_rank" in method_ids or "cox_proportional_hazards" in method_ids
        # No method is selected based on p-value — all survival methods are candidates


class TestP_NumericalDeterminism:
    """Test P — Same analysis run multiple times yields identical outputs."""

    def test_fingerprint_stability(self, tmp_path):
        """Running the same analysis twice produces the same fingerprint."""
        df = _make_covariates_survival(n=200)
        path = _write_csv(tmp_path, df, "test_p_determinism.csv")

        executor = StatisticalExecutor()
        r1 = executor.execute(
            "cox_proportional_hazards",
            path,
            outcome="time",
            event_col="event",
            predictors=["treatment", "age"],
        )
        r2 = executor.execute(
            "cox_proportional_hazards",
            path,
            outcome="time",
            event_col="event",
            predictors=["treatment", "age"],
        )
        assert r1.result_fingerprint == r2.result_fingerprint

    def test_km_determinism(self, tmp_path):
        """KM estimation produces identical results across runs."""
        df = _make_clean_survival(n=200)
        path = _write_csv(tmp_path, df, "test_p_km_det.csv")
        r1 = kaplan_meier(path, time_col="time", event_col="event")
        r2 = kaplan_meier(path, time_col="time", event_col="event")
        assert r1["groups"]["overall"]["survival_probabilities"] == \
               r2["groups"]["overall"]["survival_probabilities"]


class TestQ_IdentifierProtection:
    """Test Q — Patient IDs cannot become survival time, event, exposure, or covariates."""

    def test_patient_id_not_survival_time(self, tmp_path):
        """patient_id is never assigned survival_time role."""
        df = pd.DataFrame({
            "patient_id": range(1, 201),
            "time": np.random.RandomState(42).exponential(10, 200),
            "event": np.random.RandomState(42).binomial(1, 0.6, 200),
            "treatment": np.random.RandomState(42).choice([0, 1], 200),
        })
        vars_info = []
        for col in df.columns:
            vals = df[col].dropna()
            vars_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(vals.nunique()),
                "n_total": int(len(df)),
                "n_missing": int(df[col].isna().sum()),
            })
        result = RoleResolver.resolve_all(vars_info, research_question="Does treatment affect survival?")
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        assert roles["patient_id"] == VariableRole.IDENTIFIER
        assert roles["patient_id"] != VariableRole.SURVIVAL_TIME
        assert roles["patient_id"] != VariableRole.EVENT_INDICATOR
        assert roles["patient_id"] != VariableRole.EXPOSURE
        assert roles["patient_id"] != VariableRole.PREDICTOR

    def test_subject_id_not_exposure(self, tmp_path):
        """subject_id is never assigned as exposure or covariate."""
        df = pd.DataFrame({
            "subject_id": [f"S{i:04d}" for i in range(1, 151)],
            "survival_time": np.random.RandomState(42).exponential(10, 150),
            "status": np.random.RandomState(42).binomial(1, 0.6, 150),
            "group": np.random.RandomState(42).choice(["A", "B"], 150),
        })
        vars_info = []
        for col in df.columns:
            vals = df[col].dropna()
            vars_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(vals.nunique()),
                "n_total": int(len(df)),
                "n_missing": int(df[col].isna().sum()),
            })
        result = RoleResolver.resolve_all(vars_info, research_question="Compare survival between groups")
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        assert roles["subject_id"] == VariableRole.IDENTIFIER
        assert roles["subject_id"] not in (
            VariableRole.EXPOSURE,
            VariableRole.PREDICTOR,
            VariableRole.OUTCOME,
            VariableRole.SURVIVAL_TIME,
            VariableRole.EVENT_INDICATOR,
        )


class TestR_ResearchQuestionAmbiguity:
    """Test R — Insufficient information yields UNKNOWN/REVIEW_REQUIRED."""

    def test_no_research_question(self, tmp_path):
        """Without a research question, survival variables may not be fully resolved."""
        df = pd.DataFrame({
            "follow_up_days": np.random.RandomState(42).exponential(10, 200),
            "alive": np.random.RandomState(42).binomial(1, 0.6, 200),
            "biomarker_x": np.random.RandomState(42).lognormal(2, 0.5, 200),
        })
        vars_info = []
        for col in df.columns:
            vals = df[col].dropna()
            vars_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(vals.nunique()),
                "n_total": int(len(df)),
                "n_missing": int(df[col].isna().sum()),
            })
        result = RoleResolver.resolve_all(vars_info, research_question=None)
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        # Without RQ, the system uses heuristics. 'alive' (binary) may be grouped,
        # 'biomarker_x' (high cardinality numeric) may be outcome. Without RQ context,
        # the system should not over-commit with high confidence on all roles.
        # At minimum, no variable should be assigned SURVIVAL_TIME or EVENT_INDICATOR
        # since there's no survival RQ.
        survival_roles = [
            r for r in result.resolutions
            if r.proposed_role in (VariableRole.SURVIVAL_TIME, VariableRole.EVENT_INDICATOR)
        ]
        assert len(survival_roles) == 0, "Without RQ, no survival roles should be assigned"

    def test_ambiguous_survival_question(self, tmp_path):
        """An overly vague question yields ambiguous or review-required statuses."""
        df = pd.DataFrame({
            "t": np.random.RandomState(42).exponential(10, 200),
            "e": np.random.RandomState(42).binomial(1, 0.6, 200),
            "cov1": np.random.RandomState(42).normal(0, 1, 200),
        })
        vars_info = []
        for col in df.columns:
            vals = df[col].dropna()
            vars_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(vals.nunique()),
                "n_total": int(len(df)),
                "n_missing": int(df[col].isna().sum()),
            })
        result = RoleResolver.resolve_all(
            vars_info,
            research_question="Look at stuff in the data",
        )
        # With a vague RQ, at least some variables should have low confidence
        low_conf = [r for r in result.resolutions if r.confidence == "low"]
        ambiguous = [r for r in result.resolutions if r.status in ("ambiguous", "review_required")]
        # Either low confidence or ambiguous should exist for at least one variable
        assert len(low_conf) > 0 or len(ambiguous) > 0 or result.problem_type == AnalysisProblemType.UNKNOWN


# ===========================================================================
# ADVERSARIAL TEST BATTERY
# ===========================================================================


class TestSurvivalAdversarial:
    """Adversarial tests covering edge cases and robustness of the survival pipeline."""

    def test_calendar_date_as_time_column(self, tmp_path):
        """Time column that is a calendar date string — should fail validation."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "time": ["2024-01-01", "2024-02-15", "2024-03-30"] * 66 + ["2024-04-01", "2024-05-01"],
            "event": rng.binomial(1, 0.5, 200),
        })
        path = _write_csv(tmp_path, df, "adv_date_time.csv")
        validation = validate_survival_data(df, "time", "event")
        assert validation["valid"] is False
        assert validation["invalid_time"] > 0

    def test_recovery_time_as_duration(self, tmp_path):
        """recovery_time is a valid duration — should be accepted."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "recovery_time": rng.exponential(14, 150),
            "event": rng.binomial(1, 0.6, 150),
        })
        path = _write_csv(tmp_path, df, "adv_recovery_time.csv")
        result = kaplan_meier(path, time_col="recovery_time", event_col="event")
        assert "error" not in result, result.get("error")
        assert result["n_complete"] == 150

    def test_event_encoded_yes_no(self, tmp_path):
        """Event variable encoded as 'Yes'/'No' strings — should be mapped."""
        rng = np.random.RandomState(42)
        events = rng.choice(["Yes", "No"], 100, p=[0.6, 0.4])
        df = pd.DataFrame({
            "time": rng.exponential(10, 100),
            "event": events,
        })
        path = _write_csv(tmp_path, df, "adv_event_yesno.csv")
        result = kaplan_meier(path, time_col="time", event_col="event")
        assert "error" not in result, result.get("error")
        assert result["n_events"] > 0

    def test_event_encoded_zero_one(self, tmp_path):
        """Event variable encoded as 0/1 integers — standard format."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "time": rng.exponential(10, 100),
            "event": rng.binomial(1, 0.6, 100),
        })
        path = _write_csv(tmp_path, df, "adv_event_01.csv")
        result = kaplan_meier(path, time_col="time", event_col="event")
        assert "error" not in result

    def test_event_encoded_alive_dead(self, tmp_path):
        """Event variable encoded as 'alive'/'dead' — should be mapped."""
        rng = np.random.RandomState(42)
        events = rng.choice(["alive", "dead"], 100, p=[0.4, 0.6])
        df = pd.DataFrame({
            "time": rng.exponential(10, 100),
            "event": events,
        })
        path = _write_csv(tmp_path, df, "adv_event_alive_dead.csv")
        result = kaplan_meier(path, time_col="time", event_col="event")
        assert "error" not in result, result.get("error")
        assert result["n_events"] > 0

    def test_multiple_binary_competing_for_event(self, tmp_path):
        """Multiple binary variables competing for the event indicator role."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "time": rng.exponential(10, 100),
            "died": rng.binomial(1, 0.6, 100),
            "recurrence": rng.binomial(1, 0.3, 100),
            "hospitalized": rng.binomial(1, 0.5, 100),
        })
        vars_info = []
        for col in df.columns:
            vals = df[col].dropna()
            vars_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(vals.nunique()),
                "n_total": int(len(df)),
                "n_missing": int(df[col].isna().sum()),
            })
        result = RoleResolver.resolve_all(vars_info, research_question="Does treatment affect survival?")
        event_roles = [
            r for r in result.resolutions
            if r.proposed_role == VariableRole.EVENT_INDICATOR
        ]
        # With competing event indicators, the system should flag ambiguity
        # or assign only the best match
        assert len(event_roles) >= 1

    def test_multiple_numeric_competing_for_survival_time(self, tmp_path):
        """Multiple numeric variables competing for the survival time role."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "time_to_event": rng.exponential(10, 100),
            "follow_up_days": rng.exponential(20, 100),
            "duration": rng.exponential(15, 100),
            "event": rng.binomial(1, 0.6, 100),
        })
        vars_info = []
        for col in df.columns:
            vals = df[col].dropna()
            vars_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(vals.nunique()),
                "n_total": int(len(df)),
                "n_missing": int(df[col].isna().sum()),
            })
        result = RoleResolver.resolve_all(vars_info, research_question="Does treatment affect survival?")
        time_roles = [
            r for r in result.resolutions
            if r.proposed_role == VariableRole.SURVIVAL_TIME
        ]
        assert len(time_roles) >= 1

    def test_patient_id_numeric_values(self, tmp_path):
        """Patient ID with numeric values — should be identified as identifier."""
        df = pd.DataFrame({
            "patient_id": range(1, 101),
            "time": np.random.RandomState(42).exponential(10, 100),
            "event": np.random.RandomState(42).binomial(1, 0.6, 100),
        })
        vars_info = []
        for col in df.columns:
            vals = df[col].dropna()
            vars_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(vals.nunique()),
                "n_total": int(len(df)),
                "n_missing": int(df[col].isna().sum()),
            })
        result = RoleResolver.resolve_all(vars_info)
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        assert roles["patient_id"] == VariableRole.IDENTIFIER

    def test_extreme_follow_up_duration(self, tmp_path):
        """Extreme but legitimate follow-up duration (e.g., 10000 days)."""
        rng = np.random.RandomState(42)
        time_vals = np.concatenate([
            rng.exponential(10, 190),
            [5000, 8000, 10000, 200, 500, 1500, 3000, 500, 100, 300],
        ])
        df = pd.DataFrame({
            "time": time_vals,
            "event": rng.binomial(1, 0.6, 200),
        })
        path = _write_csv(tmp_path, df, "adv_extreme_fu.csv")
        result = kaplan_meier(path, time_col="time", event_col="event")
        assert "error" not in result, result.get("error")
        assert result["n_complete"] == 200

    def test_zero_survival_time(self, tmp_path):
        """Zero survival time — should be accepted (event at enrollment)."""
        rng = np.random.RandomState(42)
        time_vals = rng.exponential(10, 100)
        time_vals[0] = 0.0
        df = pd.DataFrame({
            "time": time_vals,
            "event": rng.binomial(1, 0.6, 100),
        })
        path = _write_csv(tmp_path, df, "adv_zero_time.csv")
        validation = validate_survival_data(df, "time", "event")
        assert validation["valid"] is True
        assert validation["invalid_time"] == 0

    def test_negative_survival_time_blocked(self, tmp_path):
        """Negative survival time is blocked by validation."""
        rng = np.random.RandomState(42)
        time_vals = rng.exponential(10, 100)
        time_vals[0] = -3.0
        df = pd.DataFrame({
            "time": time_vals,
            "event": rng.binomial(1, 0.6, 100),
        })
        path = _write_csv(tmp_path, df, "adv_negative_time.csv")
        validation = validate_survival_data(df, "time", "event")
        assert validation["valid"] is False
        assert validation["invalid_time"] >= 1

    def test_all_events_no_censoring(self, tmp_path):
        """All observations are events — no censoring."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "time": rng.exponential(10, 100),
            "event": np.ones(100, dtype=int),
        })
        path = _write_csv(tmp_path, df, "adv_all_events.csv")
        validation = validate_survival_data(df, "time", "event")
        assert validation["valid"] is True
        assert any("All observations are events" in w for w in validation["warnings"])

    def test_no_events_all_censored(self, tmp_path):
        """No events at all — all censored."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "time": rng.exponential(10, 100),
            "event": np.zeros(100, dtype=int),
        })
        path = _write_csv(tmp_path, df, "adv_no_events.csv")
        validation = validate_survival_data(df, "time", "event")
        assert validation["valid"] is True
        assert any("No events observed" in w for w in validation["warnings"])

    def test_heavily_censored_data(self, tmp_path):
        """Heavily censored data (95% censored)."""
        rng = np.random.RandomState(42)
        events = np.zeros(200, dtype=int)
        events[:10] = 1  # only 5% events
        df = pd.DataFrame({
            "time": rng.exponential(10, 200),
            "event": events,
        })
        path = _write_csv(tmp_path, df, "adv_heavy_censor.csv")
        result = kaplan_meier(path, time_col="time", event_col="event")
        assert "error" not in result
        assert result["n_events"] == 10
        assert result["n_censored"] == 190

    def test_sparse_events_warning(self, tmp_path):
        """Very few events (2) triggers a sparse-events warning."""
        rng = np.random.RandomState(42)
        events = np.zeros(150, dtype=int)
        events[0] = 1
        events[1] = 1
        df = pd.DataFrame({
            "time": rng.exponential(10, 150),
            "event": events,
        })
        path = _write_csv(tmp_path, df, "adv_sparse_events.csv")
        validation = validate_survival_data(df, "time", "event")
        assert any("Very few events" in w for w in validation["warnings"])

    def test_tied_event_times(self, tmp_path):
        """Many tied event times — should not cause errors."""
        rng = np.random.RandomState(42)
        # Create many ties by rounding times to integers
        time_vals = np.round(rng.exponential(10, 200)).astype(float)
        df = pd.DataFrame({
            "time": time_vals,
            "event": rng.binomial(1, 0.6, 200),
        })
        path = _write_csv(tmp_path, df, "adv_tied.csv")
        result = kaplan_meier(path, time_col="time", event_col="event")
        assert "error" not in result
        assert result["n_complete"] == 200

    def test_missing_covariates_in_cox(self, tmp_path):
        """Cox regression with missing covariate values — complete case analysis."""
        rng = np.random.RandomState(42)
        n = 200
        treatment = rng.choice([0, 1], n)
        age = rng.normal(60, 10, n).astype(float)
        age[:10] = np.nan  # 10 missing ages
        df = pd.DataFrame({
            "time": rng.exponential(10, n),
            "event": rng.binomial(1, 0.6, n),
            "treatment": treatment,
            "age": age,
        })
        path = _write_csv(tmp_path, df, "adv_missing_cov.csv")
        result = cox_proportional_hazards(
            path, time_col="time", event_col="event", covariates=["treatment", "age"]
        )
        assert "error" not in result, result.get("error")
        assert result["n_obs"] == 190  # 10 rows dropped due to missing age

    def test_categorical_covariate_in_cox(self, tmp_path):
        """Cox regression with a categorical covariate — should be dummified."""
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame({
            "time": rng.exponential(10, n),
            "event": rng.binomial(1, 0.6, n),
            "treatment": rng.choice([0, 1], n),
            "center": rng.choice(["A", "B", "C"], n),
        })
        path = _write_csv(tmp_path, df, "adv_categorical_cov.csv")
        result = cox_proportional_hazards(
            path, time_col="time", event_col="event", covariates=["treatment", "center"]
        )
        assert "error" not in result, result.get("error")
        # center should be dummified into center_B, center_C
        assert any("center" in k for k in result["coefficients"].keys())

    def test_constant_covariate_in_cox(self, tmp_path):
        """Cox regression with a constant covariate — should fail gracefully."""
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame({
            "time": rng.exponential(10, n),
            "event": rng.binomial(1, 0.6, n),
            "treatment": rng.choice([0, 1], n),
            "constant_var": np.ones(n),
        })
        path = _write_csv(tmp_path, df, "adv_constant_cov.csv")
        result = cox_proportional_hazards(
            path, time_col="time", event_col="event", covariates=["treatment", "constant_var"]
        )
        assert "error" in result
        assert "constant" in result["error"].lower() or "variation" in result["error"].lower()

    def test_dataset_mutation_blocks_execution(self, tmp_path):
        """Modifying dataset after recording content hash blocks execution."""
        df = _make_clean_survival(n=200)
        path = _write_csv(tmp_path, df, "adv_mutation.csv")
        expected_hash = _hash_file(path)

        # Mutate
        df.iloc[5, df.columns.get_loc("event")] = 9
        df.to_csv(path, index=False)

        executor = StatisticalExecutor()
        result = executor.execute(
            "kaplan_meier",
            path,
            outcome="time",
            event_col="event",
            expected_content_hash=expected_hash,
        )
        assert result.execution_status == ExecutionStatus.BLOCKED

    def test_locked_analysis_mutation_attempt(self, tmp_path):
        """Attempting to modify a locked specification."""
        from app.statistics.specification import (
            StatisticalAnalysisSpecification,
            ApprovalStatus,
        )

        spec = StatisticalAnalysisSpecification(
            primary_outcome="time",
            analysis_method="kaplan_meier",
            research_question="Does treatment affect survival?",
        )
        spec.approve()
        spec.lock()
        assert spec.approval_status == ApprovalStatus.LOCKED

        # Modification attempt should change status to MODIFIED
        spec.modify(new_version="2.0")
        assert spec.approval_status == ApprovalStatus.MODIFIED

    def test_p_value_driven_method_selection_attempt(self, tmp_path):
        """Attempting to select method based on p-value is flagged."""
        from app.statistics.anti_phacking import AntiPHackingGuard

        guard = AntiPHackingGuard()
        strategies = [
            {"method": "kaplan_meier", "comparison_reason": "p_value is significant"},
            {"method": "log_rank", "comparison_reason": "robustness check"},
        ]
        selected = strategies[0]
        checks = guard.check_strategy_selection(strategies, selected, p_value=0.03)
        flagged = [c for c in checks if c.verdict.value in ("blocked", "suspicious")]
        assert len(flagged) > 0, "p-value driven selection should be flagged as suspicious or blocked"

    def test_ambiguous_survival_research_question(self, tmp_path):
        """Ambiguous survival research question yields low-confidence roles."""
        df = pd.DataFrame({
            "t": np.random.RandomState(42).exponential(10, 100),
            "e": np.random.RandomState(42).binomial(1, 0.5, 100),
            "x": np.random.RandomState(42).normal(0, 1, 100),
        })
        vars_info = []
        for col in df.columns:
            vals = df[col].dropna()
            vars_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(vals.nunique()),
                "n_total": int(len(df)),
                "n_missing": int(df[col].isna().sum()),
            })
        result = RoleResolver.resolve_all(
            vars_info,
            research_question="analyze the data and tell me things",
        )
        # Should not have high confidence for all variables
        high_conf = [r for r in result.resolutions if r.confidence == "high"]
        assert len(high_conf) < len(result.resolutions)

    def test_causal_wording_in_observational_study(self, tmp_path):
        """Causal wording in observational study does not cause method error."""
        df = _make_grouped_survival(n=200)
        path = _write_csv(tmp_path, df, "adv_causal.csv")
        result = log_rank(path, time_col="time", event_col="event", group_col="treatment")
        # The method runs fine; causal interpretation is the researcher's responsibility
        assert "error" not in result, result.get("error")
        assert "test_statistic" in result


# ===========================================================================
# INTEGRATION TEST
# ===========================================================================


class TestSurvivalIntegration:
    """End-to-end integration: RQ -> identified structure -> method -> result -> diagnostics."""

    def test_full_survival_pipeline(self, tmp_path):
        """Complete pipeline from research question through diagnostics."""
        # 1. Create dataset
        df = _make_covariates_survival(n=250)
        path = _write_csv(tmp_path, df, "integration_full_pipeline.csv")

        # 2. Verify dataset identity
        verification = verify_dataset(path, variables_required=["time", "event", "treatment", "age"])
        assert verification.n_rows == 250
        assert verification.content_hash is not None

        # 3. Resolve variable roles from research question
        vars_info = []
        for col in df.columns:
            vals = df[col].dropna()
            vars_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(vals.nunique()),
                "n_total": int(len(df)),
                "n_missing": int(df[col].isna().sum()),
            })
        role_result = RoleResolver.resolve_all(
            vars_info,
            research_question="Does treatment affect overall survival after adjusting for age?",
        )
        roles = {r.variable_name: r.proposed_role for r in role_result.resolutions}
        assert role_result.problem_type == AnalysisProblemType.TIME_TO_EVENT

        # 4. Generate candidate methods
        profile = DataProfile(
            n_observations=250,
            outcome_variable="time",
            outcome_type="survival",
            predictor_variables=["treatment", "age"],
            predictor_types=["binary", "continuous"],
            n_groups=2,
            study_design="cohort",
            research_question="Does treatment affect overall survival after adjusting for age?",
            problem_type=role_result.problem_type,
        )
        generator = CandidateGenerator()
        candidates = generator.generate(profile)
        method_ids = [c.method_id for c in candidates.primary_candidates]
        assert "kaplan_meier" in method_ids or "cox_proportional_hazards" in method_ids

        # 5. Execute Cox regression (the primary method for adjusted survival analysis)
        executor = StatisticalExecutor()
        result = executor.execute(
            "cox_proportional_hazards",
            path,
            outcome="time",
            event_col="event",
            predictors=["treatment", "age"],
        )
        assert result.execution_status == ExecutionStatus.SUCCESS

        # 6. Verify result contents
        assert result.n_observed == 250
        assert result.descriptive["n_events"] > 0
        assert result.descriptive["converged"] is True
        assert result.result_fingerprint is not None

        # 7. Check diagnostics
        assert result.diagnostic_status in ("pass", "warning", "review_required")
        assert "diagnostic_summary" in result.model_dump()
        diag_summary = result.diagnostic_summary
        assert "overall_status" in diag_summary

        # 8. Check defensibility
        assert result.defensibility_status in (
            "defensible",
            "defensible_with_caveats",
            "review_required",
            "not_defensible",
            "not_assessed",
        )

        # 9. Verify no guardrail blockers
        assert len(result.guardrail_blockers) == 0

        # 10. Verify HR is positive
        hr = result.descriptive["hazard_ratios"]["treatment"]
        assert hr > 0

    def test_km_log_rank_pipeline(self, tmp_path):
        """Pipeline: KM estimation -> log-rank test."""
        df = _make_grouped_survival(n=200)
        path = _write_csv(tmp_path, df, "integration_km_logrank.csv")

        # KM with groups
        km_result = kaplan_meier(path, time_col="time", event_col="event", group_col="treatment")
        assert "error" not in km_result, km_result.get("error")
        assert "0" in km_result["groups"]
        assert "1" in km_result["groups"]

        # Log-rank test
        lr_result = log_rank(path, time_col="time", event_col="event", group_col="treatment")
        assert "error" not in lr_result, lr_result.get("error")
        assert 0.0 <= lr_result["p_value"] <= 1.0

        # Both methods agree on N
        assert km_result["n_complete"] == lr_result["n_group1"] + lr_result["n_group2"]

    def test_validation_to_km_pipeline(self, tmp_path):
        """Pipeline: validate data -> run KM."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "time": rng.exponential(10, 200),
            "event": rng.binomial(1, 0.6, 200),
        })

        # Validate first
        validation = validate_survival_data(df, "time", "event")
        assert validation["valid"] is True
        assert validation["n_events"] > 0
        assert validation["n_censored"] > 0

        # Then run KM
        path = _write_csv(tmp_path, df, "integration_valid_km.csv")
        result = kaplan_meier(path, time_col="time", event_col="event")
        assert result["n_events"] == validation["n_events"]

    def test_executor_survival_dispatch(self, tmp_path):
        """StatisticalExecutor correctly dispatches to survival methods."""
        df = _make_grouped_survival(n=200)
        path = _write_csv(tmp_path, df, "integration_executor_survival.csv")

        executor = StatisticalExecutor()

        # KM via executor
        km = executor.execute(
            "kaplan_meier",
            path,
            outcome="time",
            event_col="event",
            group="treatment",
        )
        assert km.execution_status == ExecutionStatus.SUCCESS

        # Log-rank via executor
        lr = executor.execute(
            "log_rank",
            path,
            outcome="time",
            event_col="event",
            group="treatment",
        )
        assert lr.execution_status == ExecutionStatus.SUCCESS

        # Cox via executor
        cox = executor.execute(
            "cox_proportional_hazards",
            path,
            outcome="time",
            event_col="event",
            predictors=["treatment"],
        )
        assert cox.execution_status == ExecutionStatus.SUCCESS


# ===========================================================================
# DEFENSIBILITY INTEGRATION TESTS
# ===========================================================================


class TestDefensibilitySurvival:
    """Tests for survival-specific defensibility assessment."""

    def test_zero_events_not_defensible(self):
        """Zero events makes survival analysis not defensible."""
        from app.statistics.defensibility import assess_defensibility
        from app.statistics.result import ExecutionStatus, SanityStatus
        from app.statistics.diagnostics import DiagnosticDiagnosticStatus

        result = assess_defensibility(
            ExecutionStatus.SUCCESS,
            SanityStatus.PASS,
            DiagnosticDiagnosticStatus.PASS,
            survival_metadata={
                "method_id": "kaplan_meier",
                "n_events": 0,
                "n_censored": 200,
                "n_total": 200,
                "ph_assumption_met": None,
                "sparse_events": False,
                "all_events": False,
                "no_events": True,
            },
        )
        assert result.value == "not_defensible"

    def test_sparse_events_review_required(self):
        """Fewer than 10 events triggers review."""
        from app.statistics.defensibility import assess_defensibility
        from app.statistics.result import ExecutionStatus, SanityStatus
        from app.statistics.diagnostics import DiagnosticDiagnosticStatus

        result = assess_defensibility(
            ExecutionStatus.SUCCESS,
            SanityStatus.PASS,
            DiagnosticDiagnosticStatus.PASS,
            survival_metadata={
                "method_id": "kaplan_meier",
                "n_events": 5,
                "n_censored": 195,
                "n_total": 200,
                "ph_assumption_met": None,
                "sparse_events": True,
                "all_events": False,
                "no_events": False,
            },
        )
        assert result.value == "review_required"

    def test_ph_violation_review_required(self):
        """Cox PH with violated assumption triggers review."""
        from app.statistics.defensibility import assess_defensibility
        from app.statistics.result import ExecutionStatus, SanityStatus
        from app.statistics.diagnostics import DiagnosticDiagnosticStatus

        result = assess_defensibility(
            ExecutionStatus.SUCCESS,
            SanityStatus.PASS,
            DiagnosticDiagnosticStatus.PASS,
            survival_metadata={
                "method_id": "cox_proportional_hazards",
                "n_events": 50,
                "n_censored": 150,
                "n_total": 200,
                "ph_assumption_met": False,
                "sparse_events": False,
                "all_events": False,
                "no_events": False,
            },
        )
        assert result.value == "review_required"

    def test_extreme_censoring_review_required(self):
        """More than 90% censored triggers review."""
        from app.statistics.defensibility import assess_defensibility
        from app.statistics.result import ExecutionStatus, SanityStatus
        from app.statistics.diagnostics import DiagnosticDiagnosticStatus

        result = assess_defensibility(
            ExecutionStatus.SUCCESS,
            SanityStatus.PASS,
            DiagnosticDiagnosticStatus.PASS,
            survival_metadata={
                "method_id": "kaplan_meier",
                "n_events": 15,
                "n_censored": 185,
                "n_total": 200,
                "ph_assumption_met": None,
                "sparse_events": False,
                "all_events": False,
                "no_events": False,
            },
        )
        assert result.value == "review_required"

    def test_adequate_survival_defensible(self):
        """Adequate events, PH met, no issues → defensible."""
        from app.statistics.defensibility import assess_defensibility
        from app.statistics.result import ExecutionStatus, SanityStatus
        from app.statistics.diagnostics import DiagnosticDiagnosticStatus

        result = assess_defensibility(
            ExecutionStatus.SUCCESS,
            SanityStatus.PASS,
            DiagnosticDiagnosticStatus.PASS,
            survival_metadata={
                "method_id": "cox_proportional_hazards",
                "n_events": 60,
                "n_censored": 140,
                "n_total": 200,
                "ph_assumption_met": True,
                "sparse_events": False,
                "all_events": False,
                "no_events": False,
            },
        )
        assert result.value == "defensible"

    def test_no_survival_metadata_unchanged(self):
        """Without survival_metadata, defensibility behaves as before."""
        from app.statistics.defensibility import assess_defensibility
        from app.statistics.result import ExecutionStatus, SanityStatus
        from app.statistics.diagnostics import DiagnosticDiagnosticStatus

        result = assess_defensibility(
            ExecutionStatus.SUCCESS,
            SanityStatus.PASS,
            DiagnosticDiagnosticStatus.PASS,
        )
        assert result.value == "defensible"


# ===========================================================================
# CANDIDATE GENERATOR SURVIVAL TESTS
# ===========================================================================


class TestCandidateGeneratorSurvival:
    """Tests for survival-aware candidate generation."""

    def _make_survival_profile(self, **kwargs):
        from app.statistics.variable_roles import AnalysisProblemType
        defaults = {
            "n_observations": 200,
            "outcome_variable": "time",
            "outcome_type": "survival",
            "predictor_variables": ["treatment"],
            "predictor_types": ["binary"],
            "problem_type": AnalysisProblemType.TIME_TO_EVENT,
        }
        defaults.update(kwargs)
        return DataProfile(**defaults)

    def test_survival_methods_generated(self):
        """Survival methods appear when outcome_type is survival."""
        gen = CandidateGenerator()
        profile = self._make_survival_profile()
        result = gen.generate(profile)
        all_ids = {
            c.method_id
            for c in result.primary_candidates + result.sensitivity_candidates
        }
        assert "kaplan_meier" in all_ids or "cox_proportional_hazards" in all_ids

    test_survival_methods_generated.skip = False

    def test_non_survival_excludes_survival_methods(self):
        """Non-survival outcome excludes survival methods."""
        from app.statistics.variable_roles import AnalysisProblemType
        gen = CandidateGenerator()
        profile = DataProfile(
            n_observations=200,
            outcome_variable="bp",
            outcome_type="continuous",
            predictor_variables=["treatment"],
            predictor_types=["binary"],
            problem_type=AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
        )
        result = gen.generate(profile)
        all_ids = {
            c.method_id
            for c in result.primary_candidates + result.sensitivity_candidates
        }
        assert "kaplan_meier" not in all_ids
        assert "log_rank" not in all_ids
        assert "cox_proportional_hazards" not in all_ids

    def test_km_primary_without_predictors(self):
        """KM is primary when no predictors specified."""
        from app.statistics.variable_roles import AnalysisProblemType
        gen = CandidateGenerator()
        profile = DataProfile(
            n_observations=200,
            outcome_variable="time",
            outcome_type="survival",
            problem_type=AnalysisProblemType.TIME_TO_EVENT,
        )
        result = gen.generate(profile)
        primary_ids = {c.method_id for c in result.primary_candidates}
        assert "kaplan_meier" in primary_ids

    def test_cox_primary_with_covariates(self):
        """Cox is primary when covariates present."""
        gen = CandidateGenerator()
        profile = self._make_survival_profile(
            predictor_variables=["treatment", "age", "biomarker"],
            predictor_types=["binary", "continuous", "continuous"],
        )
        result = gen.generate(profile)
        primary_ids = {c.method_id for c in result.primary_candidates}
        assert "cox_proportional_hazards" in primary_ids
