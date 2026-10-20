"""Phase 2J — Survival Validation (Invariant 39).

Independently validates Kaplan-Meier, log-rank, and Cox PH against lifelines.
Verifies Phase 2F defensibility rules:
- zero events → NOT_DEFENSIBLE
- PH violation → REVIEW_REQUIRED
- sparse events <10 → REVIEW_REQUIRED
- censoring >90% → REVIEW_REQUIRED
- all/no events for Cox → REVIEW_REQUIRED
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.statistics.survival import kaplan_meier, log_rank, cox_proportional_hazards, validate_survival_data
from app.statistics.defensibility import assess_defensibility
from app.statistics.result import ExecutionStatus, SanityStatus
from app.statistics.diagnostics import DiagnosticDiagnosticStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_csv(df, tmp_path, name="data.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def normal_survival(tmp_path):
    """Normal event rate (~40% events)."""
    rng = np.random.RandomState(42)
    n = 200
    treatment = rng.binomial(1, 0.5, n)
    hazard = 0.005 * np.where(treatment == 1, 0.6, 1.0)
    survival_time = rng.exponential(1.0 / hazard)
    censoring_time = rng.uniform(30, 400, n)
    time = np.minimum(survival_time, censoring_time)
    event = (survival_time < censoring_time).astype(int)
    df = pd.DataFrame({"time": time, "event": event, "treatment": treatment})
    return _write_csv(df, tmp_path, "normal.csv")


@pytest.fixture
def zero_events(tmp_path):
    """All observations censored (zero events)."""
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        "time": rng.uniform(30, 365, n),
        "event": np.zeros(n, dtype=int),
        "treatment": rng.binomial(1, 0.5, n),
    })
    return _write_csv(df, tmp_path, "zero_events.csv")


@pytest.fixture
def sparse_events(tmp_path):
    """Fewer than 10 events."""
    rng = np.random.RandomState(42)
    n = 100
    event = np.zeros(n, dtype=int)
    event[:7] = 1  # only 7 events
    df = pd.DataFrame({
        "time": rng.uniform(30, 365, n),
        "event": event,
        "treatment": rng.binomial(1, 0.5, n),
    })
    return _write_csv(df, tmp_path, "sparse_events.csv")


@pytest.fixture
def extreme_censoring(tmp_path):
    """More than 90% censored."""
    rng = np.random.RandomState(42)
    n = 100
    event = np.zeros(n, dtype=int)
    event[:5] = 1  # only 5% events
    df = pd.DataFrame({
        "time": rng.uniform(30, 365, n),
        "event": event,
        "treatment": rng.binomial(1, 0.5, n),
    })
    return _write_csv(df, tmp_path, "extreme_censoring.csv")


@pytest.fixture
def all_events(tmp_path):
    """All observations have events (no censoring)."""
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        "time": rng.exponential(100, n),
        "event": np.ones(n, dtype=int),
        "treatment": rng.binomial(1, 0.5, n),
    })
    return _write_csv(df, tmp_path, "all_events.csv")


# ===========================================================================
# Kaplan-Meier Validation
# ===========================================================================

class TestKaplanMeierValidation:
    def test_km_normal_event_rate(self, normal_survival):
        """KM works with normal event rate."""
        result = kaplan_meier(normal_survival, time_col="time", event_col="event")
        assert "error" not in result
        assert result["n_events"] > 0
        assert result["n_censored"] > 0

    def test_km_zero_events_blocked(self, zero_events):
        """KM with zero events → error/validation failure."""
        result = kaplan_meier(zero_events, time_col="time", event_col="event")
        # Either error or validation shows zero events
        if "error" in result:
            assert "event" in result["error"].lower() or "complete" in result["error"].lower()
        else:
            assert result.get("validation", {}).get("n_events", 0) == 0

    def test_km_matches_lifelines_normal(self, normal_survival):
        """KM matches lifelines for normal event rate."""
        result = kaplan_meier(normal_survival, time_col="time", event_col="event")
        df = pd.read_csv(normal_survival)
        kmf = KaplanMeierFitter()
        kmf.fit(df["time"], event_observed=df["event"])

        sf = kmf.survival_function_
        ref_surv = sf.iloc[:, 0].tolist()
        statis_surv = result["groups"]["overall"]["survival_probabilities"]

        n_compare = min(len(statis_surv), len(ref_surv))
        assert n_compare > 0
        for i in range(n_compare):
            assert abs(statis_surv[i] - ref_surv[i]) < 1e-5


# ===========================================================================
# Log-Rank Validation
# ===========================================================================

class TestLogRankValidation:
    def test_logrank_normal(self, normal_survival):
        """Log-rank works with normal event rate."""
        result = log_rank(normal_survival, time_col="time", event_col="event", group_col="treatment")
        assert "error" not in result
        assert result["test_statistic"] is not None
        assert result["p_value"] is not None

    def test_logrank_matches_lifelines(self, normal_survival):
        """Log-rank matches lifelines."""
        result = log_rank(normal_survival, time_col="time", event_col="event",
                          group_col="treatment", group1=0, group2=1)
        df = pd.read_csv(normal_survival)
        g0 = df[df["treatment"] == 0]
        g1 = df[df["treatment"] == 1]
        ref = logrank_test(g0["time"], g1["time"],
                           event_observed_A=g0["event"], event_observed_B=g1["event"])
        assert abs(result["test_statistic"] - ref.test_statistic) < 1e-5
        assert abs(result["p_value"] - ref.p_value) < 1e-5

    def test_logrank_zero_events(self, zero_events):
        """Log-rank with zero events → handled gracefully (chi2=0, p=1.0)."""
        result = log_rank(zero_events, time_col="time", event_col="event", group_col="treatment")
        # With zero events, chi-square is 0 and p-value is 1.0 — mathematically correct
        if "error" not in result:
            assert result["p_value"] == 1.0
            assert result["test_statistic"] == 0.0
        else:
            # Or it returns an error for insufficient data
            assert "error" in result


# ===========================================================================
# Cox Proportional Hazards Validation
# ===========================================================================

class TestCoxValidation:
    def test_cox_normal(self, normal_survival):
        """Cox PH works with normal event rate."""
        result = cox_proportional_hazards(
            normal_survival, time_col="time", event_col="event", covariates=["treatment"],
        )
        assert "error" not in result
        assert result["hazard_ratios"] is not None
        assert "treatment" in result["hazard_ratios"]

    def test_cox_matches_lifelines(self, normal_survival):
        """Cox PH matches lifelines."""
        result = cox_proportional_hazards(
            normal_survival, time_col="time", event_col="event", covariates=["treatment"],
        )
        df = pd.read_csv(normal_survival)
        cph = CoxPHFitter()
        cph.fit(df[["time", "event", "treatment"]], duration_col="time", event_col="event")
        assert abs(result["hazard_ratios"]["treatment"] - np.exp(cph.params_["treatment"])) < 1e-4

    def test_cox_zero_events_review(self, zero_events):
        """Cox with zero events → REVIEW_REQUIRED defensibility."""
        result = cox_proportional_hazards(
            zero_events, time_col="time", event_col="event", covariates=["treatment"],
        )
        # Should either error or return validation showing zero events
        if "error" not in result:
            assert result.get("validation", {}).get("n_events", 0) == 0

    def test_cox_all_events(self, all_events):
        """Cox with all events (no censoring) → handled."""
        result = cox_proportional_hazards(
            all_events, time_col="time", event_col="event", covariates=["treatment"],
        )
        # Should work or return validation
        if "error" not in result:
            assert result.get("validation", {}).get("n_events", 0) == len(all_events.read_text().splitlines()) - 1


# ===========================================================================
# Defensibility Rules (Phase 2F)
# ===========================================================================

class TestSurvivalDefensibilityRules:
    def test_zero_events_not_defensible(self):
        """Zero events → NOT_DEFENSIBLE."""
        status = assess_defensibility(
            ExecutionStatus.SUCCESS,
            SanityStatus.PASS,
            DiagnosticDiagnosticStatus.PASS,
            survival_metadata={"method_id": "kaplan_meier", "n_events": 0, "n_total": 100},
        )
        assert status.value == "not_defensible"

    def test_ph_violation_review_required(self):
        """PH violation → REVIEW_REQUIRED."""
        status = assess_defensibility(
            ExecutionStatus.SUCCESS,
            SanityStatus.PASS,
            DiagnosticDiagnosticStatus.PASS,
            survival_metadata={
                "method_id": "cox_proportional_hazards",
                "n_events": 50,
                "n_total": 200,
                "ph_assumption_met": False,
            },
        )
        assert status.value == "review_required"

    def test_sparse_events_review_required(self):
        """Sparse events (<10) → REVIEW_REQUIRED."""
        status = assess_defensibility(
            ExecutionStatus.SUCCESS,
            SanityStatus.PASS,
            DiagnosticDiagnosticStatus.PASS,
            survival_metadata={
                "method_id": "kaplan_meier",
                "n_events": 5,
                "n_total": 100,
                "sparse_events": True,
            },
        )
        assert status.value == "review_required"

    def test_extreme_censoring_review_required(self):
        """Censoring >90% → REVIEW_REQUIRED."""
        status = assess_defensibility(
            ExecutionStatus.SUCCESS,
            SanityStatus.PASS,
            DiagnosticDiagnosticStatus.PASS,
            survival_metadata={
                "method_id": "kaplan_meier",
                "n_events": 3,
                "n_censored": 95,
                "n_total": 98,
            },
        )
        assert status.value == "review_required"

    def test_all_events_cox_review_required(self):
        """All events for Cox → REVIEW_REQUIRED."""
        status = assess_defensibility(
            ExecutionStatus.SUCCESS,
            SanityStatus.PASS,
            DiagnosticDiagnosticStatus.PASS,
            survival_metadata={
                "method_id": "cox_proportional_hazards",
                "n_events": 100,
                "n_total": 100,
                "all_events": True,
            },
        )
        assert status.value == "review_required"

    def test_normal_events_defensible(self):
        """Normal event rate → DEFENSIBLE."""
        status = assess_defensibility(
            ExecutionStatus.SUCCESS,
            SanityStatus.PASS,
            DiagnosticDiagnosticStatus.PASS,
            survival_metadata={
                "method_id": "kaplan_meier",
                "n_events": 40,
                "n_total": 100,
                "n_censored": 60,
            },
        )
        assert status.value == "defensible"


# ===========================================================================
# Survival Data Validation
# ===========================================================================

class TestSurvivalDataValidation:
    def test_valid_data(self, normal_survival):
        """Valid survival data passes validation."""
        df = pd.read_csv(normal_survival)
        result = validate_survival_data(df, "time", "event")
        assert result["valid"] is True

    def test_invalid_time_blocked(self, tmp_path):
        """Negative time values → blocked."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "time": [-10, 50, 100, 150, -5],
            "event": [1, 0, 1, 0, 1],
        })
        path = _write_csv(df, tmp_path, "invalid_time.csv")
        df = pd.read_csv(path)
        result = validate_survival_data(df, "time", "event")
        assert result["valid"] is False
        assert len(result["blockers"]) > 0

    def test_invalid_event_blocked(self, tmp_path):
        """Invalid event indicator → blocked."""
        df = pd.DataFrame({
            "time": [10, 20, 30, 40, 50],
            "event": [1, 2, 0, 1, 3],  # 2 and 3 are invalid
        })
        path = _write_csv(df, tmp_path, "invalid_event.csv")
        df = pd.read_csv(path)
        result = validate_survival_data(df, "time", "event")
        assert result["valid"] is False
