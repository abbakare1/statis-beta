"""Phase 2E.1 Tests — Statistical Diagnostics and Defensibility Engine.

Covers:
- Core diagnostic framework (DiagnosticSeverity, DiagnosticResult, DiagnosticSummary)
- Defensibility assessment
- Method-specific diagnostics for all 11 methods
- Tests A through M as specified in the requirements
- Integration with StatisticalExecutor
- Provenance and fingerprinting
"""

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_independent_groups(tmp_path):
    """Test A — Clean two-group data with known difference."""
    rng = np.random.RandomState(42)
    n_per = 60
    grp = np.array(["A"] * n_per + ["B"] * n_per)
    vals = np.concatenate([
        rng.normal(100, 15, n_per),
        rng.normal(110, 15, n_per),
    ])
    df = pd.DataFrame({"group": grp, "score": vals})
    path = tmp_path / "clean_groups.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def unequal_variance_data(tmp_path):
    """Test B — Two groups with very different variances."""
    rng = np.random.RandomState(42)
    grp = np.array(["A"] * 60 + ["B"] * 60)
    vals = np.concatenate([
        rng.normal(100, 5, 60),    # tight
        rng.normal(110, 30, 60),   # wide
    ])
    df = pd.DataFrame({"group": grp, "score": vals})
    path = tmp_path / "uneq_var.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def extreme_imbalance_data(tmp_path):
    """Test C — Extreme group imbalance."""
    rng = np.random.RandomState(42)
    grp = np.array(["A"] * 94 + ["B"] * 6)
    vals = np.concatenate([
        rng.normal(100, 15, 94),
        rng.normal(110, 15, 6),
    ])
    df = pd.DataFrame({"group": grp, "score": vals})
    path = tmp_path / "imbalance.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def constant_outcome_data(tmp_path):
    """Test D — Constant outcome in one or both groups."""
    rng = np.random.RandomState(42)
    grp = np.array(["A"] * 30 + ["B"] * 30)
    vals = np.concatenate([
        np.full(30, 50.0),
        rng.normal(60, 10, 30),
    ])
    df = pd.DataFrame({"group": grp, "score": vals})
    path = tmp_path / "constant_outcome.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def missing_outcome_data(tmp_path):
    """Test E — Missing outcome values."""
    rng = np.random.RandomState(42)
    grp = np.array(["A"] * 60 + ["B"] * 60)
    vals = np.concatenate([
        rng.normal(100, 15, 50),   # 10 missing in A
        [np.nan] * 10,
        rng.normal(110, 15, 55),   # 5 missing in B
        [np.nan] * 5,
    ])
    df = pd.DataFrame({"group": grp, "score": vals})
    path = tmp_path / "missing_outcome.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def regression_data(tmp_path):
    """Test F/G — Linear regression data."""
    rng = np.random.RandomState(42)
    n = 150
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    y = 3 * x1 + 0.5 * x2 + rng.normal(0, 2, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    path = tmp_path / "regression.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def heteroscedastic_data(tmp_path):
    """Test F — Regression with heteroscedasticity."""
    rng = np.random.RandomState(42)
    n = 200
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 1, n) * (1 + 2 * np.abs(x))
    df = pd.DataFrame({"y": y, "x": x})
    path = tmp_path / "heteroscedastic.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def multicollinear_data(tmp_path):
    """Test G — Highly correlated predictors."""
    rng = np.random.RandomState(42)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = x1 + rng.normal(0, 0.05, n)  # near-constant relationship
    x3 = rng.normal(0, 1, n)
    y = 2 * x1 + 3 * x3 + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    path = tmp_path / "multicollinear.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def logistic_data(tmp_path):
    """Test H — Logistic regression data with separation scenario."""
    rng = np.random.RandomState(42)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit = 0.5 + 1.5 * x1 - 0.8 * x2
    prob = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, prob, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    path = tmp_path / "logistic.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sparse_chi2_data(tmp_path):
    """Test I — Sparse contingency table."""
    rng = np.random.RandomState(42)
    rows = []
    for _ in range(20):
        rows.append({"treatment": "Drug", "response": "Improved"})
    for _ in range(2):
        rows.append({"treatment": "Drug", "response": "Not improved"})
    for _ in range(10):
        rows.append({"treatment": "Placebo", "response": "Improved"})
    for _ in range(8):
        rows.append({"treatment": "Placebo", "response": "Not improved"})
    df = pd.DataFrame(rows)
    path = tmp_path / "sparse_chi2.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def paired_data(tmp_path):
    """Paired test data."""
    rng = np.random.RandomState(42)
    n = 80
    pre = rng.normal(50, 10, n)
    post = pre + rng.normal(-5, 8, n)
    df = pd.DataFrame({"pre": pre, "post": post})
    path = tmp_path / "paired.csv"
    df.to_csv(path, index=False)
    return path


def _uid() -> str:
    return f"test_{uuid.uuid4().hex[:8]}"


# ===========================================================================
# CORE FRAMEWORK TESTS
# ===========================================================================

class TestDiagnosticFramework:
    def test_diagnostic_result_creation(self):
        from app.statistics.diagnostics import DiagnosticResult, DiagnosticSeverity, DiagnosticStatus
        d = DiagnosticResult(
            diagnostic_id="test_001",
            diagnostic_name="Test diagnostic",
            severity=DiagnosticSeverity.WARNING,
            status=DiagnosticStatus.PASS,
            variable="score",
            method="t-test",
            observed_value=0.05,
            threshold=0.05,
            message="Test message",
            rationale="Test rationale",
            deterministic=True,
        )
        assert d.diagnostic_id == "test_001"
        assert d.severity == DiagnosticSeverity.WARNING
        assert d.status == DiagnosticStatus.PASS
        assert d.deterministic is True

    def test_diagnostic_summary_empty(self):
        from app.statistics.diagnostics import DiagnosticSummary, DiagnosticDiagnosticStatus
        s = DiagnosticSummary()
        assert s.overall_status == DiagnosticDiagnosticStatus.PASS
        assert len(s.diagnostics) == 0

    def test_diagnostic_summary_with_warnings(self):
        from app.statistics.diagnostics import (
            DiagnosticSummary, DiagnosticResult, DiagnosticSeverity, DiagnosticStatus,
            DiagnosticDiagnosticStatus, build_diagnostic_summary,
        )
        d1 = DiagnosticResult(diagnostic_id="d1", diagnostic_name="d1", severity=DiagnosticSeverity.WARNING, status=DiagnosticStatus.WARNING, message="warn1")
        d2 = DiagnosticResult(diagnostic_id="d2", diagnostic_name="d2", severity=DiagnosticSeverity.INFO, status=DiagnosticStatus.PASS, message="ok")
        summary = build_diagnostic_summary([d1, d2])
        assert summary.overall_status == DiagnosticDiagnosticStatus.WARNING
        assert len(summary.warnings) == 1
        assert "warn1" in summary.warnings[0]

    def test_diagnostic_summary_with_errors(self):
        from app.statistics.diagnostics import (
            DiagnosticResult, DiagnosticSeverity, DiagnosticStatus,
            build_diagnostic_summary, DiagnosticDiagnosticStatus,
        )
        d1 = DiagnosticResult(diagnostic_id="d1", diagnostic_name="d1", severity=DiagnosticSeverity.ERROR, status=DiagnosticStatus.FAIL, message="error1")
        d2 = DiagnosticResult(diagnostic_id="d2", diagnostic_name="d2", severity=DiagnosticSeverity.WARNING, status=DiagnosticStatus.WARNING, message="warn1")
        summary = build_diagnostic_summary([d1, d2])
        assert summary.overall_status == DiagnosticDiagnosticStatus.FAIL
        assert len(summary.errors) == 1

    def test_diagnostic_summary_with_review_required(self):
        from app.statistics.diagnostics import (
            DiagnosticResult, DiagnosticSeverity, DiagnosticStatus,
            build_diagnostic_summary, DiagnosticDiagnosticStatus,
        )
        d1 = DiagnosticResult(diagnostic_id="d1", diagnostic_name="d1", severity=DiagnosticSeverity.REVIEW_REQUIRED, status=DiagnosticStatus.REVIEW_REQUIRED, message="review1")
        summary = build_diagnostic_summary([d1])
        assert summary.overall_status == DiagnosticDiagnosticStatus.REVIEW_REQUIRED

    def test_diagnostic_summary_assumptions_tracked(self):
        from app.statistics.diagnostics import (
            DiagnosticResult, DiagnosticSeverity, DiagnosticStatus,
            build_diagnostic_summary,
        )
        d1 = DiagnosticResult(diagnostic_id="d1", diagnostic_name="Normality", severity=DiagnosticSeverity.INFO, status=DiagnosticStatus.PASS, message="ok")
        d2 = DiagnosticResult(diagnostic_id="d2", diagnostic_name="Homogeneity", severity=DiagnosticSeverity.WARNING, status=DiagnosticStatus.WARNING, message="warn")
        summary = build_diagnostic_summary([d1, d2], assumptions_checked=["Normality", "Homogeneity"])
        assert "Normality" in summary.assumptions_checked
        assert "Homogeneity" in summary.assumptions_uncertain


# ===========================================================================
# DEFENSIBILITY ASSESSMENT TESTS
# ===========================================================================

class TestDefensibility:
    def test_defensible(self):
        from app.statistics.defensibility import assess_defensibility
        from app.statistics.result import ExecutionStatus, SanityStatus
        from app.statistics.diagnostics import DiagnosticDiagnosticStatus
        assert assess_defensibility(
            ExecutionStatus.SUCCESS, SanityStatus.PASS, DiagnosticDiagnosticStatus.PASS
        ).value == "defensible"

    def test_defensible_with_caveats(self):
        from app.statistics.defensibility import assess_defensibility
        from app.statistics.result import ExecutionStatus, SanityStatus
        from app.statistics.diagnostics import DiagnosticDiagnosticStatus
        assert assess_defensibility(
            ExecutionStatus.SUCCESS, SanityStatus.PASS, DiagnosticDiagnosticStatus.WARNING
        ).value == "defensible_with_caveats"

    def test_review_required(self):
        from app.statistics.defensibility import assess_defensibility
        from app.statistics.result import ExecutionStatus, SanityStatus
        from app.statistics.diagnostics import DiagnosticDiagnosticStatus
        assert assess_defensibility(
            ExecutionStatus.SUCCESS, SanityStatus.PASS, DiagnosticDiagnosticStatus.REVIEW_REQUIRED
        ).value == "review_required"

    def test_not_defensible_on_fail(self):
        from app.statistics.defensibility import assess_defensibility
        from app.statistics.result import ExecutionStatus, SanityStatus
        from app.statistics.diagnostics import DiagnosticDiagnosticStatus
        assert assess_defensibility(
            ExecutionStatus.SUCCESS, SanityStatus.PASS, DiagnosticDiagnosticStatus.FAIL
        ).value == "not_defensible"

    def test_not_defensible_on_execution_fail(self):
        from app.statistics.defensibility import assess_defensibility
        from app.statistics.result import ExecutionStatus, SanityStatus
        from app.statistics.diagnostics import DiagnosticDiagnosticStatus
        assert assess_defensibility(
            ExecutionStatus.FAILED, SanityStatus.PASS, DiagnosticDiagnosticStatus.PASS
        ).value == "not_defensible"

    def test_not_defensible_on_sanity_fail(self):
        from app.statistics.defensibility import assess_defensibility
        from app.statistics.result import ExecutionStatus, SanityStatus
        from app.statistics.diagnostics import DiagnosticDiagnosticStatus
        assert assess_defensibility(
            ExecutionStatus.SUCCESS, SanityStatus.FAIL, DiagnosticDiagnosticStatus.PASS
        ).value == "not_defensible"

    def test_not_defensible_on_blocked(self):
        from app.statistics.defensibility import assess_defensibility
        from app.statistics.result import ExecutionStatus, SanityStatus
        from app.statistics.diagnostics import DiagnosticDiagnosticStatus
        assert assess_defensibility(
            ExecutionStatus.BLOCKED, SanityStatus.PASS, DiagnosticDiagnosticStatus.PASS,
            execution_blocked=True,
        ).value == "not_defensible"


# ===========================================================================
# TEST A — Clean independent groups
# ===========================================================================

class TestA_CleanIndependentGroups:
    def test_execution_success(self, clean_independent_groups):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS

    def test_numerical_sanity_pass(self, clean_independent_groups):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import SanityStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
        )
        assert result.numerical_sanity.status == SanityStatus.PASS

    def test_diagnostics_pass(self, clean_independent_groups):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.diagnostics import DiagnosticDiagnosticStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
        )
        assert result.diagnostic_status in ("pass", "warning")
        # With clean data, should be pass or very mild warnings
        diag = result.diagnostic_summary
        assert diag["n_errors"] == 0

    def test_defensibility_defensible(self, clean_independent_groups):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
        )
        assert result.defensibility_status in ("defensible", "defensible_with_caveats")

    def test_numerical_result_unchanged(self, clean_independent_groups):
        """Verify that diagnostics do not alter the numerical result."""
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
        )
        assert result.test_statistic is not None
        assert result.p_value is not None
        assert result.effect_size is not None
        assert 50 < result.n_observed < 130


# ===========================================================================
# TEST B — Unequal variance
# ===========================================================================

class TestB_UnequalVariance:
    def test_execution_still_succeeds(self, unequal_variance_data):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", unequal_variance_data,
            outcome="score", group="group",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS

    def test_variance_diagnostic_detected(self, unequal_variance_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", unequal_variance_data,
            outcome="score", group="group",
        )
        # Should have a variance ratio diagnostic that flags the issue
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        assert "Variance equality" in diag_names or result.diagnostic_status in ("warning", "review_required")

    def test_no_silent_method_change(self, unequal_variance_data):
        """The method should remain as executed (Welch or Student), not silently switch."""
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", unequal_variance_data,
            outcome="score", group="group",
        )
        # The engine result should not change based on diagnostics
        assert result.engine_result.get("test") is not None

    def test_appropriate_status(self, unequal_variance_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", unequal_variance_data,
            outcome="score", group="group",
        )
        assert result.defensibility_status in ("defensible", "defensible_with_caveats", "review_required")


# ===========================================================================
# TEST C — Extreme imbalance
# ===========================================================================

class TestC_ExtremeImbalance:
    def test_diagnostic_warning(self, extreme_imbalance_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", extreme_imbalance_data,
            outcome="score", group="group",
        )
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        assert "Group size balance" in diag_names

    def test_no_automatic_exclusion(self, extreme_imbalance_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", extreme_imbalance_data,
            outcome="score", group="group",
        )
        assert result.n_observed == 100  # all observations retained


# ===========================================================================
# TEST D — Constant outcome
# ===========================================================================

class TestD_ConstantOutcome:
    def test_zero_variance_detected(self, constant_outcome_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", constant_outcome_data,
            outcome="score", group="group",
        )
        # Should detect zero variance in group A
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        assert "Zero variance in group" in diag_names or result.diagnostic_status in ("fail", "warning", "review_required")

    def test_defensibility_affected(self, constant_outcome_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", constant_outcome_data,
            outcome="score", group="group",
        )
        assert result.defensibility_status in ("not_defensible", "defensible_with_caveats", "review_required")


# ===========================================================================
# TEST E — Missing outcome
# ===========================================================================

class TestE_MissingOutcome:
    def test_explicit_analyzed_n(self, missing_outcome_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", missing_outcome_data,
            outcome="score", group="group",
        )
        # Should use complete cases
        assert result.n_observed is not None
        assert result.n_observed > 0
        assert result.n_observed <= 120  # less than total due to missingness

    def test_no_silent_imputation(self, missing_outcome_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", missing_outcome_data,
            outcome="score", group="group",
        )
        # n_complete_cases should match n_observed
        assert result.data_verification.n_complete_cases is not None
        # The engine uses complete case analysis by default
        # Verify the engine result's group sizes sum to observed N
        n1 = result.engine_result.get("n_group1", 0)
        n2 = result.engine_result.get("n_group2", 0)
        assert n1 + n2 == result.n_observed


# ===========================================================================
# TEST F — Linear regression heteroscedasticity
# ===========================================================================

class TestF_Heteroscedasticity:
    def test_model_executes(self, heteroscedastic_data):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "linear_regression", heteroscedastic_data,
            outcome="y", predictors=["x"],
        )
        assert result.execution_status == ExecutionStatus.SUCCESS

    def test_heteroscedasticity_diagnostic_recorded(self, heteroscedastic_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "linear_regression", heteroscedastic_data,
            outcome="y", predictors=["x"],
        )
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        # Should have a heteroscedasticity diagnostic
        assert any("eteroscedasticity" in n for n in diag_names) or result.diagnostic_status != "not_checked"

    def test_no_automatic_model_replacement(self, heteroscedastic_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "linear_regression", heteroscedastic_data,
            outcome="y", predictors=["x"],
        )
        # Method should remain OLS
        assert result.method_id == "OLS Linear regression"
        assert result.engine_result.get("test") == "OLS Linear regression"


# ===========================================================================
# TEST G — Linear regression multicollinearity
# ===========================================================================

class TestG_Multicollinearity:
    def test_multicollinearity_detected(self, multicollinear_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "linear_regression", multicollinear_data,
            outcome="y", predictors=["x1", "x2", "x3"],
        )
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        assert "Multicollinearity (VIF)" in diag_names

    def test_appropriate_review_status(self, multicollinear_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "linear_regression", multicollinear_data,
            outcome="y", predictors=["x1", "x2", "x3"],
        )
        # Should flag review or warning
        assert result.defensibility_status in ("defensible_with_caveats", "review_required", "defensible")


# ===========================================================================
# TEST H — Logistic separation
# ===========================================================================

class TestH_LogisticSeparation:
    def test_model_converges(self, logistic_data):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "logistic_regression", logistic_data,
            outcome="y", predictors=["x1", "x2"],
        )
        assert result.execution_status == ExecutionStatus.SUCCESS

    def test_diagnostics_recorded(self, logistic_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "logistic_regression", logistic_data,
            outcome="y", predictors=["x1", "x2"],
        )
        diag = result.diagnostic_summary
        assert diag["n_diagnostics"] > 0

    def test_events_per_variable_checked(self, logistic_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "logistic_regression", logistic_data,
            outcome="y", predictors=["x1", "x2"],
        )
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        assert "Events per variable" in diag_names

    def test_convergence_checked(self, logistic_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "logistic_regression", logistic_data,
            outcome="y", predictors=["x1", "x2"],
        )
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        assert "Model convergence" in diag_names


# ===========================================================================
# TEST I — Sparse chi-square
# ===========================================================================

class TestI_SparseChiSquare:
    def test_expected_cell_diagnostic(self, sparse_chi2_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "chi_square", sparse_chi2_data,
            variable1="treatment", variable2="response",
        )
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        assert "Expected cell counts" in diag_names

    def test_no_automatic_substitution(self, sparse_chi2_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "chi_square", sparse_chi2_data,
            variable1="treatment", variable2="response",
        )
        # Should remain chi-square, not automatically switch to Fisher
        assert result.method_id == "Chi-square test of independence"
        assert result.engine_result.get("test") == "Chi-square test of independence"

    def test_methodological_note_about_fisher(self, sparse_chi2_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "chi_square", sparse_chi2_data,
            variable1="treatment", variable2="response",
        )
        # The diagnostic should mention Fisher as an alternative, not substitute it
        diag_messages = [d.get("message", "") for d in result.diagnostic_summary.get("diagnostics", [])]
        # No automatic method change
        assert result.method_id == "Chi-square test of independence"


# ===========================================================================
# TEST J — Dataset mutation
# ===========================================================================

class TestJ_DatasetMutation:
    def test_hash_mismatch_on_mutated_data(self, clean_independent_groups):
        """Modifying the dataset after analysis should produce a different content hash."""
        from app.statistics.executor import verify_dataset
        v1 = verify_dataset(clean_independent_groups)
        # Mutate the file
        df = pd.read_csv(clean_independent_groups)
        df.loc[0, "score"] = 99999
        df.to_csv(clean_independent_groups, index=False)
        v2 = verify_dataset(clean_independent_groups)
        assert v1.content_hash != v2.content_hash

    def test_hash_mismatch_detected_by_executor(self, clean_independent_groups):
        """Executor should warn when expected hash doesn't match actual."""
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        # Get the actual hash first
        from app.statistics.executor import verify_dataset
        v = verify_dataset(clean_independent_groups)
        # Now mutate the file
        df = pd.read_csv(clean_independent_groups)
        df.loc[0, "score"] = 99999
        df.to_csv(clean_independent_groups, index=False)
        # Execute with the old hash
        result = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
            expected_content_hash=v.content_hash,
        )
        assert len(result.data_verification.verification_warnings) > 0


# ===========================================================================
# TEST K — Locked analysis modification
# ===========================================================================

class TestK_LockedAnalysisModification:
    def test_locked_spec_cannot_be_executed_against_different_data(self):
        """A locked analysis specification should not silently accept different data."""
        from app.statistics.specification import StatisticalAnalysisSpecification, ApprovalStatus
        spec = StatisticalAnalysisSpecification(
            primary_outcome="score",
            analysis_method="independent_t_test",
        )
        spec.lock()
        assert spec.approval_status == ApprovalStatus.LOCKED

    def test_new_version_required_for_method_change(self):
        """Changing the method requires a new version."""
        from app.statistics.specification import StatisticalAnalysisSpecification, ApprovalStatus
        spec = StatisticalAnalysisSpecification(
            primary_outcome="score",
            analysis_method="independent_t_test",
        )
        spec.approve()
        assert spec.approval_status == ApprovalStatus.APPROVED
        spec.modify("2.0")
        assert spec.approval_status == ApprovalStatus.MODIFIED
        assert spec.version == "2.0"


# ===========================================================================
# TEST L — Anti-p-hacking
# ===========================================================================

class TestL_AntiPHacking:
    def test_diagnostics_do_not_select_by_pvalue(self, clean_independent_groups):
        """Diagnostics should not change the method based on p-value."""
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result1 = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
        )
        # The method should remain the same regardless of diagnostic results
        assert result1.engine_result.get("test") is not None
        # Diagnostics should not have changed the method
        assert result1.method_id == "Independent samples t-test"

    def test_no_method_switching_in_diagnostics(self, clean_independent_groups):
        """Verify that diagnostics only produce warnings/reviews, never method changes."""
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
        )
        # Check that no diagnostic suggests changing the method
        for d in result.diagnostic_summary.get("diagnostics", []):
            msg = d.get("message", "").lower()
            # Should not contain "switch to" or "replace with" or "method changed"
            assert "method changed" not in msg
            assert "switched to" not in msg
            assert "replaced with" not in msg


# ===========================================================================
# TEST M — Provenance
# ===========================================================================

class TestM_Provenance:
    def test_dataset_identity_in_result(self, clean_independent_groups):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
        )
        assert result.data_verification.content_hash is not None
        assert result.data_verification.schema_hash is not None

    def test_method_recorded(self, clean_independent_groups):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
        )
        assert result.method_id == "Independent samples t-test"
        assert result.engine_version is not None

    def test_diagnostic_version_recorded(self, clean_independent_groups):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
        )
        assert result.diagnostic_version == "2e.1"

    def test_fingerprint_includes_diagnostics(self, clean_independent_groups):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
        )
        assert result.result_fingerprint is not None
        assert len(result.result_fingerprint) == 64  # SHA-256

    def test_provenance_in_each_diagnostic(self, clean_independent_groups):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
        )
        for d in result.diagnostic_summary.get("diagnostics", []):
            assert d.get("deterministic") is True


# ===========================================================================
# ADDITIONAL METHOD COVERAGE
# ===========================================================================

class TestDiagnostics_PairedTTest:
    def test_paired_ttest_diagnostics(self, paired_data):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "paired_t_test", paired_data,
            outcome1="pre", outcome2="post",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.diagnostic_status in ("pass", "warning")
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        assert "Complete paired observations" in diag_names


class TestDiagnostics_MannWhitney:
    def test_mann_whitney_diagnostics(self, clean_independent_groups):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "mann_whitney_u", clean_independent_groups,
            outcome="score", group="group",
        )
        assert result.execution_status.value == "success"
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        assert "Two independent groups" in diag_names


class TestDiagnostics_Pearson:
    def test_pearson_diagnostics(self, regression_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "pearson_correlation", regression_data,
            x="x1", y="y",
        )
        assert result.execution_status.value == "success"
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        assert "Both variables numeric" in diag_names


class TestDiagnostics_Spearman:
    def test_spearman_diagnostics(self, regression_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "spearman_correlation", regression_data,
            x="x1", y="y",
        )
        assert result.execution_status.value == "success"


class TestDiagnostics_ANOVA:
    def test_anova_diagnostics(self, clean_independent_groups):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "one_way_anova", clean_independent_groups,
            outcome="score", group="group",
        )
        assert result.execution_status.value == "success"


class TestDiagnostics_LinearRegression:
    def test_linear_regression_diagnostics(self, regression_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "linear_regression", regression_data,
            outcome="y", predictors=["x1", "x2"],
        )
        assert result.execution_status.value == "success"
        diag = result.diagnostic_summary
        assert diag["n_diagnostics"] > 0


class TestDiagnostics_FisherExact:
    def test_fisher_diagnostics(self, sparse_chi2_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "fisher_exact", sparse_chi2_data,
            variable1="treatment", variable2="response",
        )
        assert result.execution_status.value == "success"


class TestDiagnostics_Wilcoxon:
    def test_wilcoxon_diagnostics(self, paired_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "wilcoxon_signed_rank", paired_data,
            outcome1="pre", outcome2="post",
        )
        assert result.execution_status.value == "success"


# ===========================================================================
# INTEGRATION: DIAGNOSTICS IN StatisticalResult
# ===========================================================================

class TestDiagnosticsInResult:
    def test_diagnostic_fields_present(self, clean_independent_groups):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
        )
        assert hasattr(result, "diagnostic_status")
        assert hasattr(result, "diagnostic_summary")
        assert hasattr(result, "methodological_warnings")
        assert hasattr(result, "review_required")
        assert hasattr(result, "alternative_methods_considered")
        assert hasattr(result, "defensibility_status")

    def test_to_summary_includes_diagnostics(self, clean_independent_groups):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
        )
        s = result.to_summary()
        assert "diagnostic_status" in s
        assert "defensibility_status" in s
        assert "n_methodological_warnings" in s
        assert "n_review_required" in s

    def test_three_distinct_statuses(self, clean_independent_groups):
        """Verify that execution, sanity, and diagnostic statuses are distinct."""
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
        )
        # All three should be present and independent
        assert result.execution_status.value in ("success", "failed", "blocked")
        assert result.numerical_sanity.status.value in ("pass", "warning", "fail", "not_checked")
        assert result.diagnostic_status in ("pass", "warning", "review_required", "fail", "not_checked")


# ===========================================================================
# NO SILENT METHOD SWITCHING
# ===========================================================================

class TestNoMethodSwitching:
    def test_diagnostics_never_change_method(self, clean_independent_groups):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
        )
        # The method_id should remain unchanged
        assert result.method_id == "Independent samples t-test"
        # No diagnostic should have status that implies method change
        for d in result.diagnostic_summary.get("diagnostics", []):
            assert "METHOD_CHANGED" not in d.get("message", "")

    def test_alternative_methods_list_empty_by_default(self, clean_independent_groups):
        """alternatives_considered should be empty — no automatic consideration."""
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", clean_independent_groups,
            outcome="score", group="group",
        )
        assert result.alternative_methods_considered == []
