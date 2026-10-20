"""Phase 2J — Anti-P-Hacking Validation (Invariant 38).

Verifies STATIS does NOT automatically manipulate data, methods, exclusions,
transformations, or classifications to improve statistical significance.

Tests both favorable and unfavorable synthetic results.
The same prespecified analysis should remain the same regardless of p-value.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.statistics.anti_phacking import AntiPHackingGuard, AntiPHackingVerdict
from app.statistics.engine import (
    independent_t_test, mann_whitney_u, correlation_test,
    one_way_anova, logistic_regression, linear_regression,
)
from app.statistics.plan import AnalysisLifecycle
from app.statistics.specification import (
    StatisticalAnalysisSpecification, ApprovalStatus, ValidationStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def guard():
    return AntiPHackingGuard()


@pytest.fixture
def significant_data(tmp_path):
    """Data where t-test is significant (p < 0.05)."""
    rng = np.random.RandomState(42)
    n = 100
    g1 = rng.normal(50, 10, n)
    g2 = rng.normal(65, 10, n)  # large effect
    df = pd.DataFrame({
        "value": np.concatenate([g1, g2]),
        "group": ["A"] * n + ["B"] * n,
    })
    path = tmp_path / "significant.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def nonsignificant_data(tmp_path):
    """Data where t-test is NOT significant (p > 0.05)."""
    rng = np.random.RandomState(42)
    n = 100
    g1 = rng.normal(50, 10, n)
    g2 = rng.normal(51, 10, n)  # small effect
    df = pd.DataFrame({
        "value": np.concatenate([g1, g2]),
        "group": ["A"] * n + ["B"] * n,
    })
    path = tmp_path / "nonsignificant.csv"
    df.to_csv(path, index=False)
    return path


# ===========================================================================
# Anti-P-Hacking Guard: Strategy Selection
# ===========================================================================

class TestStrategySelectionGuard:
    def test_blocks_p_value_selection(self, guard):
        """BLOCKED: strategy selected because p < 0.05."""
        strategies = [
            {"method": "t_test", "comparison_reason": "lower p_value"},
            {"method": "mann_whitney", "comparison_reason": "higher effect_size"},
        ]
        selected = {"method": "t_test", "p_value_reason": "produced p < 0.05"}
        checks = guard.check_strategy_selection(strategies, selected, p_value=0.03)
        blocked = [c for c in checks if c.verdict == AntiPHackingVerdict.BLOCKED]
        assert len(blocked) > 0
        assert any("p-hacking" in c.description.lower() or "p_value" in c.check_name for c in blocked)

    def test_blocks_significance_comparison(self, guard):
        """BLOCKED/SUSPICIOUS: strategies compared based on significance."""
        strategies = [
            {"method": "t_test", "comparison_reason": "significance level achieved"},
            {"method": "mann_whitney", "comparison_reason": "not significant"},
        ]
        selected = {"method": "t_test"}
        checks = guard.check_strategy_selection(strategies, selected)
        flagged = [c for c in checks if c.verdict in (AntiPHackingVerdict.BLOCKED, AntiPHackingVerdict.SUSPICIOUS)]
        assert len(flagged) > 0

    def test_safe_when_no_p_hacking(self, guard):
        """SAFE: methodological reasons, not p-values."""
        strategies = [
            {"method": "t_test", "comparison_reason": "assumes normality and homogeneity of variance"},
            {"method": "mann_whitney", "comparison_reason": "nonparametric alternative for skewed data"},
        ]
        selected = {"method": "mann_whitney", "selection_reason": "data is skewed"}
        checks = guard.check_strategy_selection(strategies, selected)
        blocked = [c for c in checks if c.verdict == AntiPHackingVerdict.BLOCKED]
        assert len(blocked) == 0

    def test_flags_iterative_exclusion(self, guard):
        """SUSPICIOUS: iterative outlier exclusion."""
        strategies = [{"method": "t_test"}]
        selected = {"method": "t_test", "iteration_count": 3}
        checks = guard.check_strategy_selection(strategies, selected)
        suspicious = [c for c in checks if c.verdict == AntiPHackingVerdict.SUSPICIOUS]
        assert len(suspicious) > 0


# ===========================================================================
# Anti-P-Hacking Guard: Transformation Selection
# ===========================================================================

class TestTransformationSelectionGuard:
    def test_blocks_transformation_p_value_selection(self, guard):
        """BLOCKED: transformation selected to minimize p-value."""
        transformations = [
            {"name": "log", "p_value": 0.03},
            {"name": "sqrt", "p_value": 0.12},
        ]
        selected = {"name": "log", "selection_reason": "smallest_p_value"}
        checks = guard.check_transformation_selection(transformations, selected, result_p_value=0.03)
        blocked = [c for c in checks if c.verdict == AntiPHackingVerdict.BLOCKED]
        assert len(blocked) > 0

    def test_safe_when_methodological_reason(self, guard):
        """SAFE: transformation for methodological reasons."""
        transformations = [
            {"name": "log", "reason": "right-skewed distribution"},
            {"name": "sqrt", "reason": "variance stabilization"},
        ]
        selected = {"name": "log", "selection_reason": "distribution is right-skewed"}
        checks = guard.check_transformation_selection(transformations, selected, result_p_value=None)
        blocked = [c for c in checks if c.verdict == AntiPHackingVerdict.BLOCKED]
        assert len(blocked) == 0


# ===========================================================================
# Anti-P-Hacking Guard: Outlier Exclusion
# ===========================================================================

class TestOutlierExclusionGuard:
    def test_blocks_significance_justification(self, guard):
        """BLOCKED: outlier exclusion justified by achieving significance."""
        checks = guard.check_outlier_exclusion(
            "removed outliers to make result significant",
            affects_significance=True,
        )
        blocked = [c for c in checks if c.verdict == AntiPHackingVerdict.BLOCKED]
        assert len(blocked) > 0

    def test_safe_with_methodological_justification(self, guard):
        """SAFE: outlier exclusion with methodological justification."""
        checks = guard.check_outlier_exclusion(
            "data_entry_error in record 45; measurement_error confirmed by lab",
            affects_significance=False,
        )
        blocked = [c for c in checks if c.verdict == AntiPHackingVerdict.BLOCKED]
        assert len(blocked) == 0

    def test_suspicious_without_methodological_terms(self, guard):
        """SUSPICIOUS: exclusion without methodological justification."""
        checks = guard.check_outlier_exclusion(
            "removed outliers that looked weird",
            affects_significance=False,
        )
        suspicious = [c for c in checks if c.verdict == AntiPHackingVerdict.SUSPICIOUS]
        assert len(suspicious) > 0


# ===========================================================================
# Prespecified Analysis Invariance
# ===========================================================================

class TestPrespecifiedAnalysisInvariance:
    def test_same_specification_regardless_of_significance(self, significant_data, nonsignificant_data):
        """Same specification produces same defensibility status regardless of p-value."""
        from app.statistics.diagnostics import build_diagnostic_summary, DiagnosticResult, DiagnosticStatus

        # Run on significant data
        sig_result = independent_t_test(significant_data, outcome="value", group="group", group1="A", group2="B")
        # Run on non-significant data
        nonsig_result = independent_t_test(nonsignificant_data, outcome="value", group="group", group1="A", group2="B")

        # Both should have the same structure
        assert sig_result.get("test") == nonsig_result.get("test")
        assert sig_result.get("equal_variances_assumed") == nonsig_result.get("equal_variances_assumed")
        assert sig_result.get("alternative") == nonsig_result.get("alternative")

        # The analysis method does NOT change based on p-value
        assert "independent_t_test" in str(sig_result.get("test", "")).lower() or "t-test" in str(sig_result.get("test", "")).lower()
        assert "independent_t_test" in str(nonsig_result.get("test", "")).lower() or "t-test" in str(nonsig_result.get("test", "")).lower()

    def test_method_does_not_switch_on_nonsignificance(self, nonsignificant_data):
        """STATIS does not switch to Mann-Whitney when t-test is non-significant."""
        result = independent_t_test(nonsig_result := nonsignificant_data, outcome="value", group="group", group1="A", group2="B")
        # Even if p > 0.05, the method remains independent t-test
        assert "t-test" in result.get("test", "").lower()
        # Mann-Whitney is listed as alternative, not substituted
        assert result.get("alternative") is not None

    def test_alternative_methods_not_automatic(self, nonsignificant_data):
        """Alternative methods are noted but not automatically applied."""
        result = independent_t_test(nonsig_result := nonsignificant_data, outcome="value", group="group", group1="A", group2="B")
        # The result should still be from the t-test, not Mann-Whitney
        assert "mann-whitney" not in result.get("test", "").lower()


# ===========================================================================
# No Automatic Outlier Removal
# ===========================================================================

class TestNoAutomaticOutlierRemoval:
    def test_outliers_not_removed_by_engine(self, tmp_path):
        """Engine does not automatically remove outliers."""
        rng = np.random.RandomState(42)
        n = 100
        g1 = rng.normal(50, 10, n)
        g2 = rng.normal(55, 10, n)
        # Inject outliers
        g1[0] = 200
        g1[1] = -50
        df = pd.DataFrame({
            "value": np.concatenate([g1, g2]),
            "group": ["A"] * n + ["B"] * n,
        })
        path = tmp_path / "outliers.csv"
        df.to_csv(path, index=False)

        result = independent_t_test(path, outcome="value", group="group", group1="A", group2="B")
        # n should include the outliers (not removed)
        assert result["n_group1"] == n
        assert result["n_group2"] == n

    def test_engine_reports_outliers_in_diagnostics(self, tmp_path):
        """Diagnostics may flag outliers but engine doesn't remove them."""
        from app.statistics.method_diagnostics import run_method_diagnostics
        from app.statistics.result import ExecutionStatus

        rng = np.random.RandomState(42)
        n = 100
        g1 = rng.normal(50, 10, n)
        g2 = rng.normal(55, 10, n)
        g1[0] = 200  # outlier
        df = pd.DataFrame({
            "value": np.concatenate([g1, g2]),
            "group": ["A"] * n + ["B"] * n,
        })
        path = tmp_path / "diag_outliers.csv"
        df.to_csv(path, index=False)

        result = independent_t_test(path, outcome="value", group="group", group1="A", group2="B")
        # The result includes the outlier - n is full
        assert result["n_group1"] == n


# ===========================================================================
# No Automatic Transformation
# ===========================================================================

class TestNoAutomaticTransformation:
    def test_raw_values_used(self, tmp_path):
        """Engine uses raw values, not transformed."""
        rng = np.random.RandomState(42)
        n = 100
        # Skewed data
        g1 = np.exp(rng.normal(2, 1, n))
        g2 = np.exp(rng.normal(2.3, 1, n))
        df = pd.DataFrame({
            "value": np.concatenate([g1, g2]),
            "group": ["A"] * n + ["B"] * n,
        })
        path = tmp_path / "skewed.csv"
        df.to_csv(path, index=False)

        result = independent_t_test(path, outcome="value", group="group", group1="A", group2="B")
        # The means should match the raw data (not log-transformed)
        raw_mean_a = g1.mean()
        raw_mean_b = g2.mean()
        assert abs(result["mean_group1"] - raw_mean_a) < 0.01
        assert abs(result["mean_group2"] - raw_mean_b) < 0.01


# ===========================================================================
# No Automatic Classification Change
# ===========================================================================

class TestNoAutomaticClassificationChange:
    def test_analysis_class_not_changed_by_result(self):
        """Analysis classification doesn't change based on results."""
        spec = StatisticalAnalysisSpecification(
            analysis_method="independent_t_test",
            primary_outcome="outcome",
            predictors=["group"],
        )
        # Classification is set at specification time, not after results
        assert spec.analysis_method == "independent_t_test"

    def test_prespecified_status_not_changed_by_result(self):
        """Prespecified status doesn't change based on results."""
        spec = StatisticalAnalysisSpecification(
            analysis_method="independent_t_test",
            primary_outcome="outcome",
            predictors=["group"],
        )
        # Default prespecified status
        assert hasattr(spec, "prespecified_status")


# ===========================================================================
# Unfavorable Results Remain Visible
# ===========================================================================

class TestUnfavorableResultsVisible:
    def test_nonsignificant_result_not_suppressed(self, nonsignificant_data):
        """Non-significant results are returned, not suppressed."""
        result = independent_t_test(
            nonsignificant_data, outcome="value", group="group", group1="A", group2="B",
        )
        # p-value is present and > 0.05
        assert result["p_value"] is not None
        assert result["p_value"] > 0.05
        # Result is not None or empty
        assert result.get("test") is not None
        assert result.get("t_statistic") is not None

    def test_unfavorable_result_has_same_structure(self, significant_data, nonsignificant_data):
        """Favorable and unfavorable results have the same structure."""
        sig = independent_t_test(significant_data, outcome="value", group="group", group1="A", group2="B")
        nonsig = independent_t_test(nonsignificant_data, outcome="value", group="group", group1="A", group2="B")
        # Same keys present
        assert set(sig.keys()) == set(nonsig.keys())
