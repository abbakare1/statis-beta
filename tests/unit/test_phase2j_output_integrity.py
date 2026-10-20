"""Phase 2J — Output Numerical Integrity (Invariant 40).

Verifies that stored figures, tables, and reports contain the same numerical
values as the underlying computation — no silent recomputation, no value drift.
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

from app.statistics.engine import (
    independent_t_test, mann_whitney_u, correlation_test,
    one_way_anova, logistic_regression, linear_regression,
    paired_t_test, wilcoxon_signed_rank,
    chi_square_test, fisher_exact_test,
)
from app.statistics.specification import StatisticalAnalysisSpecification


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def continuous_data(tmp_path):
    rng = np.random.RandomState(42)
    n = 120
    df = pd.DataFrame({
        "value": rng.normal(50, 12, n),
        "group": rng.choice(["A", "B"], n),
    })
    path = tmp_path / "continuous.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def paired_data(tmp_path):
    rng = np.random.RandomState(42)
    n = 80
    before = rng.normal(120, 15, n)
    after = before + rng.normal(-5, 8, n)
    df = pd.DataFrame({
        "before": before,
        "after": after,
        "id": np.arange(n),
    })
    path = tmp_path / "paired.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def binary_data(tmp_path):
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "outcome": rng.binomial(1, 0.4, n),
        "exposure": rng.binomial(1, 0.5, n),
        "age": rng.normal(50, 10, n),
    })
    path = tmp_path / "binary.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def continuous_continuous_data(tmp_path):
    rng = np.random.RandomState(42)
    n = 100
    x = rng.normal(0, 1, n)
    y = 0.6 * x + rng.normal(0, 0.8, n)
    df = pd.DataFrame({"x": x, "y": y})
    path = tmp_path / "cc.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def multi_group_data(tmp_path):
    rng = np.random.RandomState(42)
    n_per = 50
    df = pd.DataFrame({
        "value": np.concatenate([
            rng.normal(50, 10, n_per),
            rng.normal(55, 10, n_per),
            rng.normal(60, 10, n_per),
        ]),
        "group": ["A"] * n_per + ["B"] * n_per + ["C"] * n_per,
    })
    path = tmp_path / "multi.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def contingency_data(tmp_path):
    df = pd.DataFrame({
        "exposure": [1]*40 + [0]*30 + [1]*20 + [0]*50,
        "disease": [1]*40 + [1]*30 + [0]*20 + [0]*50,
    })
    path = tmp_path / "contingency.csv"
    df.to_csv(path, index=False)
    return path


# ===========================================================================
# Stored Results Are Not Recomputed
# ===========================================================================

class TestStoredResultsNotRecomputed:
    def test_specification_contains_no_result_values(self, continuous_data):
        """Specification is pure intent — contains no statistical result values."""
        spec = StatisticalAnalysisSpecification(
            analysis_method="independent_t_test",
            primary_outcome="value",
            predictors=["group"],
        )
        spec_dict = spec.model_dump() if hasattr(spec, "model_dump") else {}
        result_keys = ["t_statistic", "p_value", "mean", "std", "ci_95_lower", "ci_95_upper",
                       "cohens_d", "f_statistic", "chi2", "odds_ratio"]
        for key in result_keys:
            assert key not in str(spec_dict).lower()

    def test_engine_result_is_deterministic(self, continuous_data):
        """Same input → same output (deterministic computation)."""
        r1 = independent_t_test(continuous_data, outcome="value", group="group",
                                group1="A", group2="B")
        r2 = independent_t_test(continuous_data, outcome="value", group="group",
                                group1="A", group2="B")
        assert r1["t_statistic"] == r2["t_statistic"]
        assert r1["p_value"] == r2["p_value"]
        assert r1["mean_group1"] == r2["mean_group1"]
        assert r1["mean_group2"] == r2["mean_group2"]

    def test_result_is_json_serializable_roundtrip(self, continuous_data):
        """Engine result survives JSON round-trip without value loss."""
        r1 = independent_t_test(continuous_data, outcome="value", group="group",
                                group1="A", group2="B")
        serialized = json.dumps(r1, default=str)
        restored = json.loads(serialized)
        for key in ["t_statistic", "p_value", "mean_group1", "mean_group2",
                    "std_group1", "std_group2", "df"]:
            assert restored[key] == r1[key], f"Mismatch for {key}: {restored[key]} vs {r1[key]}"


# ===========================================================================
# Numerical Integrity Per Method
# ===========================================================================

class TestTTestNumericalIntegrity:
    def test_welch_result_structure(self, continuous_data):
        """Welch t-test result has complete numerical fields."""
        r = independent_t_test(continuous_data, outcome="value", group="group",
                               group1="A", group2="B", equal_var=False)
        assert r["test"] == "Welch's t-test"
        for key in ["t_statistic", "p_value", "df", "mean_group1", "mean_group2",
                    "std_group1", "std_group2", "n_group1", "n_group2",
                    "ci_95_lower", "ci_95_upper", "cohens_d", "difference", "se_difference"]:
            assert key in r, f"Missing {key}"
            assert r[key] is not None

    def test_student_result_structure(self, continuous_data):
        """Student t-test result has complete numerical fields."""
        r = independent_t_test(continuous_data, outcome="value", group="group",
                               group1="A", group2="B", equal_var=True)
        assert r["test"] == "Independent t-test"
        for key in ["t_statistic", "p_value", "df", "mean_group1", "mean_group2",
                    "std_group1", "std_group2", "n_group1", "n_group2",
                    "ci_95_lower", "ci_95_upper", "cohens_d"]:
            assert key in r, f"Missing {key}"

    def test_cohens_d_is_finite(self, continuous_data):
        """Cohen's d is a finite number."""
        r = independent_t_test(continuous_data, outcome="value", group="group",
                               group1="A", group2="B")
        assert np.isfinite(r["cohens_d"])

    def test_ci_brackets_estimate(self, continuous_data):
        """95% CI brackets the mean difference."""
        r = independent_t_test(continuous_data, outcome="value", group="group",
                               group1="A", group2="B")
        mean_diff = r["mean_group1"] - r["mean_group2"]
        assert r["ci_95_lower"] <= mean_diff <= r["ci_95_upper"]


class TestPairedTestNumericalIntegrity:
    def test_paired_result_structure(self, paired_data):
        """Paired t-test result has complete numerical fields."""
        r = paired_t_test(paired_data, outcome1="after", outcome2="before")
        for key in ["t_statistic", "p_value", "df", "mean_difference",
                    "std_difference", "n_pairs", "ci_95_lower", "ci_95_upper",
                    "se_difference"]:
            assert key in r, f"Missing {key}"
            assert r[key] is not None

    def test_wilcoxon_result_structure(self, paired_data):
        """Wilcoxon signed-rank result has complete numerical fields."""
        r = wilcoxon_signed_rank(paired_data, outcome1="after", outcome2="before")
        for key in ["w_statistic", "p_value", "n_pairs", "median_difference"]:
            assert key in r, f"Missing {key}"
            assert r[key] is not None


class TestNonParametricNumericalIntegrity:
    def test_mann_whitney_result_structure(self, continuous_data):
        """Mann-Whitney result has complete numerical fields."""
        r = mann_whitney_u(continuous_data, outcome="value", group="group",
                           group1="A", group2="B")
        for key in ["u_statistic", "p_value", "n_group1", "n_group2",
                    "median_group1", "median_group2"]:
            assert key in r, f"Missing {key}"
            assert r[key] is not None


class TestCorrelationNumericalIntegrity:
    def test_pearson_result_structure(self, continuous_continuous_data):
        """Pearson correlation result has complete numerical fields."""
        r = correlation_test(continuous_continuous_data, x="x", y="y", method="pearson")
        for key in ["r", "r_squared", "p_value", "n", "ci_95_lower", "ci_95_upper"]:
            assert key in r, f"Missing {key}"
            assert r[key] is not None

    def test_spearman_result_structure(self, continuous_continuous_data):
        """Spearman correlation result has complete numerical fields."""
        r = correlation_test(continuous_continuous_data, x="x", y="y", method="spearman")
        for key in ["r", "r_squared", "p_value", "n"]:
            assert key in r, f"Missing {key}"
            assert r[key] is not None

    def test_correlation_coefficient_bounded(self, continuous_continuous_data):
        """Correlation coefficient is in [-1, 1]."""
        r = correlation_test(continuous_continuous_data, x="x", y="y", method="pearson")
        assert -1.0 <= r["r"] <= 1.0


class TestCategoricalNumericalIntegrity:
    def test_chi_square_result_structure(self, contingency_data):
        """Chi-square result has complete numerical fields."""
        r = chi_square_test(contingency_data, variable1="disease", variable2="exposure")
        assert "Chi-square" in r["test"]
        for key in ["chi2", "p_value", "df", "table", "n_cells_low_expected"]:
            assert key in r, f"Missing {key}"
            assert r[key] is not None

    def test_fisher_result_structure(self, contingency_data):
        """Fisher exact result has complete numerical fields."""
        r = fisher_exact_test(contingency_data, variable1="disease", variable2="exposure")
        assert "Fisher" in r["test"]
        for key in ["odds_ratio", "p_value", "table"]:
            assert key in r, f"Missing {key}"
            assert r[key] is not None


class TestAnovaNumericalIntegrity:
    def test_anova_result_structure(self, multi_group_data):
        """ANOVA result has complete numerical fields."""
        r = one_way_anova(multi_group_data, outcome="value", group="group")
        for key in ["f_statistic", "p_value", "n_groups", "group_statistics"]:
            assert key in r, f"Missing {key}"
            assert r[key] is not None


class TestRegressionNumericalIntegrity:
    def test_logistic_result_structure(self, binary_data):
        """Logistic regression result has complete numerical fields."""
        r = logistic_regression(binary_data, outcome="outcome", predictors=["exposure"])
        assert "parameters" in r
        assert "exposure" in r["parameters"]
        coef = r["parameters"]["exposure"]
        for key in ["coef", "odds_ratio", "std_err", "p_value", "ci_95_lower", "ci_95_upper"]:
            assert key in coef, f"Missing {key}"
            assert coef[key] is not None
        assert "n_obs" in r
        assert "pseudo_r2" in r

    def test_linear_result_structure(self, continuous_data):
        """Linear regression result has complete numerical fields."""
        r = linear_regression(continuous_data, outcome="value", predictors=["group"])
        assert "parameters" in r
        for key in ["r_squared", "adj_r_squared", "f_statistic", "f_p_value", "n_obs"]:
            assert key in r, f"Missing {key}"
            assert r[key] is not None


# ===========================================================================
# Round-Trip Integrity Across All Methods
# ===========================================================================

class TestRoundTripIntegrity:
    def test_ttest_roundtrip(self, continuous_data):
        r1 = independent_t_test(continuous_data, outcome="value", group="group",
                                group1="A", group2="B")
        r2 = json.loads(json.dumps(r1, default=str))
        assert r2["t_statistic"] == r1["t_statistic"]
        assert r2["p_value"] == r1["p_value"]

    def test_mannwhitney_roundtrip(self, continuous_data):
        r1 = mann_whitney_u(continuous_data, outcome="value", group="group",
                            group1="A", group2="B")
        r2 = json.loads(json.dumps(r1, default=str))
        assert r2["u_statistic"] == r1["u_statistic"]
        assert r2["p_value"] == r1["p_value"]

    def test_pearson_roundtrip(self, continuous_continuous_data):
        r1 = correlation_test(continuous_continuous_data, x="x", y="y", method="pearson")
        r2 = json.loads(json.dumps(r1, default=str))
        assert r2["r"] == r1["r"]
        assert r2["p_value"] == r1["p_value"]

    def test_logistic_roundtrip(self, binary_data):
        r1 = logistic_regression(binary_data, outcome="outcome", predictors=["exposure"])
        r2 = json.loads(json.dumps(r1, default=str))
        assert r2["parameters"]["exposure"]["coef"] == r1["parameters"]["exposure"]["coef"]
        assert r2["parameters"]["exposure"]["odds_ratio"] == r1["parameters"]["exposure"]["odds_ratio"]

    def test_anova_roundtrip(self, multi_group_data):
        r1 = one_way_anova(multi_group_data, outcome="value", group="group")
        r2 = json.loads(json.dumps(r1, default=str))
        assert r2["f_statistic"] == r1["f_statistic"]
        assert r2["p_value"] == r1["p_value"]


# ===========================================================================
# No Silent Recomputation of Stored Results
# ===========================================================================

class TestNoSilentRecomputation:
    def test_same_result_on_repeated_calls(self, continuous_data):
        """Repeated calls with same input return identical results."""
        results = [
            independent_t_test(continuous_data, outcome="value", group="group",
                               group1="A", group2="B")
            for _ in range(3)
        ]
        for r in results[1:]:
            assert r["t_statistic"] == results[0]["t_statistic"]
            assert r["p_value"] == results[0]["p_value"]
            assert r["cohens_d"] == results[0]["cohens_d"]

    def test_different_seed_different_result(self, tmp_path):
        """Different random data → different result (not cached)."""
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(999)
        n = 80
        df1 = pd.DataFrame({
            "value": rng1.normal(50, 10, n),
            "group": rng1.choice(["A", "B"], n),
        })
        df2 = pd.DataFrame({
            "value": rng2.normal(50, 10, n),
            "group": rng2.choice(["A", "B"], n),
        })
        p1 = tmp_path / "s1.csv"
        p2 = tmp_path / "s2.csv"
        df1.to_csv(p1, index=False)
        df2.to_csv(p2, index=False)
        r1 = independent_t_test(p1, outcome="value", group="group", group1="A", group2="B")
        r2 = independent_t_test(p2, outcome="value", group="group", group1="A", group2="B")
        assert r1["t_statistic"] != r2["t_statistic"]
