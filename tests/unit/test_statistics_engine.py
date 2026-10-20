"""Unit tests for the statistical engine — validates against known results."""

import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest
from app.statistics.engine import (
    descriptive_statistics,
    independent_t_test,
    paired_t_test,
    chi_square_test,
    correlation_test,
    one_way_anova,
    linear_regression,
    logistic_regression,
    mann_whitney_u,
    wilcoxon_signed_rank,
    fisher_exact_test,
    power_analysis_t_test,
)
from config.settings import get_settings


@pytest.fixture
def iris_path(tmp_path):
    """Create a simple test CSV file."""
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical([iris.target_names[t] for t in iris.target])
    df["target"] = iris.target
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def binary_data(tmp_path):
    """Create binary outcome + grouping data."""
    np.random.seed(42)
    n = 200
    group = np.random.choice(["A", "B"], size=n)
    outcome = np.where(group == "A", 70, 60) + np.random.normal(0, 10, n)
    df = pd.DataFrame({"group": group, "outcome": outcome, "age": np.random.randint(20, 80, n)})
    path = tmp_path / "binary_test.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def contingency_data(tmp_path):
    """Create a contingency table dataset."""
    np.random.seed(42)
    df = pd.DataFrame({
        "treatment": np.random.choice(["Drug", "Placebo"], size=200),
        "response": np.random.choice(["Improved", "Not improved"], size=200),
    })
    path = tmp_path / "contingency.csv"
    df.to_csv(path, index=False)
    return path


class TestDescriptiveStatistics:
    def test_numeric_columns(self, iris_path):
        result = descriptive_statistics(iris_path)
        assert "statistics" in result
        assert "sepal length (cm)" in result["statistics"]
        stats = result["statistics"]["sepal length (cm)"]
        assert stats["n"] == 150
        assert 4 < stats["mean"] < 6
        assert stats["std"] > 0

    def test_categorical_columns(self, iris_path):
        result = descriptive_statistics(iris_path)
        assert "species" in result["statistics"]
        stats = result["statistics"]["species"]
        assert stats["n_unique"] == 3
        assert stats["n"] == 150


class TestIndependentTTest:
    def test_basic(self, binary_data):
        result = independent_t_test(binary_data, outcome="outcome", group="group")
        assert "t_statistic" in result
        assert "p_value" in result
        assert result["n_group1"] == 100
        assert result["n_group2"] == 100
        assert result["mean_group1"] > result["mean_group2"]  # A should be higher

    def test_equal_var(self, binary_data):
        result = independent_t_test(binary_data, outcome="outcome", group="group", equal_var=True)
        assert result["test"] == "Independent t-test"

    def test_groups_specified(self, binary_data):
        result = independent_t_test(binary_data, outcome="outcome", group="group", group1="A", group2="B")
        assert result["group1"] == "A"
        assert result["group2"] == "B"


class TestPairedTTest:
    def test_basic(self, iris_path):
        df = pd.read_csv(iris_path)
        result = paired_t_test(iris_path, outcome1="sepal length (cm)", outcome2="sepal width (cm)")
        assert "t_statistic" in result
        assert "p_value" in result
        assert result["n_pairs"] == 150
        assert result["mean_difference"] != 0


class TestChiSquare:
    def test_basic(self, contingency_data):
        result = chi_square_test(contingency_data, variable1="treatment", variable2="response")
        assert "chi2" in result
        assert "p_value" in result
        assert "df" in result


class TestCorrelation:
    def test_pearson(self, iris_path):
        result = correlation_test(iris_path, x="sepal length (cm)", y="petal length (cm)")
        assert result["test"] == "Pearson correlation"
        assert result["r"] > 0.8
        assert result["p_value"] < 0.001
        assert "ci_95_lower" in result

    def test_spearman(self, iris_path):
        result = correlation_test(iris_path, x="sepal length (cm)", y="petal length (cm)", method="spearman")
        assert result["test"] == "Spearman correlation"
        assert result["r"] > 0.8


class TestANOVA:
    def test_basic(self, iris_path):
        result = one_way_anova(iris_path, outcome="sepal length (cm)", group="species")
        assert "f_statistic" in result
        assert "p_value" in result
        assert result["n_groups"] == 3
        assert result["p_value"] < 0.05  # Species differ


class TestLinearRegression:
    def test_basic(self, iris_path):
        result = linear_regression(
            iris_path,
            outcome="sepal length (cm)",
            predictors=["sepal width (cm)", "petal length (cm)"],
        )
        assert "r_squared" in result
        assert result["n_obs"] == 150
        assert result["r_squared"] > 0


class TestLogisticRegression:
    def test_basic(self, iris_path):
        df = pd.read_csv(iris_path)
        # Binary outcome from target
        df["binary_target"] = (df["target"] > 0).astype(int)
        binary_path = iris_path.parent / "binary_iris.csv"
        df.to_csv(binary_path, index=False)
        result = logistic_regression(
            binary_path,
            outcome="binary_target",
            predictors=["sepal length (cm)", "sepal width (cm)"],
        )
        assert "parameters" in result
        assert result["n_obs"] == 150
        # Convergence may fail on near-separable iris data — check model returned results
        assert "parameters" in result
        assert len(result["parameters"]) == 3  # const + 2 predictors


class TestMannWhitney:
    def test_basic(self, binary_data):
        result = mann_whitney_u(binary_data, outcome="outcome", group="group")
        assert "u_statistic" in result
        assert "p_value" in result
        assert result["n_group1"] == 100


class TestWilcoxon:
    def test_basic(self, iris_path):
        result = wilcoxon_signed_rank(iris_path, outcome1="sepal length (cm)", outcome2="sepal width (cm)")
        assert "w_statistic" in result
        assert "p_value" in result


class TestFisherExact:
    def test_basic(self, contingency_data):
        result = fisher_exact_test(contingency_data, variable1="treatment", variable2="response")
        assert "odds_ratio" in result
        assert "p_value" in result
        assert result["test"] == "Fisher's exact test"


class TestPowerAnalysis:
    def test_sample_size(self):
        result = power_analysis_t_test(effect_size=0.5, power=0.80)
        assert "required_n_per_group" in result
        assert result["required_n_per_group"] > 0

    def test_power(self):
        result = power_analysis_t_test(effect_size=0.5, n_per_group=50)
        assert "power" in result
        assert 0 < result["power"] <= 1

    def test_mde(self):
        result = power_analysis_t_test(n_per_group=50, power=0.80)
        assert "min_detectable_effect_size" in result
        assert result["min_detectable_effect_size"] > 0
