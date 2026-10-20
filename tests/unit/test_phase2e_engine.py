"""Phase 2E Execution Engine Tests — Deterministic Statistical Execution.

Covers:
- StatisticalResult contract (creation, fingerprint, summary, serialization)
- Dataset verification (hashes, schema, variable presence, complete cases)
- Input validation (missing vars, sample size, group requirements, constant predictors)
- Numerical sanity validation (p-values, correlations, effect sizes, CIs, F/chi2/U stats)
- Guardrail blocking (missing variables, insufficient sample, unknown methods)
- Execution of all 8 primary methods through StatisticalExecutor
- End-to-end guardrails + sanity + result fingerprinting
"""

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hashlib
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def continuous_group_data(tmp_path):
    """Two-group continuous outcome with known difference."""
    rng = np.random.RandomState(42)
    n_per = 60
    grp = np.array(["A"] * n_per + ["B"] * n_per)
    vals = np.concatenate([
        rng.normal(100, 15, n_per),
        rng.normal(110, 15, n_per),
    ])
    df = pd.DataFrame({"group": grp, "score": vals, "cov1": rng.normal(0, 1, 2 * n_per)})
    path = tmp_path / "continuous_group.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def paired_data(tmp_path):
    """Pre/post paired measurements."""
    rng = np.random.RandomState(42)
    n = 80
    pre = rng.normal(50, 10, n)
    post = pre + rng.normal(-5, 8, n)  # systematic decrease
    df = pd.DataFrame({"pre": pre, "post": post, "id": range(n)})
    path = tmp_path / "paired.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def contingency_data(tmp_path):
    """2x2 contingency table."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "treatment": rng.choice(["Drug", "Placebo"], n),
        "response": rng.choice(["Improved", "Not improved"], n),
    })
    path = tmp_path / "contingency.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def correlation_data(tmp_path):
    """Two correlated continuous variables."""
    rng = np.random.RandomState(42)
    n = 100
    x = rng.normal(0, 1, n)
    y = 0.7 * x + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"x": x, "y": y})
    path = tmp_path / "correlation.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def regression_data(tmp_path):
    """Continuous outcome with multiple predictors."""
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
def logistic_data(tmp_path):
    """Binary outcome with predictors."""
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


def _uid() -> str:
    return f"test_{uuid.uuid4().hex[:8]}"


# ===========================================================================
# STATISTICALRESULT CONTRACT
# ===========================================================================

class TestStatisticalResult:
    def test_creation_with_defaults(self):
        from app.statistics.result import StatisticalResult, ExecutionStatus
        r = StatisticalResult(method_id="independent_t_test", method_name="t-test")
        assert r.method_id == "independent_t_test"
        assert r.method_name == "t-test"
        assert r.execution_status == ExecutionStatus.SUCCESS
        assert r.test_statistic is None
        assert r.p_value is None
        assert r.effect_size is None
        assert r.confidence_interval is None
        assert r.n_observed is None
        assert r.guardrails_passed is True
        assert r.engine_version == "2e.1"

    def test_full_creation(self):
        from app.statistics.result import StatisticalResult, ExecutionStatus, SanityStatus
        r = StatisticalResult(
            method_id="independent_t_test",
            method_name="Independent t-test",
            execution_status=ExecutionStatus.SUCCESS,
            test_statistic=2.5,
            p_value=0.013,
            effect_size=0.5,
            effect_size_name="Cohen's d",
            confidence_interval=(0.1, 0.9),
            n_observed=100,
            n_groups=2,
            group_sizes={"A": 50, "B": 50},
        )
        assert r.test_statistic == 2.5
        assert r.p_value == 0.013
        assert r.effect_size == 0.5
        assert r.confidence_interval == (0.1, 0.9)
        assert r.n_observed == 100
        assert r.group_sizes == {"A": 50, "B": 50}

    def test_fingerprint_deterministic(self):
        from app.statistics.result import StatisticalResult
        r1 = StatisticalResult(method_id="t", method_name="t", test_statistic=2.5, p_value=0.013, n_observed=100)
        r2 = StatisticalResult(method_id="t", method_name="t", test_statistic=2.5, p_value=0.013, n_observed=100)
        fp1 = r1.compute_fingerprint()
        fp2 = r2.compute_fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256

    def test_fingerprint_different_values(self):
        from app.statistics.result import StatisticalResult
        r1 = StatisticalResult(method_id="t", method_name="t", test_statistic=2.5, p_value=0.013)
        r2 = StatisticalResult(method_id="t", method_name="t", test_statistic=3.1, p_value=0.002)
        fp1 = r1.compute_fingerprint()
        fp2 = r2.compute_fingerprint()
        assert fp1 != fp2

    def test_to_summary(self):
        from app.statistics.result import StatisticalResult, ExecutionStatus
        r = StatisticalResult(
            method_id="t", method_name="t-test",
            execution_status=ExecutionStatus.SUCCESS,
            test_statistic=2.0, p_value=0.05, n_observed=50,
            guardrails_passed=True,
        )
        s = r.to_summary()
        assert s["method_id"] == "t"
        assert s["execution_status"] == "success"
        assert s["test_statistic"] == 2.0
        assert s["p_value"] == 0.05
        assert s["n_observed"] == 50
        assert s["guardrails_passed"] is True

    def test_serialization_roundtrip(self):
        from app.statistics.result import StatisticalResult
        r = StatisticalResult(method_id="t", method_name="t-test", test_statistic=2.5, p_value=0.013)
        data = r.model_dump()
        r2 = StatisticalResult.model_validate(data)
        assert r2.method_id == r.method_id
        assert r2.test_statistic == r.test_statistic
        assert r2.p_value == r.p_value


# ===========================================================================
# DATASET VERIFICATION
# ===========================================================================

class TestDatasetVerification:
    def test_basic_verification(self, continuous_group_data):
        from app.statistics.executor import verify_dataset
        v = verify_dataset(continuous_group_data, variables_required=["group", "score"])
        assert v.content_hash is not None
        assert v.schema_hash is not None
        assert v.n_rows == 120
        assert v.n_cols == 3
        assert "group" in v.columns
        assert "score" in v.columns
        assert v.n_complete_cases == 120

    def test_hash_matches_recompute(self, continuous_group_data):
        from app.statistics.executor import verify_dataset
        v1 = verify_dataset(continuous_group_data)
        v2 = verify_dataset(continuous_group_data)
        assert v1.content_hash == v2.content_hash
        assert v1.schema_hash == v2.schema_hash

    def test_hash_mismatch_warning(self, continuous_group_data):
        from app.statistics.executor import verify_dataset
        v = verify_dataset(continuous_group_data, expected_content_hash="wrong_hash_value")
        assert len(v.verification_warnings) > 0
        assert "CONTENT_HASH_MISMATCH" in v.verification_warnings[0]

    def test_missing_variable_warning(self, continuous_group_data):
        from app.statistics.executor import verify_dataset
        v = verify_dataset(continuous_group_data, variables_required=["group", "score", "nonexistent"])
        assert any("not found" in w.lower() for w in v.verification_warnings)

    def test_complete_cases_count(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, None, 4], "b": [1, None, 3, 4]})
        path = tmp_path / "missing.csv"
        df.to_csv(path, index=False)
        from app.statistics.executor import verify_dataset
        v = verify_dataset(path, variables_required=["a", "b"])
        assert v.n_complete_cases == 2

    def test_different_files_different_hash(self, tmp_path):
        from app.statistics.executor import verify_dataset
        df1 = pd.DataFrame({"x": [1, 2, 3]})
        df2 = pd.DataFrame({"x": [1, 2, 4]})
        p1 = tmp_path / "f1.csv"
        p2 = tmp_path / "f2.csv"
        df1.to_csv(p1, index=False)
        df2.to_csv(p2, index=False)
        v1 = verify_dataset(p1)
        v2 = verify_dataset(p2)
        assert v1.content_hash != v2.content_hash


# ===========================================================================
# INPUT VALIDATION
# ===========================================================================

class TestInputValidation:
    def test_valid_inputs(self, continuous_group_data):
        from app.statistics.executor import validate_inputs
        import pandas as pd
        df = pd.read_csv(continuous_group_data)
        warnings, blockers = validate_inputs(
            df, "independent_t_test", outcome="score", group="group"
        )
        assert blockers == []

    def test_missing_variable(self, continuous_group_data):
        from app.statistics.executor import validate_inputs
        import pandas as pd
        df = pd.read_csv(continuous_group_data)
        warnings, blockers = validate_inputs(
            df, "independent_t_test", outcome="nonexistent", group="group"
        )
        assert len(blockers) > 0
        assert "nonexistent" in blockers[0]

    def test_unknown_method(self, continuous_group_data):
        from app.statistics.executor import validate_inputs
        import pandas as pd
        df = pd.read_csv(continuous_group_data)
        warnings, blockers = validate_inputs(df, "fake_method", outcome="score")
        assert len(blockers) > 0
        assert "Unknown method" in blockers[0]

    def test_insufficient_sample_size(self, tmp_path):
        from app.statistics.executor import validate_inputs
        df = pd.DataFrame({"group": ["A", "B"], "score": [1.0, 2.0]})
        path = tmp_path / "small.csv"
        df.to_csv(path, index=False)
        df_loaded = pd.read_csv(path)
        warnings, blockers = validate_inputs(
            df_loaded, "independent_t_test", outcome="score", group="group"
        )
        assert len(blockers) > 0
        assert "at least" in blockers[0]

    def test_insufficient_groups(self, tmp_path):
        from app.statistics.executor import validate_inputs
        df = pd.DataFrame({
            "group": ["A"] * 20,
            "score": np.random.normal(0, 1, 20),
        })
        path = tmp_path / "onegroup.csv"
        df.to_csv(path, index=False)
        df_loaded = pd.read_csv(path)
        warnings, blockers = validate_inputs(
            df_loaded, "independent_t_test", outcome="score", group="group"
        )
        assert len(blockers) > 0
        assert "at least" in blockers[0]

    def test_constant_predictor_warning(self, tmp_path):
        from app.statistics.executor import validate_inputs
        df = pd.DataFrame({
            "y": np.random.normal(0, 1, 50),
            "x": [5.0] * 50,  # constant
        })
        path = tmp_path / "const.csv"
        df.to_csv(path, index=False)
        df_loaded = pd.read_csv(path)
        warnings, blockers = validate_inputs(
            df_loaded, "linear_regression", outcome="y", predictors=["x"]
        )
        assert len(blockers) > 0
        assert "no variation" in blockers[0].lower()

    def test_no_complete_cases(self, tmp_path):
        from app.statistics.executor import validate_inputs
        df = pd.DataFrame({
            "a": [np.nan] * 20,
            "b": [1.0] * 20,
        })
        path = tmp_path / "allnan.csv"
        df.to_csv(path, index=False)
        df_loaded = pd.read_csv(path)
        warnings, blockers = validate_inputs(
            df_loaded, "pearson_correlation", variables=["a", "b"]
        )
        assert len(blockers) > 0
        assert "complete cases" in blockers[0].lower()


# ===========================================================================
# NUMERICAL SANITY VALIDATION
# ===========================================================================

class TestNumericalSanity:
    def test_passing_sanity(self):
        from app.statistics.executor import validate_numerical_sanity
        from app.statistics.result import SanityStatus
        result = {
            "t_statistic": 2.5,
            "p_value": 0.013,
            "cohens_d": 0.5,
            "ci_95_lower": 0.1,
            "ci_95_upper": 0.9,
            "n_group1": 50,
            "n_group2": 50,
            "df": 98,
        }
        sanity = validate_numerical_sanity("independent_t_test", result)
        assert sanity.status == SanityStatus.PASS
        assert len(sanity.blockers) == 0

    def test_p_value_out_of_range(self):
        from app.statistics.executor import validate_numerical_sanity
        from app.statistics.result import SanityStatus
        result = {"p_value": -0.05}
        sanity = validate_numerical_sanity("t", result)
        assert sanity.status == SanityStatus.FAIL
        assert any("p-value" in b.lower() for b in sanity.blockers)

    def test_p_value_greater_than_one(self):
        from app.statistics.executor import validate_numerical_sanity
        from app.statistics.result import SanityStatus
        result = {"p_value": 1.5}
        sanity = validate_numerical_sanity("t", result)
        assert sanity.status == SanityStatus.FAIL

    def test_correlation_out_of_range(self):
        from app.statistics.executor import validate_numerical_sanity
        from app.statistics.result import SanityStatus
        result = {"r": 1.5}
        sanity = validate_numerical_sanity("pearson", result)
        assert sanity.status == SanityStatus.FAIL
        assert any("correlation" in b.lower() for b in sanity.blockers)

    def test_non_finite_p_value(self):
        from app.statistics.executor import validate_numerical_sanity
        from app.statistics.result import SanityStatus
        result = {"p_value": float("nan")}
        sanity = validate_numerical_sanity("t", result)
        assert sanity.status == SanityStatus.FAIL

    def test_infinite_correlation(self):
        from app.statistics.executor import validate_numerical_sanity
        from app.statistics.result import SanityStatus
        result = {"r": float("inf")}
        sanity = validate_numerical_sanity("pearson", result)
        assert sanity.status == SanityStatus.FAIL

    def test_negative_sample_size(self):
        from app.statistics.executor import validate_numerical_sanity
        from app.statistics.result import SanityStatus
        result = {"n_group1": -5, "n_group2": 50}
        sanity = validate_numerical_sanity("t", result)
        assert sanity.status == SanityStatus.FAIL

    def test_negative_chi_square(self):
        from app.statistics.executor import validate_numerical_sanity
        from app.statistics.result import SanityStatus
        result = {"chi2": -1.5}
        sanity = validate_numerical_sanity("chi_square", result)
        assert sanity.status == SanityStatus.FAIL

    def test_negative_f_statistic(self):
        from app.statistics.executor import validate_numerical_sanity
        from app.statistics.result import SanityStatus
        result = {"f_statistic": -2.0}
        sanity = validate_numerical_sanity("anova", result)
        assert sanity.status == SanityStatus.FAIL

    def test_ci_lower_greater_than_upper(self):
        from app.statistics.executor import validate_numerical_sanity
        from app.statistics.result import SanityStatus
        result = {"ci_95_lower": 0.9, "ci_95_upper": 0.1}
        sanity = validate_numerical_sanity("t", result)
        assert sanity.status == SanityStatus.WARNING
        assert len(sanity.warnings) > 0

    def test_zero_df_warning(self):
        from app.statistics.executor import validate_numerical_sanity
        from app.statistics.result import SanityStatus
        result = {"df": 0}
        sanity = validate_numerical_sanity("t", result)
        assert sanity.status == SanityStatus.WARNING

    def test_all_checks_recorded(self):
        from app.statistics.executor import validate_numerical_sanity
        result = {
            "t_statistic": 2.0,
            "p_value": 0.05,
            "cohens_d": 0.4,
            "ci_95_lower": 0.0,
            "ci_95_upper": 0.8,
            "n_group1": 50,
            "n_group2": 50,
            "df": 98,
        }
        sanity = validate_numerical_sanity("t", result)
        assert len(sanity.checks) >= 5


# ===========================================================================
# EXECUTOR — METHOD EXECUTION
# ===========================================================================

class TestExecutorIndependentTTest:
    def test_basic_execution(self, continuous_group_data):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus, SanityStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test",
            continuous_group_data,
            outcome="score",
            group="group",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.test_statistic is not None
        assert result.p_value is not None
        assert result.p_value >= 0
        assert result.p_value <= 1
        assert result.n_observed == 120
        assert result.n_groups == 2
        assert result.group_sizes is not None
        assert result.data_verification.n_complete_cases == 120
        assert result.numerical_sanity.status in (SanityStatus.PASS, SanityStatus.WARNING)
        assert result.result_fingerprint is not None
        assert len(result.result_fingerprint) == 64

    def test_with_groups_specified(self, continuous_group_data):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test",
            continuous_group_data,
            outcome="score",
            group="group",
            group1="A",
            group2="B",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert "A" in result.group_sizes
        assert "B" in result.group_sizes

    def test_guardrail_blocks_missing_var(self, continuous_group_data):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test",
            continuous_group_data,
            outcome="nonexistent",
            group="group",
        )
        assert result.execution_status == ExecutionStatus.BLOCKED
        assert result.guardrails_passed is False
        assert len(result.guardrail_blockers) > 0

    def test_guardrail_blocks_insufficient_sample(self, tmp_path):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus
        df = pd.DataFrame({"group": ["A", "B"], "score": [1.0, 2.0]})
        path = tmp_path / "tiny.csv"
        df.to_csv(path, index=False)
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test",
            path,
            outcome="score",
            group="group",
        )
        assert result.execution_status == ExecutionStatus.BLOCKED
        assert "at least" in result.guardrail_blockers[0]


class TestExecutorPairedTTest:
    def test_basic_execution(self, paired_data):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "paired_t_test",
            paired_data,
            outcome1="pre",
            outcome2="post",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.test_statistic is not None
        assert result.p_value is not None
        assert result.n_observed == 80


class TestExecutorChiSquare:
    def test_basic_execution(self, contingency_data):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "chi_square",
            contingency_data,
            variable1="treatment",
            variable2="response",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.test_statistic is not None
        assert result.p_value is not None


class TestExecutorFisherExact:
    def test_basic_execution(self, contingency_data):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "fisher_exact",
            contingency_data,
            variable1="treatment",
            variable2="response",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.test_statistic is not None
        assert result.p_value is not None


class TestExecutorCorrelation:
    def test_pearson(self, correlation_data):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "pearson_correlation",
            correlation_data,
            x="x",
            y="y",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.test_statistic is not None
        assert abs(result.test_statistic) <= 1.0
        assert result.p_value is not None
        assert result.effect_size is not None

    def test_spearman(self, correlation_data):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "spearman_correlation",
            correlation_data,
            x="x",
            y="y",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.test_statistic is not None


class TestExecutorANOVA:
    def test_basic_execution(self, continuous_group_data):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus
        executor = StatisticalExecutor()
        # Add a third group
        import pandas as pd
        df = pd.read_csv(continuous_group_data)
        df.loc[df.sample(frac=0.3, random_state=42).index, "group"] = "C"
        path = continuous_group_data.parent / "anova_data.csv"
        df.to_csv(path, index=False)
        result = executor.execute(
            "one_way_anova",
            path,
            outcome="score",
            group="group",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.test_statistic is not None
        assert result.p_value is not None
        assert result.n_groups == 3


class TestExecutorLinearRegression:
    def test_basic_execution(self, regression_data):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "linear_regression",
            regression_data,
            outcome="y",
            predictors=["x1", "x2"],
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.test_statistic is not None
        assert result.p_value is not None
        assert result.n_observed == 150
        assert result.effect_size is not None
        assert result.effect_size_name == "R-squared"


class TestExecutorLogisticRegression:
    def test_basic_execution(self, logistic_data):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "logistic_regression",
            logistic_data,
            outcome="y",
            predictors=["x1", "x2"],
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.p_value is not None
        assert result.n_observed == 200
        assert result.effect_size_name == "Pseudo R-squared"


class TestExecutorMannWhitney:
    def test_basic_execution(self, continuous_group_data):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.result import ExecutionStatus
        executor = StatisticalExecutor()
        result = executor.execute(
            "mann_whitney_u",
            continuous_group_data,
            outcome="score",
            group="group",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.test_statistic is not None
        assert result.p_value is not None
        assert result.n_observed == 120


# ===========================================================================
# END-TO-END: HASH VERIFICATION
# ===========================================================================

class TestEndToEndHashVerification:
    def test_same_file_same_hash(self, continuous_group_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        r1 = executor.execute(
            "independent_t_test", continuous_group_data,
            outcome="score", group="group",
        )
        r2 = executor.execute(
            "independent_t_test", continuous_group_data,
            outcome="score", group="group",
        )
        assert r1.data_verification.content_hash == r2.data_verification.content_hash
        assert r1.result_fingerprint == r2.result_fingerprint

    def test_wrong_hash_triggers_warning(self, continuous_group_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", continuous_group_data,
            outcome="score", group="group",
            expected_content_hash="wrong_hash",
        )
        assert len(result.data_verification.verification_warnings) > 0

    def test_correct_hash_no_warning(self, continuous_group_data):
        from app.statistics.executor import StatisticalExecutor
        from app.statistics.executor import verify_dataset
        executor = StatisticalExecutor()
        v = verify_dataset(continuous_group_data)
        result = executor.execute(
            "independent_t_test", continuous_group_data,
            outcome="score", group="group",
            expected_content_hash=v.content_hash,
        )
        content_warnings = [w for w in result.data_verification.verification_warnings if "Content hash" in w]
        assert len(content_warnings) == 0


# ===========================================================================
# INTEGRATION: RESULT FINGERPRINT REPRODUCIBILITY
# ===========================================================================

class TestFingerprintReproducibility:
    def test_same_inputs_same_fingerprint(self, continuous_group_data):
        from app.statistics.executor import StatisticalExecutor
        executor = StatisticalExecutor()
        fp1 = executor.execute(
            "independent_t_test", continuous_group_data,
            outcome="score", group="group",
        ).result_fingerprint
        fp2 = executor.execute(
            "independent_t_test", continuous_group_data,
            outcome="score", group="group",
        ).result_fingerprint
        assert fp1 == fp2

    def test_different_data_different_fingerprint(self, tmp_path):
        from app.statistics.executor import StatisticalExecutor
        rng = np.random.RandomState(42)
        df1 = pd.DataFrame({"group": ["A"]*30+["B"]*30, "score": rng.normal(100, 15, 60)})
        df2 = pd.DataFrame({"group": ["A"]*30+["B"]*30, "score": rng.normal(100, 30, 60)})
        p1 = tmp_path / "d1.csv"
        p2 = tmp_path / "d2.csv"
        df1.to_csv(p1, index=False)
        df2.to_csv(p2, index=False)
        executor = StatisticalExecutor()
        fp1 = executor.execute("independent_t_test", p1, outcome="score", group="group").result_fingerprint
        fp2 = executor.execute("independent_t_test", p2, outcome="score", group="group").result_fingerprint
        assert fp1 != fp2
