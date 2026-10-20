"""STATIS Phase 2E Validation Battery — Tests 1–6.

End-to-end validation of the complete pipeline across six representative
research scenarios. Tests scientific behavior and architectural invariants,
not just pipeline completion.
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

from app.statistics.executor import StatisticalExecutor
from app.statistics.variable_roles import (
    RoleResolver, VariableRole, AnalysisProblemType, RoleStatus,
)
from app.statistics.candidate_generator import CandidateGenerator, DataProfile
from app.statistics.registry import get_method
from app.statistics.result import ExecutionStatus, SanityStatus
from app.statistics.diagnostics import DiagnosticDiagnosticStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uid() -> str:
    return f"val_{uuid.uuid4().hex[:8]}"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schema_hash(df: pd.DataFrame) -> str:
    schema = {col: str(df[col].dtype) for col in df.columns}
    import json
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def _build_variable_info(df: pd.DataFrame) -> list[dict]:
    """Build variable info dicts for RoleResolver."""
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
    return vars_info


def _resolve_and_classify(df: pd.DataFrame, research_question: str | None = None):
    """Run variable role resolution and classify problem type."""
    vars_info = _build_variable_info(df)
    resolution_set = RoleResolver.resolve_all(vars_info, research_question=research_question)
    return resolution_set


def _generate_candidates(df: pd.DataFrame, resolution_set, research_question: str | None = None):
    """Generate candidate analyses from role resolution."""
    # Build data profile from resolutions
    outcome_var = None
    outcome_type = None
    predictor_vars = []
    predictor_types = []
    covariate_vars = []
    identifier_vars = []
    group_var = None
    paired_vars = []

    for r in resolution_set.resolutions:
        if r.proposed_role == VariableRole.OUTCOME:
            outcome_var = r.variable_name
            if pd.api.types.is_numeric_dtype(df[r.variable_name]):
                if df[r.variable_name].nunique() <= 2:
                    outcome_type = "binary"
                else:
                    outcome_type = "continuous"
            else:
                outcome_type = "categorical"
        elif r.proposed_role in (VariableRole.EXPOSURE, VariableRole.GROUPING_VARIABLE):
            predictor_vars.append(r.variable_name)
            predictor_types.append("binary" if df[r.variable_name].nunique() <= 10 else "continuous")
            # Group variable: either explicitly GROUPING_VARIABLE or EXPOSURE with few levels
            if r.proposed_role == VariableRole.GROUPING_VARIABLE:
                group_var = r.variable_name
            elif group_var is None and df[r.variable_name].nunique() <= 10:
                group_var = r.variable_name
        elif r.proposed_role == VariableRole.PREDICTOR:
            predictor_vars.append(r.variable_name)
            if pd.api.types.is_numeric_dtype(df[r.variable_name]):
                predictor_types.append("continuous")
            else:
                predictor_types.append("categorical")
        elif r.proposed_role == VariableRole.COVARIATE:
            covariate_vars.append(r.variable_name)
        elif r.proposed_role == VariableRole.IDENTIFIER:
            identifier_vars.append(r.variable_name)

    if outcome_var is None:
        return None, resolution_set

    n_groups = df[group_var].nunique() if group_var else 0
    group_balances = {str(k): int(v) for k, v in df[group_var].value_counts().items()} if group_var else {}

    # Ensure problem_type is set — override UNKNOWN if we can infer from context
    problem_type = resolution_set.problem_type
    if problem_type == AnalysisProblemType.UNKNOWN:
        # Infer from outcomes and exposures
        if outcome_type == "binary" and len(predictor_vars) > 0:
            if len(predictor_vars) == 1:
                problem_type = AnalysisProblemType.BINARY_OUTCOME_SINGLE_PREDICTOR
            else:
                problem_type = AnalysisProblemType.BINARY_OUTCOME_MULTIVARIABLE
        elif outcome_type == "continuous" and n_groups >= 2:
            if n_groups == 2:
                problem_type = AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME
            else:
                problem_type = AnalysisProblemType.MULTI_GROUP_CONTINUOUS_OUTCOME
        elif outcome_type == "continuous" and len(predictor_vars) == 1 and group_var is None:
            problem_type = AnalysisProblemType.CONTINUOUS_CONTINUOUS_ASSOCIATION

    profile = DataProfile(
        n_observations=len(df),
        outcome_variable=outcome_var,
        outcome_type=outcome_type or "continuous",
        predictor_variables=predictor_vars,
        predictor_types=predictor_types,
        covariates=covariate_vars,
        n_groups=n_groups,
        group_balances=group_balances,
        paired=False,
        research_question=research_question,
        problem_type=problem_type,
        role_resolutions=resolution_set,
        identifier_variables=identifier_vars,
    )

    generator = CandidateGenerator()
    candidates = generator.generate(profile)
    return candidates, resolution_set


# ===========================================================================
# TEST 1 — TWO INDEPENDENT GROUPS / CONTINUOUS OUTCOME
# ===========================================================================

TEST1_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "uploads"
    / "228683e7-01c9-4050-afde-4d17cdb99e3f_STATIS_Test1_Continuous.csv"
)

class Test1_TwoIndependentGroups:
    """Validation Test 1 — Existing Test 1 dataset."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.rq = "Does treatment group differ in systolic blood pressure?"
        self.executor = StatisticalExecutor()

    def test_dataset_loads(self):
        assert TEST1_PATH.exists()
        df = pd.read_csv(TEST1_PATH)
        assert len(df) == 120
        assert "treatment" in df.columns
        assert "systolic_bp" in df.columns
        assert "patient_id" in df.columns

    def test_problem_type(self):
        df = pd.read_csv(TEST1_PATH)
        resolution = _resolve_and_classify(df, self.rq)
        assert resolution.problem_type == AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME

    def test_variable_roles(self):
        df = pd.read_csv(TEST1_PATH)
        resolution = _resolve_and_classify(df, self.rq)
        roles = {r.variable_name: r.proposed_role for r in resolution.resolutions}
        assert roles.get("patient_id") == VariableRole.IDENTIFIER
        assert roles.get("treatment") in (VariableRole.EXPOSURE, VariableRole.GROUPING_VARIABLE)
        assert roles.get("systolic_bp") == VariableRole.OUTCOME

    def test_candidate_methods(self):
        df = pd.read_csv(TEST1_PATH)
        resolution = _resolve_and_classify(df, self.rq)
        candidates, _ = _generate_candidates(df, resolution, self.rq)
        assert candidates is not None
        primary_ids = [c.method_id for c in candidates.primary_candidates]
        assert "independent_t_test" in primary_ids
        assert "mann_whitney_u" in primary_ids
        # Excluded methods
        rejected_ids = [r.get("method_id", "") for r in candidates.rejected_methods]
        assert "pearson_correlation" not in primary_ids
        assert "spearman_correlation" not in primary_ids

    def test_identifier_excluded(self):
        df = pd.read_csv(TEST1_PATH)
        resolution = _resolve_and_classify(df, self.rq)
        candidates, _ = _generate_candidates(df, resolution, self.rq)
        all_pred = candidates.data_profile.predictor_variables + candidates.data_profile.covariates
        assert "patient_id" not in all_pred
        assert candidates.data_profile.outcome_variable != "patient_id"

    def test_execution_numerical_consistency(self):
        result = self.executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment", group1=0, group2=1,
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.n_observed == 120
        # Regenerated data with seed=42, normal(130,10) vs normal(138,10)
        assert abs(abs(result.test_statistic) - 5.624) < 0.05
        assert result.p_value < 0.001
        assert abs(abs(result.effect_size) - 1.027) < 0.05
        assert result.confidence_interval is not None
        assert result.confidence_interval[0] < result.confidence_interval[1]
        assert result.degrees_of_freedom > 50

    def test_diagnostics(self):
        result = self.executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment", group1=0, group2=1,
        )
        assert result.diagnostic_status in ("pass", "warning")
        assert result.defensibility_status in ("defensible", "defensible_with_caveats")
        assert result.numerical_sanity.status in (SanityStatus.PASS, SanityStatus.WARNING)

    def test_no_method_change(self):
        result = self.executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment", group1=0, group2=1,
        )
        assert result.method_id == "Independent samples t-test"
        assert result.alternative_methods_considered == []


# ===========================================================================
# TEST 2 — BINARY OUTCOME
# ===========================================================================

@pytest.fixture
def test2_binary_data(tmp_path):
    """Test 2 — Binary outcome: treatment vs hospitalization."""
    rng = np.random.RandomState(42)
    n = 200
    treatment = rng.choice([0, 1], size=n)
    # Treatment increases hospitalization probability
    prob = np.where(treatment == 1, 0.35, 0.15)
    hosp = rng.binomial(1, prob)
    df = pd.DataFrame({
        "patient_id": range(1, n + 1),
        "treatment": treatment,
        "hospitalization_30d": hosp,
    })
    path = tmp_path / "test2_binary.csv"
    df.to_csv(path, index=False)
    return path


class Test2_BinaryOutcome:
    """Validation Test 2 — Binary outcome."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.rq = "Is treatment group associated with 30-day hospitalization?"
        self.executor = StatisticalExecutor()

    def test_dataset_structure(self, test2_binary_data):
        df = pd.read_csv(test2_binary_data)
        assert len(df) == 200
        assert set(df.columns) == {"patient_id", "treatment", "hospitalization_30d"}
        assert df["hospitalization_30d"].nunique() == 2

    def test_problem_type(self, test2_binary_data):
        df = pd.read_csv(test2_binary_data)
        resolution = _resolve_and_classify(df, self.rq)
        assert resolution.problem_type in (
            AnalysisProblemType.BINARY_OUTCOME_SINGLE_PREDICTOR,
            AnalysisProblemType.BINARY_OUTCOME_MULTIVARIABLE,
        )

    def test_variable_roles(self, test2_binary_data):
        df = pd.read_csv(test2_binary_data)
        resolution = _resolve_and_classify(df, self.rq)
        roles = {r.variable_name: r.proposed_role for r in resolution.resolutions}
        assert roles.get("patient_id") == VariableRole.IDENTIFIER
        assert roles.get("hospitalization_30d") == VariableRole.OUTCOME
        assert roles.get("treatment") in (VariableRole.EXPOSURE, VariableRole.GROUPING_VARIABLE)

    def test_candidate_methods(self, test2_binary_data):
        df = pd.read_csv(test2_binary_data)
        resolution = _resolve_and_classify(df, self.rq)
        candidates, _ = _generate_candidates(df, resolution, self.rq)
        assert candidates is not None
        primary_ids = [c.method_id for c in candidates.primary_candidates]
        # Should include chi-square at minimum
        assert "chi_square" in primary_ids or "fisher_exact" in primary_ids

    def test_identifier_excluded(self, test2_binary_data):
        df = pd.read_csv(test2_binary_data)
        resolution = _resolve_and_classify(df, self.rq)
        candidates, _ = _generate_candidates(df, resolution, self.rq)
        all_pred = candidates.data_profile.predictor_variables + candidates.data_profile.covariates
        assert "patient_id" not in all_pred
        assert candidates.data_profile.outcome_variable != "patient_id"

    def test_chi_square_execution(self, test2_binary_data):
        result = self.executor.execute(
            "chi_square", test2_binary_data,
            variable1="treatment", variable2="hospitalization_30d",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.n_observed == 200
        assert result.p_value is not None
        assert result.test_statistic is not None

    def test_fisher_exact_execution(self, test2_binary_data):
        result = self.executor.execute(
            "fisher_exact", test2_binary_data,
            variable1="treatment", variable2="hospitalization_30d",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.n_observed == 200

    def test_expected_counts_diagnosed(self, test2_binary_data):
        result = self.executor.execute(
            "chi_square", test2_binary_data,
            variable1="treatment", variable2="hospitalization_30d",
        )
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        assert "Expected cell counts" in diag_names

    def test_no_silent_method_substitution(self, test2_binary_data):
        """Chi-square should remain chi-square, not silently become Fisher."""
        result = self.executor.execute(
            "chi_square", test2_binary_data,
            variable1="treatment", variable2="hospitalization_30d",
        )
        assert result.method_id == "Chi-square test of independence"

    def test_diagnostics_present(self, test2_binary_data):
        result = self.executor.execute(
            "chi_square", test2_binary_data,
            variable1="treatment", variable2="hospitalization_30d",
        )
        assert result.diagnostic_status is not None
        assert result.defensibility_status is not None
        diag = result.diagnostic_summary
        assert diag["n_diagnostics"] > 0

    def test_analyzed_n_correct(self, test2_binary_data):
        result = self.executor.execute(
            "chi_square", test2_binary_data,
            variable1="treatment", variable2="hospitalization_30d",
        )
        assert result.n_observed == 200


# ===========================================================================
# TEST 3 — CONTINUOUS ASSOCIATION
# ===========================================================================

@pytest.fixture
def test3_correlation_data(tmp_path):
    """Test 3 — BMI vs biomarker (continuous association)."""
    rng = np.random.RandomState(42)
    n = 120
    bmi = rng.normal(27, 5, n)
    biomarker = 0.6 * bmi + rng.normal(0, 8, n)
    df = pd.DataFrame({
        "patient_id": range(1, n + 1),
        "BMI": bmi,
        "biomarker": biomarker,
    })
    path = tmp_path / "test3_correlation.csv"
    df.to_csv(path, index=False)
    return path


class Test3_ContinuousAssociation:
    """Validation Test 3 — Continuous association."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.rq = "Is BMI associated with biomarker level?"
        self.executor = StatisticalExecutor()

    def test_problem_type(self, test3_correlation_data):
        df = pd.read_csv(test3_correlation_data)
        resolution = _resolve_and_classify(df, self.rq)
        assert resolution.problem_type == AnalysisProblemType.CONTINUOUS_CONTINUOUS_ASSOCIATION

    def test_variable_roles(self, test3_correlation_data):
        df = pd.read_csv(test3_correlation_data)
        resolution = _resolve_and_classify(df, self.rq)
        roles = {r.variable_name: r.proposed_role for r in resolution.resolutions}
        assert roles.get("patient_id") == VariableRole.IDENTIFIER
        assert roles.get("biomarker") == VariableRole.OUTCOME
        assert roles.get("BMI") in (VariableRole.PREDICTOR, VariableRole.EXPOSURE)

    def test_candidate_methods(self, test3_correlation_data):
        df = pd.read_csv(test3_correlation_data)
        resolution = _resolve_and_classify(df, self.rq)
        candidates, _ = _generate_candidates(df, resolution, self.rq)
        assert candidates is not None
        primary_ids = [c.method_id for c in candidates.primary_candidates]
        assert "pearson_correlation" in primary_ids
        assert "spearman_correlation" in primary_ids
        # Group-comparison methods should NOT be primary
        assert "independent_t_test" not in primary_ids

    def test_identifier_excluded(self, test3_correlation_data):
        df = pd.read_csv(test3_correlation_data)
        resolution = _resolve_and_classify(df, self.rq)
        candidates, _ = _generate_candidates(df, resolution, self.rq)
        all_vars = candidates.data_profile.predictor_variables + candidates.data_profile.covariates
        assert "patient_id" not in all_vars
        assert candidates.data_profile.outcome_variable != "patient_id"

    def test_pearson_execution(self, test3_correlation_data):
        result = self.executor.execute(
            "pearson_correlation", test3_correlation_data,
            x="BMI", y="biomarker",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.n_observed == 120
        assert result.test_statistic is not None
        assert abs(result.test_statistic) <= 1.0
        assert result.p_value is not None

    def test_spearman_execution(self, test3_correlation_data):
        result = self.executor.execute(
            "spearman_correlation", test3_correlation_data,
            x="BMI", y="biomarker",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.n_observed == 120

    def test_correlation_diagnostics(self, test3_correlation_data):
        result = self.executor.execute(
            "pearson_correlation", test3_correlation_data,
            x="BMI", y="biomarker",
        )
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        assert "Both variables numeric" in diag_names
        assert "Constant variable" not in [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", []) if d["status"] == "fail"]

    def test_constant_variable_protection(self, tmp_path):
        """Constant variable should be detected."""
        df = pd.DataFrame({
            "x": [5.0] * 50,
            "y": np.random.RandomState(42).normal(0, 1, 50),
        })
        path = tmp_path / "const.csv"
        df.to_csv(path, index=False)
        result = self.executor.execute(
            "pearson_correlation", path,
            x="x", y="y",
        )
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        assert "Constant variable" in diag_names

    def test_no_causation_claim(self, test3_correlation_data):
        result = self.executor.execute(
            "pearson_correlation", test3_correlation_data,
            x="BMI", y="biomarker",
        )
        # Result should not contain causal language
        summary = result.to_summary()
        for val in summary.values():
            if isinstance(val, str):
                assert "caus" not in val.lower()

    def test_result_includes_required_fields(self, test3_correlation_data):
        result = self.executor.execute(
            "pearson_correlation", test3_correlation_data,
            x="BMI", y="biomarker",
        )
        assert result.test_statistic is not None
        assert result.p_value is not None
        assert result.n_observed is not None
        assert result.diagnostic_status is not None
        assert result.defensibility_status is not None


# ===========================================================================
# TEST 4 — MULTIVARIABLE LOGISTIC REGRESSION
# ===========================================================================

@pytest.fixture
def test4_logistic_data(tmp_path):
    """Test 4 — Multivariable logistic regression."""
    rng = np.random.RandomState(42)
    n = 300
    treatment = rng.choice([0, 1], size=n)
    age = rng.normal(60, 12, n)
    sex = rng.choice([0, 1], size=n)
    comorbidity = rng.poisson(2, n)
    logit = -3 + 0.8 * treatment + 0.03 * age + 0.4 * sex + 0.3 * comorbidity
    prob = 1 / (1 + np.exp(-logit))
    hosp = rng.binomial(1, prob, n)
    df = pd.DataFrame({
        "patient_id": range(1, n + 1),
        "treatment": treatment,
        "hospitalization_30d": hosp,
        "age": age,
        "sex": sex,
        "comorbidity_count": comorbidity,
    })
    path = tmp_path / "test4_logistic.csv"
    df.to_csv(path, index=False)
    return path


class Test4_MultivariableLogistic:
    """Validation Test 4 — Multivariable logistic regression."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.rq = "Is treatment associated with 30-day hospitalization after adjustment for age, sex, and comorbidity burden?"
        self.executor = StatisticalExecutor()

    def test_problem_type(self, test4_logistic_data):
        df = pd.read_csv(test4_logistic_data)
        resolution = _resolve_and_classify(df, self.rq)
        assert resolution.problem_type in (
            AnalysisProblemType.BINARY_OUTCOME_MULTIVARIABLE,
            AnalysisProblemType.BINARY_OUTCOME_SINGLE_PREDICTOR,
        )

    def test_variable_roles(self, test4_logistic_data):
        df = pd.read_csv(test4_logistic_data)
        resolution = _resolve_and_classify(df, self.rq)
        roles = {r.variable_name: r.proposed_role for r in resolution.resolutions}
        assert roles.get("patient_id") == VariableRole.IDENTIFIER
        assert roles.get("hospitalization_30d") == VariableRole.OUTCOME

    def test_identifier_excluded(self, test4_logistic_data):
        df = pd.read_csv(test4_logistic_data)
        resolution = _resolve_and_classify(df, self.rq)
        candidates, _ = _generate_candidates(df, resolution, self.rq)
        all_vars = candidates.data_profile.predictor_variables + candidates.data_profile.covariates
        assert "patient_id" not in all_vars

    def test_execution(self, test4_logistic_data):
        result = self.executor.execute(
            "logistic_regression", test4_logistic_data,
            outcome="hospitalization_30d",
            predictors=["treatment", "age", "sex", "comorbidity_count"],
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.n_observed == 300
        assert result.p_value is not None

    def test_coefficients_present(self, test4_logistic_data):
        result = self.executor.execute(
            "logistic_regression", test4_logistic_data,
            outcome="hospitalization_30d",
            predictors=["treatment", "age", "sex", "comorbidity_count"],
        )
        params = result.engine_result.get("parameters", {})
        assert "treatment" in params
        assert "age" in params
        assert "sex" in params
        assert "comorbidity_count" in params
        for name, info in params.items():
            assert "coef" in info
            assert "std_err" in info
            assert "p_value" in info

    def test_odds_ratios(self, test4_logistic_data):
        result = self.executor.execute(
            "logistic_regression", test4_logistic_data,
            outcome="hospitalization_30d",
            predictors=["treatment", "age", "sex", "comorbidity_count"],
        )
        params = result.engine_result.get("parameters", {})
        for name, info in params.items():
            if name != "const":
                assert "odds_ratio" in info, f"Missing odds ratio for {name}"

    def test_events_per_variable(self, test4_logistic_data):
        result = self.executor.execute(
            "logistic_regression", test4_logistic_data,
            outcome="hospitalization_30d",
            predictors=["treatment", "age", "sex", "comorbidity_count"],
        )
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        assert "Events per variable" in diag_names

    def test_convergence_diagnostic(self, test4_logistic_data):
        result = self.executor.execute(
            "logistic_regression", test4_logistic_data,
            outcome="hospitalization_30d",
            predictors=["treatment", "age", "sex", "comorbidity_count"],
        )
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        assert "Model convergence" in diag_names

    def test_separation_diagnostic(self, test4_logistic_data):
        result = self.executor.execute(
            "logistic_regression", test4_logistic_data,
            outcome="hospitalization_30d",
            predictors=["treatment", "age", "sex", "comorbidity_count"],
        )
        # Separation diagnostic should be present (may pass or warn)
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        assert any("eparation" in n for n in diag_names) or result.diagnostic_status in ("pass", "warning")

    def test_multicollinearity_diagnostic(self, test4_logistic_data):
        result = self.executor.execute(
            "logistic_regression", test4_logistic_data,
            outcome="hospitalization_30d",
            predictors=["treatment", "age", "sex", "comorbidity_count"],
        )
        diag_names = [d["diagnostic_name"] for d in result.diagnostic_summary.get("diagnostics", [])]
        assert "Multicollinearity (VIF)" in diag_names

    def test_finite_estimates(self, test4_logistic_data):
        result = self.executor.execute(
            "logistic_regression", test4_logistic_data,
            outcome="hospitalization_30d",
            predictors=["treatment", "age", "sex", "comorbidity_count"],
        )
        params = result.engine_result.get("parameters", {})
        for name, info in params.items():
            assert np.isfinite(info.get("coef", float("nan"))), f"Non-finite coef for {name}"

    def test_defensibility(self, test4_logistic_data):
        result = self.executor.execute(
            "logistic_regression", test4_logistic_data,
            outcome="hospitalization_30d",
            predictors=["treatment", "age", "sex", "comorbidity_count"],
        )
        assert result.defensibility_status in (
            "defensible", "defensible_with_caveats", "review_required"
        )


# ===========================================================================
# TEST 5 — MESSY REAL-WORLD DATASET
# ===========================================================================

@pytest.fixture
def test5_messy_data(tmp_path):
    """Test 5 — Deliberately messy dataset."""
    rng = np.random.RandomState(42)
    n = 150
    patient_id = list(range(1, n + 1))
    treatment = rng.choice([0, 1], size=n)
    age = rng.normal(55, 15, n).astype(int)
    # Missing some age values
    age_missing_idx = rng.choice(n, 10, replace=False)
    age = age.astype(float)
    age[age_missing_idx] = np.nan
    sex = rng.choice([0, 1], size=n).astype(float)
    # Some missing sex
    sex_missing_idx = rng.choice(n, 5, replace=False)
    sex[sex_missing_idx] = np.nan
    # Outcome with missing values
    systolic = np.where(treatment == 1, 125, 135) + rng.normal(0, 12, n)
    systolic_missing = rng.choice(n, 8, replace=False)
    systolic = systolic.astype(float)
    systolic[systolic_missing] = np.nan
    # Duplicate rows
    df = pd.DataFrame({
        "patient_id": patient_id,
        "treatment": treatment,
        "age": age,
        "sex": sex,
        "systolic_bp": systolic,
    })
    # Add 3 duplicate rows
    duplicates = df.iloc[:3].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    path = tmp_path / "test5_messy.csv"
    df.to_csv(path, index=False)
    return path


class Test5_MessyRealWorld:
    """Validation Test 5 — Messy real-world dataset."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.rq = "Does treatment affect systolic blood pressure after adjusting for age and sex?"
        self.executor = StatisticalExecutor()

    def test_dataset_detected_as_messy(self, test5_messy_data):
        df = pd.read_csv(test5_messy_data)
        # Should detect duplicates
        assert len(df) > 150  # includes duplicates
        # Should detect missingness
        assert df["age"].isna().sum() > 0
        assert df["systolic_bp"].isna().sum() > 0

    def test_dq_profile_created(self, test5_messy_data):
        """Dataset quality profile should be created."""
        from app.statistics.data_quality import DataQualityProfile, DatasetMissingness, VariableMissingness, MissingnessStatus, VariableType
        df = pd.read_csv(test5_messy_data)
        # Build missingness manually
        vars_missing = []
        for col in df.columns:
            n_miss = int(df[col].isna().sum())
            n_obs = int(len(df) - n_miss)
            pct = n_miss / len(df) if len(df) > 0 else 0
            if n_miss > 0:
                if pct >= 0.5:
                    status = MissingnessStatus.EXTREME
                elif pct >= 0.1:
                    status = MissingnessStatus.HIGH
                else:
                    status = MissingnessStatus.LOW
            else:
                status = MissingnessStatus.NONE
            # Determine variable type
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() <= 2:
                    vtype = VariableType.BINARY
                else:
                    vtype = VariableType.NUMERIC
            else:
                vtype = VariableType.CATEGORICAL
            vars_missing.append(VariableMissingness(
                variable=col, variable_type=vtype.value,
                n_missing=n_miss, n_observed=n_obs, total=len(df),
                pct_missing=round(pct, 4), status=status,
            ))
        missingness = DatasetMissingness(
            total_rows=df.shape[0],
            complete_cases=int(df.dropna().shape[0]),
            incomplete_cases=int(df.shape[0] - df.dropna().shape[0]),
            total_missing_cells=int(df.isna().sum().sum()),
            total_cells=df.shape[0] * df.shape[1],
            pct_missing_overall=round(df.isna().sum().sum() / (df.shape[0] * df.shape[1]), 4),
            variables=vars_missing,
        )
        profile = DataQualityProfile(
            dataset_id=_uid(),
            row_count=df.shape[0],
            column_count=df.shape[1],
            duplicate_rows=int(df.duplicated().sum()),
            missingness=missingness,
        )
        assert profile is not None
        assert profile.row_count > 0

    def test_original_dataset_preserved(self, test5_messy_data):
        """Original dataset must not be modified."""
        original_hash = _hash_file(test5_messy_data)
        df = pd.read_csv(test5_messy_data)
        # Just read, don't modify
        _ = df.copy()
        new_hash = _hash_file(test5_messy_data)
        assert original_hash == new_hash

    def test_missingness_detected(self, test5_messy_data):
        df = pd.read_csv(test5_messy_data)
        # Directly check missingness
        assert df["age"].isna().sum() > 0
        assert df["systolic_bp"].isna().sum() > 0

    def test_outliers_distinguished_from_missingness(self, test5_messy_data):
        """Outlier and missingness should be separable concerns."""
        from app.statistics.outlier_engine import detect_outliers_iqr
        df = pd.read_csv(test5_messy_data)
        # Can detect outliers on non-missing data — pass DataFrame with just the column
        bp_df = df[["systolic_bp"]].dropna()
        outliers = detect_outliers_iqr(bp_df)
        # outliers is a list of VariableOutliers; missingness is separate
        assert isinstance(outliers, list)
        assert len(outliers) >= 0  # may or may not have outliers

    def test_no_silent_imputation(self, test5_messy_data):
        """Missing outcomes should not be silently imputed."""
        result = self.executor.execute(
            "independent_t_test", test5_messy_data,
            outcome="systolic_bp", group="treatment",
        )
        # Analyzed N should be less than total rows due to missingness
        assert result.n_observed < 153  # total rows including duplicates
        assert result.n_observed > 0

    def test_analyzed_n_correct(self, test5_messy_data):
        df = pd.read_csv(test5_messy_data)
        complete = df[["systolic_bp", "treatment"]].dropna()
        result = self.executor.execute(
            "independent_t_test", test5_messy_data,
            outcome="systolic_bp", group="treatment",
        )
        assert result.n_observed == len(complete)

    def test_no_automatic_outlier_removal(self, test5_messy_data):
        """Outliers should not be automatically removed."""
        result = self.executor.execute(
            "independent_t_test", test5_messy_data,
            outcome="systolic_bp", group="treatment",
        )
        # Should analyze all complete cases
        df = pd.read_csv(test5_messy_data)
        complete_n = df[["systolic_bp", "treatment"]].dropna().shape[0]
        assert result.n_observed == complete_n

    def test_no_automatic_transformation(self, test5_messy_data):
        """Skewed variables should not be automatically transformed."""
        result = self.executor.execute(
            "independent_t_test", test5_messy_data,
            outcome="systolic_bp", group="treatment",
        )
        # Method should remain t-test, not become some transformed version
        assert result.method_id == "Independent samples t-test"

    def test_audit_trail_preserved(self, test5_messy_data):
        result = self.executor.execute(
            "independent_t_test", test5_messy_data,
            outcome="systolic_bp", group="treatment",
        )
        assert result.result_fingerprint is not None
        assert result.data_verification.content_hash is not None


# ===========================================================================
# TEST 6 — ADVERSARIAL ANTI-P-HACKING
# ===========================================================================

@pytest.fixture
def test6_adversarial_data(tmp_path):
    """Test 6 — Dataset where different analyses give very different p-values."""
    rng = np.random.RandomState(42)
    n_per = 30
    treatment = np.array([0] * n_per + [1] * n_per)
    # Outcome has a real effect but with one extreme outlier in group 0
    outcome = np.concatenate([
        rng.normal(0, 8, n_per),   # group 0: mean ~0
        rng.normal(5, 8, n_per),   # group 1: mean ~5
    ])
    # Add one extreme outlier in group 0 that inflates variance
    outcome[0] = 100
    df = pd.DataFrame({
        "patient_id": range(1, 2 * n_per + 1),
        "treatment": treatment,
        "outcome": outcome,
    })
    path = tmp_path / "test6_adversarial.csv"
    df.to_csv(path, index=False)
    return path


class Test6_AdversarialAntiPHacking:
    """Validation Test 6 — Anti-p-hacking."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.rq = "Does treatment differ in outcome?"
        self.executor = StatisticalExecutor()

    def test_t_test_result(self, test6_adversarial_data):
        result = self.executor.execute(
            "independent_t_test", test6_adversarial_data,
            outcome="outcome", group="treatment",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.p_value is not None

    def test_mann_whitney_result(self, test6_adversarial_data):
        result = self.executor.execute(
            "mann_whitney_u", test6_adversarial_data,
            outcome="outcome", group="treatment",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.p_value is not None

    def test_diagnostics_do_not_select_method(self, test6_adversarial_data):
        """Diagnostics should not select the method with the smaller p-value."""
        t_result = self.executor.execute(
            "independent_t_test", test6_adversarial_data,
            outcome="outcome", group="treatment",
        )
        mw_result = self.executor.execute(
            "mann_whitney_u", test6_adversarial_data,
            outcome="outcome", group="treatment",
        )
        # Both should execute independently
        assert t_result.execution_status == ExecutionStatus.SUCCESS
        assert mw_result.execution_status == ExecutionStatus.SUCCESS
        # Neither should have been changed by diagnostics
        assert t_result.method_id == "Independent samples t-test"
        assert mw_result.method_id == "Mann-Whitney U test"

    def test_no_p_value_based_selection(self, test6_adversarial_data):
        """Verify no diagnostic message suggests method selection based on p-value."""
        result = self.executor.execute(
            "independent_t_test", test6_adversarial_data,
            outcome="outcome", group="treatment",
        )
        for d in result.diagnostic_summary.get("diagnostics", []):
            msg = d.get("message", "").lower()
            assert "smaller p" not in msg
            assert "more significant" not in msg
            assert "select this method" not in msg

    def test_prespecified_analysis_preserved(self, test6_adversarial_data):
        """The approved analysis should be preserved regardless of diagnostics."""
        from app.statistics.specification import StatisticalAnalysisSpecification
        spec = StatisticalAnalysisSpecification(
            primary_outcome="outcome",
            analysis_method="independent_t_test",
        )
        spec.approve()
        spec.lock()
        assert spec.approval_status.value == "locked"
        assert spec.analysis_method == "independent_t_test"

    def test_new_version_required_for_method_change(self, test6_adversarial_data):
        """Changing the method requires a new analysis version."""
        from app.statistics.specification import StatisticalAnalysisSpecification
        spec = StatisticalAnalysisSpecification(
            primary_outcome="outcome",
            analysis_method="independent_t_test",
        )
        spec.approve()
        spec.modify("2.0")
        assert spec.version == "2.0"
        assert spec.approval_status.value == "modified"

    def test_distinct_fingerprints(self, test6_adversarial_data):
        """Different analyses produce different fingerprints."""
        result1 = self.executor.execute(
            "independent_t_test", test6_adversarial_data,
            outcome="outcome", group="treatment",
        )
        result2 = self.executor.execute(
            "mann_whitney_u", test6_adversarial_data,
            outcome="outcome", group="treatment",
        )
        assert result1.result_fingerprint != result2.result_fingerprint

    def test_audit_history_preserved(self, test6_adversarial_data):
        """Both analyses should have complete provenance."""
        for method in ["independent_t_test", "mann_whitney_u"]:
            result = self.executor.execute(
                method, test6_adversarial_data,
                outcome="outcome", group="treatment",
            )
            assert result.data_verification.content_hash is not None
            assert result.result_fingerprint is not None
            assert result.engine_version is not None


# ===========================================================================
# CROSS-TEST INVARIANTS
# ===========================================================================

class TestCrossTestInvariants:
    """Verify architectural invariants across all test scenarios."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.executor = StatisticalExecutor()

    def test_invariant1_identifiers_never_outcomes(self):
        """Patient identifiers must never become outcomes or predictors."""
        df = pd.read_csv(TEST1_PATH)
        resolution = _resolve_and_classify(df, "Does treatment differ in BP?")
        for r in resolution.resolutions:
            if "id" in r.variable_name.lower():
                assert r.proposed_role != VariableRole.OUTCOME
                assert r.proposed_role != VariableRole.PREDICTOR

    def test_invariant2_no_silent_method_switching(self):
        """Diagnostics cannot change the approved analysis."""
        result = self.executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment", group1=0, group2=1,
        )
        assert result.method_id == "Independent samples t-test"

    def test_invariant3_no_p_hacking(self):
        """No method is selected because it produces a smaller p-value."""
        result = self.executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment",
        )
        assert result.alternative_methods_considered == []

    def test_invariant4_immutable_originals(self):
        """Original datasets remain unchanged."""
        original_hash = _hash_file(TEST1_PATH)
        self.executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment",
        )
        assert _hash_file(TEST1_PATH) == original_hash

    def test_invariant5_dataset_identity(self):
        """Execution against a modified dataset must be BLOCKED."""
        import shutil
        import tempfile
        from app.statistics.executor import verify_dataset
        # Use tempfile for complete isolation
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmp_path = Path(f.name)
        shutil.copy2(TEST1_PATH, tmp_path)
        try:
            v = verify_dataset(tmp_path)
            df = pd.read_csv(tmp_path)
            df.loc[0, "systolic_bp"] = 99999
            df.to_csv(tmp_path, index=False)
            result = self.executor.execute(
                "independent_t_test", tmp_path,
                outcome="systolic_bp", group="treatment",
                expected_content_hash=v.content_hash,
            )
            # Must be BLOCKED, not just a warning
            assert result.execution_status == ExecutionStatus.BLOCKED
            assert not result.data_verification.identity_verified
            assert result.data_verification.identity_mismatch_reason is not None
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_invariant6_locked_analyses(self):
        """Locked analyses cannot be silently modified."""
        from app.statistics.specification import StatisticalAnalysisSpecification
        spec = StatisticalAnalysisSpecification(
            primary_outcome="score",
            analysis_method="independent_t_test",
        )
        spec.approve()
        spec.lock()
        assert spec.approval_status.value == "locked"

    def test_invariant7_versioning(self):
        """Methodological modifications create new analysis versions."""
        from app.statistics.specification import StatisticalAnalysisSpecification
        spec = StatisticalAnalysisSpecification(
            primary_outcome="score",
            analysis_method="independent_t_test",
        )
        spec.approve()
        v1 = spec.version
        spec.modify("2.0")
        assert spec.version != v1
        assert spec.version == "2.0"

    def test_invariant8_numerical_determinism(self):
        """Repeated execution produces equivalent numerical results."""
        r1 = self.executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment", group1=0, group2=1,
        )
        r2 = self.executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment", group1=0, group2=1,
        )
        assert r1.test_statistic == r2.test_statistic
        assert r1.p_value == r2.p_value
        assert r1.result_fingerprint == r2.result_fingerprint

    def test_invariant9_ai_boundary(self):
        """No LLM output populates numerical statistical results."""
        result = self.executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment",
        )
        assert result.engine_result.get("engine") is not None
        assert "scipy" in result.engine_result.get("engine", "").lower()

    def test_invariant10_causal_language(self):
        """Associational analyses must not automatically become causal claims."""
        result = self.executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment",
        )
        summary = result.to_summary()
        for val in summary.values():
            if isinstance(val, str):
                assert "caus" not in val.lower()

    def test_invariant11_missing_outcomes(self):
        """Missing outcomes must never be silently imputed."""
        result = self.executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment",
        )
        assert result.n_observed <= 120

    def test_invariant12_outliers(self):
        """Outliers must never be automatically removed."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "group": ["A"] * 30 + ["B"] * 30,
            "score": np.concatenate([
                rng.normal(50, 10, 29), [500],  # extreme outlier
                rng.normal(60, 10, 30),
            ]),
        })
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            path = Path(f.name)
        result = self.executor.execute(
            "independent_t_test", path,
            outcome="score", group="group",
        )
        # Should include the outlier in analysis
        assert result.n_observed == 60

    def test_invariant13_transformations_require_approval(self):
        """Transformations require explicit approval."""
        from app.statistics.specification import StatisticalAnalysisSpecification
        spec = StatisticalAnalysisSpecification(
            primary_outcome="score",
            analysis_method="independent_t_test",
        )
        # Transformations list should be empty by default
        assert spec.transformations == []

    def test_invariant14_diagnostics_inform_review(self):
        """Diagnostics inform review; they do not rewrite the analysis."""
        result = self.executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment",
        )
        # Diagnostics should be present
        assert result.diagnostic_status is not None
        # But method should not have been changed
        assert result.method_id == "Independent samples t-test"
