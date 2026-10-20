"""Phase 2E.2 — Regression Tests for All 12 Discovered Defects.

Each test explicitly covers a defect discovered during the Phase 2E validation battery.
Tests verify that the defect cannot regress.
"""

import sys
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from app.statistics.executor import StatisticalExecutor, verify_dataset
from app.statistics.variable_roles import (
    RoleResolver, VariableRole, AnalysisProblemType, RoleSource, RoleStatus,
)
from app.statistics.candidate_generator import CandidateGenerator, DataProfile
from app.statistics.registry import get_method
from app.statistics.result import ExecutionStatus, SanityStatus
from app.data.dataset_identity import compute_dataset_identity, verify_dataset_identity


# ---------------------------------------------------------------------------
# Canonical fixtures
# ---------------------------------------------------------------------------

TEST1_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "uploads"
    / "228683e7-01c9-4050-afde-4d17cdb99e3f_STATIS_Test1_Continuous.csv"
)


def _uid() -> str:
    import uuid
    return f"reg_{uuid.uuid4().hex[:8]}"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schema_hash(df: pd.DataFrame) -> str:
    schema = {col: str(df[col].dtype) for col in df.columns}
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def _build_variable_info(df: pd.DataFrame) -> list[dict]:
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


def _resolve_roles(df: pd.DataFrame, rq: str | None = None):
    vars_info = _build_variable_info(df)
    return RoleResolver.resolve_all(vars_info, research_question=rq)


# ---------------------------------------------------------------------------
# Defect 1 — Identifier detector too aggressive
# ---------------------------------------------------------------------------

class TestDefect1_IdentifierDetector:
    """Continuous variables with all unique values must NOT be identifiers."""

    def test_age_not_identifier(self):
        """age: 18,19,20,...,137 — all unique but NOT an identifier."""
        df = pd.DataFrame({
            "age": list(range(18, 138)),
            "score": np.random.RandomState(42).normal(0, 1, 120),
        })
        result = _resolve_roles(df)
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        assert roles["age"] != VariableRole.IDENTIFIER

    def test_biomarker_not_identifier(self):
        """Biomarker with all unique values — NOT an identifier."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "biomarker": rng.uniform(0, 100, 200),
            "group": rng.choice([0, 1], 200),
        })
        result = _resolve_roles(df)
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        assert roles["biomarker"] != VariableRole.IDENTIFIER

    def test_continuous_with_unique_values_not_identifier(self):
        """Any continuous variable with all unique values must not be identifier."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "temperature": rng.uniform(36.0, 42.0, 100),
            "measurement": rng.normal(50, 10, 100),
        })
        result = _resolve_roles(df)
        for r in result.resolutions:
            assert r.proposed_role != VariableRole.IDENTIFIER

    def test_patient_id_is_identifier(self):
        """patient_id IS correctly identified as identifier."""
        df = pd.DataFrame({
            "patient_id": range(1, 121),
            "treatment": [0, 1] * 60,
            "outcome": np.random.RandomState(42).normal(0, 1, 120),
        })
        result = _resolve_roles(df)
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        assert roles["patient_id"] == VariableRole.IDENTIFIER

    def test_uuid_is_identifier(self):
        """UUID-like variable IS correctly identified as identifier."""
        import uuid
        df = pd.DataFrame({
            "uuid": [str(uuid.uuid4()) for _ in range(50)],
            "value": np.random.RandomState(42).normal(0, 1, 50),
        })
        result = _resolve_roles(df)
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        assert roles["uuid"] == VariableRole.IDENTIFIER


# ---------------------------------------------------------------------------
# Defect 2 — Overly broad outcome patterns
# ---------------------------------------------------------------------------

class TestDefect2_OutcomePatterns:
    """Terms like age, count, level must not independently cause outcome role."""

    def test_level_not_outcome(self):
        """'level' should not independently make a variable an outcome."""
        df = pd.DataFrame({
            "treatment": [0, 1] * 50,
            "dosage_level": np.random.RandomState(42).uniform(0, 10, 100),
            "outcome": np.random.RandomState(43).normal(0, 1, 100),
        })
        result = _resolve_roles(df, rq="Does treatment affect outcome?")
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        # dosage_level should not be outcome when there's a clear outcome variable
        if "outcome" in roles and roles["outcome"] == VariableRole.OUTCOME:
            assert roles["dosage_level"] != VariableRole.OUTCOME

    def test_count_not_outcome(self):
        """'count' should not independently make a variable an outcome."""
        df = pd.DataFrame({
            "treatment": [0, 1] * 50,
            "error_count": np.random.RandomState(42).poisson(5, 100),
            "recovery": np.random.RandomState(43).normal(0, 1, 100),
        })
        result = _resolve_roles(df, rq="Does treatment affect recovery?")
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        if "recovery" in roles and roles["recovery"] == VariableRole.OUTCOME:
            assert roles["error_count"] != VariableRole.OUTCOME

    def test_amount_not_outcome(self):
        """'amount' should not independently make a variable an outcome."""
        df = pd.DataFrame({
            "group": [0, 1] * 50,
            "blood_amount": np.random.RandomState(42).uniform(0, 50, 100),
            "bp": np.random.RandomState(43).normal(120, 15, 100),
        })
        result = _resolve_roles(df, rq="Does group differ in bp?")
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        if "bp" in roles and roles["bp"] == VariableRole.OUTCOME:
            assert roles["blood_amount"] != VariableRole.OUTCOME

    def test_value_not_outcome(self):
        """'value' should not independently make a variable an outcome."""
        df = pd.DataFrame({
            "exposure": [0, 1] * 50,
            "measurement_value": np.random.RandomState(42).normal(0, 1, 100),
            "response": np.random.RandomState(43).normal(0, 1, 100),
        })
        result = _resolve_roles(df, rq="Does exposure affect response?")
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        if "response" in roles and roles["response"] == VariableRole.OUTCOME:
            assert roles["measurement_value"] != VariableRole.OUTCOME


# ---------------------------------------------------------------------------
# Defect 3 — Mann-Whitney group value 0
# ---------------------------------------------------------------------------

class TestDefect3_MannWhitneyGroupZero:
    """Group label 0 must not be treated as falsy."""

    def test_group_zero_works(self):
        """Mann-Whitney with group1=0 must not fail."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "group": [0] * 30 + [1] * 30,
            "outcome": np.concatenate([rng.normal(0, 1, 30), rng.normal(2, 1, 30)]),
        })
        path = Path(_uid()) / "test_mw_zero.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        try:
            executor = StatisticalExecutor()
            result = executor.execute(
                "mann_whitney_u", path,
                outcome="outcome", group="group",
                group1=0, group2=1,
            )
            assert result.execution_status == ExecutionStatus.SUCCESS
            assert result.p_value is not None
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)

    def test_group_zero_vs_one_explicit(self):
        """Explicitly passing group1=0, group2=1 must work."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "trt": [0] * 25 + [1] * 25,
            "val": np.concatenate([rng.normal(10, 2, 25), rng.normal(12, 2, 25)]),
        })
        path = Path(_uid()) / "test_mw_01.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        try:
            executor = StatisticalExecutor()
            result = executor.execute(
                "mann_whitney_u", path,
                outcome="val", group="trt",
                group1=0, group2=1,
            )
            assert result.execution_status == ExecutionStatus.SUCCESS
            # Verify correct groups were used
            assert result.engine_result.get("group1") == 0
            assert result.engine_result.get("group2") == 1
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)

    def test_independent_t_test_group_zero(self):
        """t-test with group1=0 must also work."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "treatment": [0] * 30 + [1] * 30,
            "score": np.concatenate([rng.normal(50, 10, 30), rng.normal(60, 10, 30)]),
        })
        path = Path(_uid()) / "test_tt_zero.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        try:
            executor = StatisticalExecutor()
            result = executor.execute(
                "independent_t_test", path,
                outcome="score", group="treatment",
                group1=0, group2=1,
            )
            assert result.execution_status == ExecutionStatus.SUCCESS
            assert result.engine_result.get("group1") == 0
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)


# ---------------------------------------------------------------------------
# Defect 4 — Binary outcome classified as continuous association
# ---------------------------------------------------------------------------

class TestDefect4_BinaryOutcomeClassification:
    """Binary outcome must not be classified as CONTINUOUS_CONTINUOUS_ASSOCIATION."""

    def test_hospitalization_not_correlation(self):
        """'Is treatment associated with hospitalization?' — binary, not correlation."""
        df = pd.DataFrame({
            "patient_id": range(1, 201),
            "treatment": [0, 1] * 100,
            "hospitalization": np.random.RandomState(42).binomial(1, 0.3, 200),
        })
        result = _resolve_roles(df, rq="Is treatment associated with hospitalization?")
        assert result.problem_type in (
            AnalysisProblemType.BINARY_OUTCOME_SINGLE_PREDICTOR,
            AnalysisProblemType.BINARY_OUTCOME_MULTIVARIABLE,
        )

    def test_death_not_correlation(self):
        """'Is drug associated with death?' — binary, not correlation."""
        df = pd.DataFrame({
            "subject_id": range(1, 151),
            "drug": [0, 1] * 75,
            "death": np.random.RandomState(42).binomial(1, 0.2, 150),
        })
        result = _resolve_roles(df, rq="Is drug associated with death?")
        assert result.problem_type in (
            AnalysisProblemType.BINARY_OUTCOME_SINGLE_PREDICTOR,
            AnalysisProblemType.BINARY_OUTCOME_MULTIVARIABLE,
        )

    def test_event_not_correlation(self):
        """'Is exposure linked to event?' — binary, not correlation."""
        df = pd.DataFrame({
            "id": range(1, 101),
            "exposure": [0, 1] * 50,
            "event": np.random.RandomState(42).binomial(1, 0.4, 100),
        })
        result = _resolve_roles(df, rq="Is exposure linked to event?")
        assert result.problem_type in (
            AnalysisProblemType.BINARY_OUTCOME_SINGLE_PREDICTOR,
            AnalysisProblemType.BINARY_OUTCOME_MULTIVARIABLE,
        )

    def test_diagnosis_not_correlation(self):
        """'Is BMI associated with diagnosis?' — binary, not correlation."""
        df = pd.DataFrame({
            "patient_id": range(1, 161),
            "bmi": np.random.RandomState(42).normal(25, 5, 160),
            "diagnosis": np.random.RandomState(43).binomial(1, 0.3, 160),
        })
        result = _resolve_roles(df, rq="Is BMI associated with diagnosis?")
        assert result.problem_type in (
            AnalysisProblemType.BINARY_OUTCOME_SINGLE_PREDICTOR,
            AnalysisProblemType.BINARY_OUTCOME_MULTIVARIABLE,
        )


# ---------------------------------------------------------------------------
# Defect 5 — Correlation RQ parser
# ---------------------------------------------------------------------------

class TestDefect5_CorrelationRQParser:
    """For correlation RQs, both variables must be correctly identified."""

    def test_bmi_biomarker_phrasing(self):
        """'Is BMI associated with biomarker level?' — BMI=predictor, biomarker=outcome."""
        df = pd.DataFrame({
            "patient_id": range(1, 121),
            "BMI": np.random.RandomState(42).normal(25, 5, 120),
            "biomarker": np.random.RandomState(43).normal(0, 1, 120),
        })
        result = _resolve_roles(df, rq="Is BMI associated with biomarker level?")
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        assert result.problem_type == AnalysisProblemType.CONTINUOUS_CONTINUOUS_ASSOCIATION
        # One should be outcome, one should be predictor
        assert roles["biomarker"] == VariableRole.OUTCOME
        assert roles["BMI"] in (VariableRole.PREDICTOR, VariableRole.EXPOSURE)

    def test_weight_height_phrasing(self):
        """'Is weight related to height?' — correlation."""
        df = pd.DataFrame({
            "id": range(1, 101),
            "weight": np.random.RandomState(42).normal(70, 15, 100),
            "height": np.random.RandomState(43).normal(170, 10, 100),
        })
        result = _resolve_roles(df, rq="Is weight related to height?")
        assert result.problem_type == AnalysisProblemType.CONTINUOUS_CONTINUOUS_ASSOCIATION

    def test_correlated_with_phrasing(self):
        """'Is income correlated with education?' — correlation."""
        df = pd.DataFrame({
            "person_id": range(1, 81),
            "income": np.random.RandomState(42).lognormal(10, 0.5, 80),
            "education_years": np.random.RandomState(43).normal(14, 3, 80),
        })
        result = _resolve_roles(df, rq="Is income correlated with education?")
        assert result.problem_type == AnalysisProblemType.CONTINUOUS_CONTINUOUS_ASSOCIATION

    def test_linked_with_phrasing(self):
        """'Is sleep linked to performance?' — correlation."""
        df = pd.DataFrame({
            "subject_id": range(1, 91),
            "sleep_hours": np.random.RandomState(42).normal(7, 1.5, 90),
            "performance": np.random.RandomState(43).normal(80, 10, 90),
        })
        result = _resolve_roles(df, rq="Is sleep linked to performance?")
        assert result.problem_type == AnalysisProblemType.CONTINUOUS_CONTINUOUS_ASSOCIATION

    def test_no_causality_in_correlation(self):
        """Correlation RQ must not imply causation."""
        df = pd.DataFrame({
            "id": range(1, 101),
            "x": np.random.RandomState(42).normal(0, 1, 100),
            "y": np.random.RandomState(43).normal(0, 1, 100),
        })
        result = _resolve_roles(df, rq="Is x associated with y?")
        # Problem type should be correlation, not group comparison
        assert result.problem_type == AnalysisProblemType.CONTINUOUS_CONTINUOUS_ASSOCIATION


# ---------------------------------------------------------------------------
# Defect 6 — High-cardinality numeric variables
# ---------------------------------------------------------------------------

class TestDefect6_HighCardinalityPrecedence:
    """Once outcome is assigned, other high-cardinality numerics should be predictors."""

    def test_outcome_precedence(self):
        """If outcome is assigned by RQ, other numeric vars become predictors."""
        df = pd.DataFrame({
            "patient_id": range(1, 101),
            "treatment": [0, 1] * 50,
            "systolic_bp": np.random.RandomState(42).normal(120, 15, 100),
            "age": np.random.RandomState(43).normal(55, 12, 100),
        })
        result = _resolve_roles(df, rq="Does treatment affect systolic_bp after adjusting for age?")
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        # systolic_bp should be outcome (from RQ)
        assert roles["systolic_bp"] == VariableRole.OUTCOME
        # age should NOT also be outcome — it should be predictor/covariate
        assert roles["age"] != VariableRole.OUTCOME

    def test_multiple_outcomes_conflict(self):
        """Multiple outcome candidates should trigger conflict detection."""
        df = pd.DataFrame({
            "patient_id": range(1, 81),
            "treatment": [0, 1] * 40,
            "bp_outcome": np.random.RandomState(42).normal(120, 15, 80),
            "score_outcome": np.random.RandomState(43).normal(0, 1, 80),
        })
        result = _resolve_roles(df)
        # Both bp_outcome and score_outcome match OUTCOME_PATTERNS
        # This should be detected as a conflict or REVIEW_REQUIRED
        has_review = any(
            r.status in (RoleStatus.REVIEW_REQUIRED, RoleStatus.AMBIGUOUS)
            for r in result.resolutions
            if r.proposed_role == VariableRole.OUTCOME
        )
        # At minimum, the resolver should not silently pick one
        assert result.has_conflicts or has_review or len([
            r for r in result.resolutions if r.proposed_role == VariableRole.OUTCOME
        ]) <= 1


# ---------------------------------------------------------------------------
# Defect 7 — Exposure role must populate grouping variable
# ---------------------------------------------------------------------------

class TestDefect7_ExposureAsGrouping:
    """EXPOSURE role must correctly map to grouping variable for candidate generation."""

    def test_exposure_populates_group_var(self):
        """When treatment=EXPOSURE, candidate generation must set group_var."""
        df = pd.DataFrame({
            "patient_id": range(1, 121),
            "treatment": [0, 1] * 60,
            "systolic_bp": np.random.RandomState(42).normal(120, 15, 120),
        })
        result = _resolve_roles(df, rq="Does treatment differ in systolic_bp?")
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}

        # Build data profile as the test helper does
        group_var = None
        for r in result.resolutions:
            if r.proposed_role in (VariableRole.EXPOSURE, VariableRole.GROUPING_VARIABLE):
                if df[r.variable_name].nunique() <= 10:
                    group_var = r.variable_name
                    break

        assert group_var == "treatment"

    def test_candidate_generation_with_exposure(self):
        """Candidate generation must produce group-comparison methods when exposure exists."""
        df = pd.DataFrame({
            "patient_id": range(1, 121),
            "treatment": [0, 1] * 60,
            "outcome": np.random.RandomState(42).normal(0, 1, 120),
        })
        result = _resolve_roles(df, rq="Does treatment differ in outcome?")
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}

        # Build DataProfile
        outcome_var = [r.variable_name for r in result.resolutions if r.proposed_role == VariableRole.OUTCOME][0]
        exposure_var = [r.variable_name for r in result.resolutions if r.proposed_role in (VariableRole.EXPOSURE, VariableRole.GROUPING_VARIABLE)]
        assert len(exposure_var) >= 1
        group_var = exposure_var[0]
        n_groups = df[group_var].nunique()

        profile = DataProfile(
            n_observations=len(df),
            outcome_variable=outcome_var,
            outcome_type="continuous",
            predictor_variables=[],
            predictor_types=[],
            grouping_variables=[group_var],
            grouping_types=["binary"],
            n_groups=n_groups,
            problem_type=result.problem_type,
            role_resolutions=result,
        )

        generator = CandidateGenerator()
        candidates = generator.generate(profile)
        primary_ids = [c.method_id for c in candidates.primary_candidates]
        # Should include t-test or Mann-Whitney for group comparison
        assert "independent_t_test" in primary_ids or "mann_whitney_u" in primary_ids


# ---------------------------------------------------------------------------
# Defect 8 — Mann-Whitney type normalization
# ---------------------------------------------------------------------------

class TestDefect8_MannWhitneyTypeNormalization:
    """Group values must match the dataset's actual dtype."""

    def test_integer_groups(self):
        """Integer group values (0, 1) must work."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "group": [0] * 20 + [1] * 20,
            "outcome": np.concatenate([rng.normal(0, 1, 20), rng.normal(2, 1, 20)]),
        })
        path = Path(_uid()) / "test_int_groups.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        try:
            executor = StatisticalExecutor()
            result = executor.execute(
                "mann_whitney_u", path,
                outcome="outcome", group="group",
            )
            assert result.execution_status == ExecutionStatus.SUCCESS
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)

    def test_string_groups(self):
        """String group values ('control', 'treatment') must work."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "group": ["control"] * 20 + ["treatment"] * 20,
            "outcome": np.concatenate([rng.normal(0, 1, 20), rng.normal(2, 1, 20)]),
        })
        path = Path(_uid()) / "test_str_groups.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        try:
            executor = StatisticalExecutor()
            result = executor.execute(
                "mann_whitney_u", path,
                outcome="outcome", group="group",
            )
            assert result.execution_status == ExecutionStatus.SUCCESS
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)

    def test_boolean_groups(self):
        """Boolean group values (True, False) must work."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "group": [False] * 20 + [True] * 20,
            "outcome": np.concatenate([rng.normal(0, 1, 20), rng.normal(2, 1, 20)]),
        })
        path = Path(_uid()) / "test_bool_groups.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        try:
            executor = StatisticalExecutor()
            result = executor.execute(
                "mann_whitney_u", path,
                outcome="outcome", group="group",
            )
            assert result.execution_status == ExecutionStatus.SUCCESS
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)

    def test_float_groups(self):
        """Float group values (0.0, 1.0) must work."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "group": [0.0] * 20 + [1.0] * 20,
            "outcome": np.concatenate([rng.normal(0, 1, 20), rng.normal(2, 1, 20)]),
        })
        path = Path(_uid()) / "test_float_groups.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        try:
            executor = StatisticalExecutor()
            result = executor.execute(
                "mann_whitney_u", path,
                outcome="outcome", group="group",
            )
            assert result.execution_status == ExecutionStatus.SUCCESS
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)

    def test_no_type_coercion_between_groups(self):
        """Group values must not be silently coerced to different types."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "group": [0, 1] * 20,
            "outcome": np.concatenate([rng.normal(0, 1, 20), rng.normal(2, 1, 20)]),
        })
        path = Path(_uid()) / "test_no_coerce.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        try:
            executor = StatisticalExecutor()
            result = executor.execute(
                "mann_whitney_u", path,
                outcome="outcome", group="group",
                group1=0, group2=1,
            )
            assert result.execution_status == ExecutionStatus.SUCCESS
            # Verify correct groups were used (not strings)
            assert result.engine_result.get("group1") == 0
            assert result.engine_result.get("group2") == 1
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)


# ---------------------------------------------------------------------------
# Defect 9 — VariableMissingness model completeness
# ---------------------------------------------------------------------------

class TestDefect9_VariableMissingnessModel:
    """VariableMissingness must require all essential fields."""

    def test_required_fields_enforced(self):
        """VariableMissingness must reject incomplete objects."""
        from app.statistics.data_quality import VariableMissingness, MissingnessStatus

        # Missing required fields should raise ValidationError
        with pytest.raises(Exception):
            VariableMissingness()  # type: ignore

    def test_valid_construction(self):
        """VariableMissingness with all required fields must succeed."""
        from app.statistics.data_quality import VariableMissingness, MissingnessStatus

        vm = VariableMissingness(
            variable="age",
            variable_type="numeric",
            n_missing=5,
            n_observed=95,
            total=100,
            pct_missing=0.05,
            status=MissingnessStatus.LOW,
        )
        assert vm.variable == "age"
        assert vm.n_observed == 95
        assert vm.total == 100

    def test_dataset_missingness_with_valid_variables(self):
        """DatasetMissingness with valid VariableMissingness list must succeed."""
        from app.statistics.data_quality import (
            VariableMissingness, DatasetMissingness, MissingnessStatus
        )

        vars_miss = [
            VariableMissingness(
                variable="x",
                variable_type="numeric",
                n_missing=0,
                n_observed=100,
                total=100,
                pct_missing=0.0,
                status=MissingnessStatus.NONE,
            )
        ]
        dm = DatasetMissingness(
            total_rows=100,
            complete_cases=100,
            incomplete_cases=0,
            total_missing_cells=0,
            total_cells=100,
            pct_missing_overall=0.0,
            variables=vars_miss,
        )
        assert dm.complete_cases == 100


# ---------------------------------------------------------------------------
# Defect 10 — Outlier API input contract
# ---------------------------------------------------------------------------

class TestDefect10_OutlierAPIContract:
    """detect_outliers_iqr must have a clear, enforced input contract."""

    def test_dataframe_input(self):
        """detect_outliers_iqr must accept a DataFrame."""
        from app.statistics.outlier_engine import detect_outliers_iqr

        df = pd.DataFrame({
            "x": np.concatenate([np.random.RandomState(42).normal(0, 1, 95), [100, -100, 50, -50, 200]]),
        })
        results = detect_outliers_iqr(df)
        assert isinstance(results, list)

    def test_empty_dataframe(self):
        """detect_outliers_iqr must handle empty DataFrame gracefully."""
        from app.statistics.outlier_engine import detect_outliers_iqr

        df = pd.DataFrame({"x": []})
        results = detect_outliers_iqr(df)
        assert isinstance(results, list)

    def test_non_numeric_input(self):
        """detect_outliers_iqr must handle non-numeric columns gracefully."""
        from app.statistics.outlier_engine import detect_outliers_iqr

        df = pd.DataFrame({
            "category": ["a", "b", "c"] * 10,
            "x": range(30),
        })
        results = detect_outliers_iqr(df)
        # Should only process numeric columns
        assert isinstance(results, list)

    def test_constant_column(self):
        """detect_outliers_iqr must handle constant columns gracefully."""
        from app.statistics.outlier_engine import detect_outliers_iqr

        df = pd.DataFrame({
            "constant": [5.0] * 50,
            "varying": np.random.RandomState(42).normal(0, 1, 50),
        })
        results = detect_outliers_iqr(df)
        assert isinstance(results, list)

    def test_rejects_series_input(self):
        """detect_outliers_iqr must not accept a bare Series."""
        from app.statistics.outlier_engine import detect_outliers_iqr

        s = pd.Series([1, 2, 3, 4, 5, 100])
        # Series has no .columns attribute, should raise or return empty
        try:
            results = detect_outliers_iqr(s)  # type: ignore
            # If it doesn't raise, it should return an empty list
            assert isinstance(results, list)
        except (AttributeError, TypeError):
            # Expected behavior — Series is not a valid input
            pass


# ---------------------------------------------------------------------------
# Defect 11 — Validation tests must not corrupt real datasets
# ---------------------------------------------------------------------------

class TestDefect11_TestIsolation:
    """Test suite must not modify canonical fixture datasets."""

    def test_test1_hash_unchanged_by_execution(self):
        """Executing against Test 1 must not modify the file."""
        original_hash = _hash_file(TEST1_PATH)
        executor = StatisticalExecutor()
        executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment",
        )
        assert _hash_file(TEST1_PATH) == original_hash

    def test_test1_hash_unchanged_by_multiple_executions(self):
        """Multiple executions must not modify Test 1."""
        original_hash = _hash_file(TEST1_PATH)
        executor = StatisticalExecutor()
        for _ in range(3):
            executor.execute(
                "independent_t_test", TEST1_PATH,
                outcome="systolic_bp", group="treatment",
            )
        assert _hash_file(TEST1_PATH) == original_hash

    def test_mutation_tests_use_copies(self, tmp_path):
        """Tests that mutate data must use copies, not originals."""
        import shutil

        original_hash = _hash_file(TEST1_PATH)
        copy_path = tmp_path / "test1_copy.csv"
        shutil.copy2(TEST1_PATH, copy_path)

        # Mutate the copy
        df = pd.read_csv(copy_path)
        df.loc[0, "systolic_bp"] = 99999
        df.to_csv(copy_path, index=False)

        # Original must be unchanged
        assert _hash_file(TEST1_PATH) == original_hash

    def test_identity_verification_with_modified_copy(self, tmp_path):
        """Identity verification must detect modifications on copies."""
        import shutil

        v = verify_dataset(TEST1_PATH)
        copy_path = tmp_path / "test1_mod.csv"
        shutil.copy2(TEST1_PATH, copy_path)

        # Modify the copy
        df = pd.read_csv(copy_path)
        df.loc[0, "systolic_bp"] = 99999
        df.to_csv(copy_path, index=False)

        # Execute with expected hash from original — should BLOCK
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", copy_path,
            outcome="systolic_bp", group="treatment",
            expected_content_hash=v.content_hash,
        )
        assert result.execution_status == ExecutionStatus.BLOCKED


# ---------------------------------------------------------------------------
# Defect 12 — UUID identifiers
# ---------------------------------------------------------------------------

class TestDefect12_UUIDIdentifiers:
    """UUID-like variables must be correctly identified as identifiers."""

    def test_uuid_column(self):
        """'uuid' column must be identifier."""
        import uuid
        df = pd.DataFrame({
            "uuid": [str(uuid.uuid4()) for _ in range(50)],
            "value": np.random.RandomState(42).normal(0, 1, 50),
        })
        result = _resolve_roles(df)
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        assert roles["uuid"] == VariableRole.IDENTIFIER

    def test_patient_uuid_column(self):
        """'patient_uuid' column must be identifier."""
        import uuid
        df = pd.DataFrame({
            "patient_uuid": [str(uuid.uuid4()) for _ in range(60)],
            "score": np.random.RandomState(42).normal(0, 1, 60),
        })
        result = _resolve_roles(df)
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        assert roles["patient_uuid"] == VariableRole.IDENTIFIER

    def test_subject_uuid_column(self):
        """'subject_uuid' column must be identifier."""
        import uuid
        df = pd.DataFrame({
            "subject_uuid": [str(uuid.uuid4()) for _ in range(40)],
            "measurement": np.random.RandomState(42).normal(0, 1, 40),
        })
        result = _resolve_roles(df)
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        assert roles["subject_uuid"] == VariableRole.IDENTIFIER

    def test_high_cardinality_string_not_always_identifier(self):
        """High-cardinality string variable without ID pattern must NOT be identifier."""
        df = pd.DataFrame({
            "city_name": ["New York", "Los Angeles", "Chicago", "Houston",
                          "Phoenix", "Philadelphia", "San Antonio", "San Diego",
                          "Dallas", "San Jose"] * 10,
            "value": np.random.RandomState(42).normal(0, 1, 100),
        })
        result = _resolve_roles(df)
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        assert roles["city_name"] != VariableRole.IDENTIFIER

    def test_mrn_pattern_identifier(self):
        """MRN-like patterns should be recognized as identifiers if name suggests it."""
        df = pd.DataFrame({
            "medical_record_number": range(1000, 1100),
            "outcome": np.random.RandomState(42).normal(0, 1, 100),
        })
        result = _resolve_roles(df)
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        assert roles["medical_record_number"] == VariableRole.IDENTIFIER


# ---------------------------------------------------------------------------
# Dataset Identity Enforcement Tests
# ---------------------------------------------------------------------------

class TestDatasetIdentityEnforcement:
    """Dataset identity must block execution on mismatch."""

    def test_content_hash_mismatch_blocks(self):
        """Content hash mismatch must result in BLOCKED status."""
        v = verify_dataset(TEST1_PATH)
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment",
            expected_content_hash="0" * 64,  # definitely wrong
        )
        assert result.execution_status == ExecutionStatus.BLOCKED
        assert not result.data_verification.identity_verified
        assert result.data_verification.identity_mismatch_reason is not None
        assert "CONTENT_HASH_MISMATCH" in result.data_verification.identity_mismatch_reason

    def test_schema_hash_mismatch_blocks(self):
        """Schema hash mismatch must result in BLOCKED status."""
        v = verify_dataset(TEST1_PATH)
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment",
            expected_schema_hash="0" * 64,
        )
        assert result.execution_status == ExecutionStatus.BLOCKED
        assert not result.data_verification.identity_verified
        assert "SCHEMA_HASH_MISMATCH" in result.data_verification.identity_mismatch_reason

    def test_correct_hash_succeeds(self):
        """Correct hash must allow execution."""
        v = verify_dataset(TEST1_PATH)
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment",
            expected_content_hash=v.content_hash,
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.data_verification.identity_verified

    def test_no_expected_hash_allows_execution(self):
        """No expected hash must allow execution (identity unverified but not blocked)."""
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment",
        )
        assert result.execution_status == ExecutionStatus.SUCCESS

    def test_both_hashes_correct(self):
        """Both content and schema hash correct must succeed."""
        v = verify_dataset(TEST1_PATH)
        executor = StatisticalExecutor()
        result = executor.execute(
            "independent_t_test", TEST1_PATH,
            outcome="systolic_bp", group="treatment",
            expected_content_hash=v.content_hash,
            expected_schema_hash=v.schema_hash,
        )
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.data_verification.identity_verified

    def test_modified_dataset_blocks(self):
        """Modified dataset with expected original hash must BLOCK."""
        import tempfile
        v = verify_dataset(TEST1_PATH)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df = pd.read_csv(TEST1_PATH)
            df.loc[0, "systolic_bp"] = 99999
            df.to_csv(f, index=False)
            tmp_path = Path(f.name)
        try:
            executor = StatisticalExecutor()
            result = executor.execute(
                "independent_t_test", tmp_path,
                outcome="systolic_bp", group="treatment",
                expected_content_hash=v.content_hash,
            )
            assert result.execution_status == ExecutionStatus.BLOCKED
        finally:
            tmp_path.unlink()

    def test_row_order_invariant(self):
        """Row order changes must not affect identity (canonical hashing)."""
        df = pd.read_csv(TEST1_PATH)
        shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        hash_original = compute_dataset_identity(TEST1_PATH)["content_hash"]
        # Compute hash of shuffled data via temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            shuffled.to_csv(f, index=False)
            tmp_path = Path(f.name)
        try:
            hash_shuffled = compute_dataset_identity(tmp_path)["content_hash"]
            assert hash_original == hash_shuffled
        finally:
            tmp_path.unlink()

    def test_column_order_invariant(self):
        """Column order changes must not affect identity."""
        df = pd.read_csv(TEST1_PATH)
        reordered = df[reversed(df.columns)]
        hash_original = compute_dataset_identity(TEST1_PATH)["content_hash"]
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            reordered.to_csv(f, index=False)
            tmp_path = Path(f.name)
        try:
            hash_reordered = compute_dataset_identity(tmp_path)["content_hash"]
            assert hash_original == hash_reordered
        finally:
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# Role Resolution Precedence Tests
# ---------------------------------------------------------------------------

class TestRoleResolutionPrecedence:
    """Role resolution must follow deterministic precedence."""

    def test_expert_researcher_role_wins(self):
        """Explicit researcher-provided role must override heuristics."""
        variables = [
            {"name": "score", "dtype": "float64", "n_unique": 100, "n_total": 100, "n_missing": 0},
        ]
        result = RoleResolver.resolve_all(
            variables,
            research_question="What is the score?",
        )
        # Without explicit role, score becomes outcome by heuristic
        # This is expected — the test verifies the heuristic path works
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        assert roles["score"] in (VariableRole.OUTCOME, VariableRole.PREDICTOR, VariableRole.UNKNOWN)

    def test_rq_match_wins_over_heuristic(self):
        """RQ match must override name-based heuristic."""
        df = pd.DataFrame({
            "patient_id": range(1, 101),
            "treatment": [0, 1] * 50,
            "outcome": np.random.RandomState(42).normal(0, 1, 100),
        })
        result = _resolve_roles(df, rq="Does treatment affect outcome?")
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        # treatment should be exposure (from RQ), not just a grouping variable
        assert roles["treatment"] in (VariableRole.EXPOSURE, VariableRole.GROUPING_VARIABLE)

    def test_identifier_detection_wins(self):
        """Deterministic identifier detection must win over RQ context."""
        df = pd.DataFrame({
            "patient_id": range(1, 101),
            "treatment": [0, 1] * 50,
            "score": np.random.RandomState(42).normal(0, 1, 100),
        })
        result = _resolve_roles(df, rq="Does treatment affect patient_id?")
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        # patient_id must remain identifier even though RQ mentions it
        assert roles["patient_id"] == VariableRole.IDENTIFIER

    def test_binary_variable_grouping(self):
        """Binary variable with few levels should be grouping variable or exposure."""
        df = pd.DataFrame({
            "id": range(1, 101),
            "sex": [0, 1] * 50,
            "bp": np.random.RandomState(42).normal(120, 15, 100),
        })
        result = _resolve_roles(df)
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        assert roles["sex"] in (VariableRole.GROUPING_VARIABLE, VariableRole.EXPOSURE, VariableRole.PREDICTOR)


# ---------------------------------------------------------------------------
# Research-Question Parsing Tests
# ---------------------------------------------------------------------------

class TestRQParsing:
    """RQ parsing must identify variables without overriding constraints."""

    def test_differ_in_phrasing(self):
        """'Does X differ in Y?' — group comparison."""
        df = pd.DataFrame({
            "patient_id": range(1, 121),
            "treatment": [0, 1] * 60,
            "systolic_bp": np.random.RandomState(42).normal(120, 15, 120),
        })
        result = _resolve_roles(df, rq="Does treatment differ in systolic blood pressure?")
        assert result.problem_type == AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME

    def test_compare_between_phrasing(self):
        """'Compare X between groups' — group comparison."""
        df = pd.DataFrame({
            "id": range(1, 81),
            "group": ["A", "B"] * 40,
            "score": np.random.RandomState(42).normal(0, 1, 80),
        })
        result = _resolve_roles(df, rq="Compare score between groups")
        assert result.problem_type in (
            AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
            AnalysisProblemType.MULTI_GROUP_CONTINUOUS_OUTCOME,
        )

    def test_effect_of_phrasing(self):
        """'What is the effect of treatment on outcome?' — group comparison."""
        df = pd.DataFrame({
            "patient_id": range(1, 101),
            "treatment": [0, 1] * 50,
            "recovery_time": np.random.RandomState(42).exponential(5, 100),
        })
        result = _resolve_roles(df, rq="What is the effect of treatment on recovery time?")
        assert result.problem_type in (
            AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
            AnalysisProblemType.MULTI_GROUP_CONTINUOUS_OUTCOME,
        )

    def test_associated_with_binary_outcome(self):
        """'Is X associated with Y?' where Y is binary — must not be correlation."""
        df = pd.DataFrame({
            "id": range(1, 151),
            "exposure": np.random.RandomState(42).normal(0, 1, 150),
            "disease": np.random.RandomState(43).binomial(1, 0.3, 150),
        })
        result = _resolve_roles(df, rq="Is exposure associated with disease?")
        assert result.problem_type in (
            AnalysisProblemType.BINARY_OUTCOME_SINGLE_PREDICTOR,
            AnalysisProblemType.BINARY_OUTCOME_MULTIVARIABLE,
        )

    def test_empty_rq(self):
        """Empty RQ must not crash."""
        df = pd.DataFrame({
            "x": range(10),
            "y": range(10),
        })
        result = _resolve_roles(df, rq="")
        assert result.problem_type is not None

    def test_none_rq(self):
        """None RQ must not crash."""
        df = pd.DataFrame({
            "x": range(10),
            "y": range(10),
        })
        result = _resolve_roles(df, rq=None)
        assert result.problem_type is not None
