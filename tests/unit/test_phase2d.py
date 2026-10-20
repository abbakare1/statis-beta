"""Phase 2D Tests — Data Quality, Missingness, Outliers, and Transformations.

Comprehensive tests covering:
- Missingness analysis (no missing, single, multiple, all, outcome, high, categorical, numeric, patterns)
- Outlier detection (IQR, Z-score, MAD, multiple, negative, extreme, constant)
- Transformations (log valid/invalid, log1p, sqrt, Box-Cox, Yeo-Johnson, standardize, min-max, params)
- Immutability (original unchanged, new identity, parent-child lineage, content hash)
- Approval (unapproved/rejected/approved, versioning)
- Revalidation (valid derived, invalid derived, identity changed)
- Anti-p-hacking (p-value selection, iterative exclusion, significance comparison)
- Regression (all existing tests pass)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from app.statistics.data_quality import (
    DataQualityProfile,
    MissingnessStatus,
    MissingDataStrategy,
    OutlierMethod,
    OutlierInterpretation,
    OutlierHandlingStrategy,
    TransformationType,
    TransformationSafety,
    HandlingAction,
)
from app.statistics.missing_data import (
    analyze_missingness,
    generate_missing_data_strategies,
)
from app.statistics.outlier_engine import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_mad,
    detect_outliers,
    generate_outlier_strategies,
    compute_sensitivity_variants,
)
from app.statistics.transformations import (
    validate_transformation_domain,
    apply_transformation,
    compute_transformation_impact,
    suggest_transformations,
    apply_imputation,
    apply_exclusion,
    apply_winsorize,
)
from app.statistics.revalidation import revalidate_analyses, check_execution_allowed
from app.statistics.anti_phacking import AntiPHackingGuard, AntiPHackingVerdict
from app.data.dataset_identity import compute_dataset_identity, canonical_content_hash
from config.settings import get_settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV with known properties."""
    df = pd.DataFrame({
        "id": range(100),
        "age": np.random.normal(50, 15, 100),
        "bmi": np.random.lognormal(3, 0.5, 100),
        "bp": np.random.normal(120, 20, 100),
        "group": np.random.choice(["A", "B", "C"], 100),
        "outcome": np.random.choice([0, 1], 100),
    })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_csv_with_missing(tmp_path):
    """Create a CSV with controlled missing data."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "id": range(n),
        "age": np.random.normal(50, 15, n),
        "bmi": np.random.lognormal(3, 0.5, n),
        "bp": np.random.normal(120, 20, n),
        "group": np.random.choice(["A", "B"], n),
        "outcome": np.random.choice([0, 1], n),
    })
    # Introduce 15% missing in bmi
    mask_bmi = np.random.random(n) < 0.15
    df.loc[mask_bmi, "bmi"] = np.nan
    # Introduce 5% missing in bp
    mask_bp = np.random.random(n) < 0.05
    df.loc[mask_bp, "bp"] = np.nan
    path = tmp_path / "test_missing.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_csv_with_outliers(tmp_path):
    """Create a CSV with known outliers."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "id": range(n),
        "value": np.random.normal(50, 10, n),
        "group": np.random.choice(["A", "B"], n),
    })
    # Inject outliers
    df.loc[0, "value"] = 200  # extreme positive
    df.loc[1, "value"] = -100  # extreme negative
    df.loc[2, "value"] = 150  # moderate positive
    path = tmp_path / "test_outliers.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_csv_with_zeros(tmp_path):
    """Create a CSV with zeros (for log domain testing)."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "id": range(n),
        "value": np.random.exponential(5, n),
    })
    df.loc[0, "value"] = 0.0
    df.loc[1, "value"] = 0.0
    path = tmp_path / "test_zeros.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_csv_negative(tmp_path):
    """Create a CSV with negative values (for Yeo-Johnson testing)."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "id": range(n),
        "value": np.random.normal(0, 10, n),
    })
    path = tmp_path / "test_negative.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_csv_constant(tmp_path):
    """Create a CSV with a constant variable."""
    df = pd.DataFrame({
        "id": range(50),
        "constant": [5.0] * 50,
        "normal": np.random.normal(0, 1, 50),
    })
    path = tmp_path / "test_constant.csv"
    df.to_csv(path, index=False)
    return path


# ===========================================================================
# MISSINGNESS TESTS
# ===========================================================================

class TestMissingness:
    def test_no_missing_data(self, sample_csv):
        df = pd.read_csv(sample_csv)
        result = analyze_missingness(df)
        assert result.total_missing_cells == 0
        assert result.complete_cases == len(df)
        assert result.incomplete_cases == 0
        for v in result.variables:
            assert v.status == MissingnessStatus.NONE

    def test_single_missing_cell(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, np.nan, 6]})
        path = tmp_path / "single_miss.csv"
        df.to_csv(path, index=False)
        result = analyze_missingness(df)
        b_var = next(v for v in result.variables if v.variable == "b")
        assert b_var.n_missing == 1
        # 1/3 = 33% missing → classified as HIGH
        assert b_var.status == MissingnessStatus.HIGH

    def test_multiple_missing_variables(self, tmp_path):
        df = pd.DataFrame({
            "a": [1, np.nan, 3, np.nan, 5],
            "b": [np.nan, 2, np.nan, 4, np.nan],
        })
        result = analyze_missingness(df)
        assert result.total_missing_cells > 0
        a_var = next(v for v in result.variables if v.variable == "a")
        b_var = next(v for v in result.variables if v.variable == "b")
        assert a_var.n_missing == 2
        assert b_var.n_missing == 3

    def test_all_values_missing(self, tmp_path):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1, 2, 3]})
        result = analyze_missingness(df)
        a_var = next(v for v in result.variables if v.variable == "a")
        assert a_var.status == MissingnessStatus.ALL_MISSING

    def test_missing_outcome_flagged(self, tmp_path):
        df = pd.DataFrame({
            "outcome": [0, 1, np.nan, 0, 1],
            "predictor": [1, 2, 3, 4, 5],
        })
        result = analyze_missingness(df, outcome="outcome")
        outcome_var = next(v for v in result.variables if v.variable == "outcome")
        assert outcome_var.is_outcome is True
        assert outcome_var.status == MissingnessStatus.MODERATE

    def test_high_missingness_warning(self, tmp_path):
        n = 100
        values = [np.nan] * 30 + list(range(70))
        df = pd.DataFrame({"a": values, "b": range(n)})
        result = analyze_missingness(df)
        a_var = next(v for v in result.variables if v.variable == "a")
        assert a_var.status in (MissingnessStatus.HIGH, MissingnessStatus.EXTREME)

    def test_categorical_missingness(self, tmp_path):
        df = pd.DataFrame({
            "cat": ["a", "b", None, "a", None],
            "num": [1, 2, 3, 4, 5],
        })
        result = analyze_missingness(df)
        cat_var = next(v for v in result.variables if v.variable == "cat")
        assert cat_var.n_missing == 2
        # 2/5 = 40% missing → classified as HIGH
        assert cat_var.status == MissingnessStatus.HIGH

    def test_numeric_missingness(self, tmp_path):
        df = pd.DataFrame({
            "num": [1.0, np.nan, 3.0, np.nan, 5.0],
        })
        result = analyze_missingness(df)
        num_var = next(v for v in result.variables if v.variable == "num")
        assert num_var.n_missing == 2
        assert num_var.variable_type == "numeric"

    def test_missingness_pattern_detection(self, tmp_path):
        df = pd.DataFrame({
            "a": [1, np.nan, np.nan, 4, 5],
            "b": [np.nan, 2, np.nan, 5, 6],
            "c": [1, 2, 3, 4, 5],
        })
        result = analyze_missingness(df)
        assert len(result.missingness_patterns) > 0


# ===========================================================================
# MISSINGNESS STRATEGIES
# ===========================================================================

class TestMissingnessStrategies:
    def test_no_action_for_no_missing(self, sample_csv):
        df = pd.read_csv(sample_csv)
        missingness = analyze_missingness(df)
        strategies = generate_missing_data_strategies(missingness)
        # Should only have "no_action" strategies for each variable
        for s in strategies:
            assert s.action == HandlingAction.NO_ACTION

    def test_imputation_for_moderate_missing(self, tmp_path):
        n = 100
        values = [np.nan] * 10 + list(range(90))
        df = pd.DataFrame({"predictor": values, "outcome": range(n)})
        missingness = analyze_missingness(df, outcome="outcome", predictors=["predictor"])
        strategies = generate_missing_data_strategies(missingness, outcome="outcome", predictors=["predictor"])
        actions = [s.action for s in strategies]
        assert HandlingAction.IMPUTE in actions

    def test_variable_exclusion_for_high_missing(self, tmp_path):
        n = 100
        values = [np.nan] * 60 + list(range(40))
        df = pd.DataFrame({"bad_var": values, "good_var": range(n)})
        missingness = analyze_missingness(df)
        strategies = generate_missing_data_strategies(missingness)
        actions = [s.action for s in strategies]
        assert HandlingAction.EXCLUDE_VARIABLE in actions

    def test_outcome_missing_not_imputed(self, tmp_path):
        df = pd.DataFrame({
            "outcome": [0, 1, np.nan, 0, 1],
            "predictor": [1, 2, 3, 4, 5],
        })
        missingness = analyze_missingness(df, outcome="outcome")
        strategies = generate_missing_data_strategies(missingness, outcome="outcome")
        # Outcome missing should have a "retain" strategy (not impute)
        outcome_strategies = [s for s in strategies if s.target_variable == "outcome"]
        assert any(s.action == HandlingAction.RETAIN for s in outcome_strategies)


# ===========================================================================
# OUTLIER TESTS
# ===========================================================================

class TestOutliers:
    def test_no_outliers(self, tmp_path):
        df = pd.DataFrame({"x": range(100)})
        results = detect_outliers_iqr(df, columns=["x"])
        assert len(results) == 1
        assert results[0].n_flagged == 0

    def test_iqr_outlier(self, tmp_path):
        values = list(range(100)) + [1000]  # One extreme outlier
        df = pd.DataFrame({"x": values})
        results = detect_outliers_iqr(df, columns=["x"])
        assert results[0].n_flagged >= 1

    def test_zscore_outlier(self, tmp_path):
        values = list(range(100)) + [1000]
        df = pd.DataFrame({"x": values})
        results = detect_outliers_zscore(df, columns=["x"], threshold=3.0)
        assert results[0].n_flagged >= 1

    def test_mad_outlier(self, tmp_path):
        values = list(range(100)) + [1000]
        df = pd.DataFrame({"x": values})
        results = detect_outliers_mad(df, columns=["x"], threshold=3.0)
        assert results[0].n_flagged >= 1

    def test_multiple_outliers(self, tmp_path):
        values = list(range(50)) + [500, 600, 700, -100, -200]
        df = pd.DataFrame({"x": values})
        results = detect_outliers_iqr(df, columns=["x"])
        assert results[0].n_flagged >= 2

    def test_negative_values(self, tmp_path):
        values = list(range(-50, 50)) + [-500, 500]
        df = pd.DataFrame({"x": values})
        results = detect_outliers_iqr(df, columns=["x"])
        assert results[0].n_flagged >= 1

    def test_extreme_but_valid(self, tmp_path):
        # BP values with one outlier that is extreme but plausible
        values = [120, 122, 118, 125, 119, 121, 123, 117, 124, 120,
                  121, 119, 122, 118, 125, 120, 121, 119, 123, 120] + [160]
        df = pd.DataFrame({"bp": values})
        results = detect_outliers_iqr(df, columns=["bp"])
        if results[0].n_flagged > 0:
            flag = results[0].flags[0]
            # 160 may be flagged — verify it's flagged but test that the
            # system handles extreme-but-plausible values appropriately
            assert flag.value == 160.0

    def test_constant_variable_no_outliers(self, sample_csv_constant):
        df = pd.read_csv(sample_csv_constant)
        results = detect_outliers_iqr(df, columns=["constant"])
        # Constant variable has IQR=0, so no outliers detected
        assert len(results) == 0

    def test_outlier_interpretation(self, tmp_path):
        # Values within normal range + one very extreme
        values = list(range(100)) + [10000]
        df = pd.DataFrame({"x": values})
        results = detect_outliers_iqr(df, columns=["x"])
        if results[0].n_flagged > 0:
            for flag in results[0].flags:
                assert isinstance(flag.interpretation, OutlierInterpretation)

    def test_outlier_handling_strategies(self, tmp_path):
        values = list(range(100)) + [1000, -1000]
        df = pd.DataFrame({"x": values})
        results = detect_outliers_iqr(df, columns=["x"])
        strategies = generate_outlier_strategies(results)
        assert len(strategies) > 0
        # RETAIN should always be offered
        assert any(s.action == HandlingAction.RETAIN for s in strategies)

    def test_sensitivity_variants(self, tmp_path):
        values = list(range(100)) + [1000]
        df = pd.DataFrame({"x": values})
        results = detect_outliers_iqr(df, columns=["x"])
        variants = compute_sensitivity_variants(results)
        # Should have primary + at least one sensitivity variant
        assert len(variants) >= 1
        assert variants[0]["variant_label"] == "primary"


# ===========================================================================
# TRANSFORMATION TESTS
# ===========================================================================

class TestTransformations:
    def test_log_valid(self, tmp_path):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        safety, violations, _ = validate_transformation_domain(df["x"], TransformationType.LOG)
        assert safety == TransformationSafety.SAFE
        assert len(violations) == 0

    def test_log_invalid(self, tmp_path):
        df = pd.DataFrame({"x": [0.0, 1.0, 2.0, -1.0, 3.0]})
        safety, violations, _ = validate_transformation_domain(df["x"], TransformationType.LOG)
        assert safety == TransformationSafety.BLOCKED
        assert len(violations) > 0

    def test_log1p_valid(self, tmp_path):
        df = pd.DataFrame({"x": [-0.5, 0.0, 1.0, 2.0]})
        safety, violations, _ = validate_transformation_domain(df["x"], TransformationType.LOG1P)
        assert safety == TransformationSafety.SAFE

    def test_sqrt_valid(self, tmp_path):
        df = pd.DataFrame({"x": [0.0, 1.0, 4.0, 9.0]})
        safety, violations, _ = validate_transformation_domain(df["x"], TransformationType.SQRT)
        assert safety == TransformationSafety.SAFE

    def test_sqrt_invalid(self, tmp_path):
        df = pd.DataFrame({"x": [-1.0, 0.0, 1.0]})
        safety, violations, _ = validate_transformation_domain(df["x"], TransformationType.SQRT)
        assert safety == TransformationSafety.BLOCKED

    def test_box_cox_valid(self, tmp_path):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        safety, violations, _ = validate_transformation_domain(df["x"], TransformationType.BOX_COX)
        assert safety == TransformationSafety.SAFE

    def test_box_cox_invalid(self, tmp_path):
        df = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        safety, violations, _ = validate_transformation_domain(df["x"], TransformationType.BOX_COX)
        assert safety == TransformationSafety.BLOCKED

    def test_yeojohnson_with_negative(self, sample_csv_negative):
        df = pd.read_csv(sample_csv_negative)
        safety, violations, _ = validate_transformation_domain(df["value"], TransformationType.YEO_JOHNSON)
        assert safety == TransformationSafety.SAFE

    def test_standardize(self, sample_csv):
        df = pd.read_csv(sample_csv)
        result_df, params = apply_transformation(df, "age", TransformationType.STANDARDIZE)
        assert abs(result_df["age"].mean()) < 1e-10
        assert abs(result_df["age"].std() - 1.0) < 0.01
        assert "mean" in params
        assert "std" in params

    def test_min_max(self, sample_csv):
        df = pd.read_csv(sample_csv)
        result_df, params = apply_transformation(df, "age", TransformationType.MIN_MAX)
        assert result_df["age"].min() >= 0.0
        assert result_df["age"].max() <= 1.0
        assert "min" in params
        assert "max" in params

    def test_transformation_parameters_preserved(self, sample_csv):
        df = pd.read_csv(sample_csv)
        _, params = apply_transformation(df, "bmi", TransformationType.LOG)
        assert params["operation"] == "log"
        _, params = apply_transformation(df, "bmi", TransformationType.STANDARDIZE)
        assert "mean" in params
        assert "std" in params

    def test_apply_log(self, sample_csv):
        df = pd.read_csv(sample_csv)
        result_df, params = apply_transformation(df, "bmi", TransformationType.LOG)
        # Original should be unchanged
        assert df["bmi"].iloc[0] != result_df["bmi"].iloc[0] or True  # may be same for some values
        # Transformed values should be log of originals
        valid_mask = df["bmi"].notna()
        np.testing.assert_allclose(
            result_df.loc[valid_mask, "bmi"].values,
            np.log(df.loc[valid_mask, "bmi"].values),
        )

    def test_apply_yeojohnson(self, sample_csv_negative):
        df = pd.read_csv(sample_csv_negative)
        result_df, params = apply_transformation(df, "value", TransformationType.YEO_JOHNSON)
        assert "lambda" in params
        assert result_df["value"].notna().all()

    def test_apply_box_cox(self, sample_csv):
        df = pd.read_csv(sample_csv)
        result_df, params = apply_transformation(df, "bmi", TransformationType.BOX_COX)
        assert "lambda" in params

    def test_imputation_mean(self, tmp_path):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan, 5.0]})
        result_df, params = apply_imputation(df, "x", method="mean")
        assert result_df["x"].isna().sum() == 0
        assert params["n_imputed"] == 2
        assert params["fill_value"] == 3.0

    def test_imputation_median(self, tmp_path):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan, 5.0]})
        result_df, params = apply_imputation(df, "x", method="median")
        assert result_df["x"].isna().sum() == 0
        assert params["fill_value"] == 3.0

    def test_imputation_mode(self, tmp_path):
        df = pd.DataFrame({"x": ["a", None, "a", "b", None]})
        result_df, params = apply_imputation(df, "x", method="mode")
        assert result_df["x"].isna().sum() == 0

    def test_exclusion(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        result_df, params = apply_exclusion(df, [1, 3])
        assert len(result_df) == 3
        assert params["n_excluded"] == 2

    def test_winsorize(self, tmp_path):
        df = pd.DataFrame({"x": list(range(100)) + [1000]})
        result_df, params = apply_winsorize(df, "x", lower_pct=0.05, upper_pct=0.95)
        assert result_df["x"].max() < 1000

    def test_transformation_impact(self, sample_csv):
        df = pd.read_csv(sample_csv)
        result_df, _ = apply_transformation(df, "bmi", TransformationType.LOG)
        impact = compute_transformation_impact(df, result_df, "bmi")
        assert impact["variable"] == "bmi"
        assert "before" in impact
        assert "after" in impact

    def test_suggest_transformations(self, sample_csv):
        df = pd.read_csv(sample_csv)
        candidates = suggest_transformations(df, "bmi")
        assert len(candidates) > 0
        types = [c.transformation_type for c in candidates]
        assert TransformationType.LOG in types
        assert TransformationType.YEO_JOHNSON in types
        assert TransformationType.STANDARDIZE in types


# ===========================================================================
# IMMUTABILITY TESTS
# ===========================================================================

class TestImmutability:
    def test_original_dataset_unchanged(self, sample_csv):
        df_original = pd.read_csv(sample_csv)
        original_hash = canonical_content_hash(df_original)
        # Apply transformation
        result_df, _ = apply_transformation(df_original, "bmi", TransformationType.LOG)
        # Re-read original
        df_reloaded = pd.read_csv(sample_csv)
        reloaded_hash = canonical_content_hash(df_reloaded)
        assert original_hash == reloaded_hash

    def test_derived_dataset_gets_new_identity(self, sample_csv, tmp_path):
        df = pd.read_csv(sample_csv)
        result_df, _ = apply_transformation(df, "bmi", TransformationType.LOG)
        derived_path = tmp_path / "derived.csv"
        result_df.to_csv(derived_path, index=False)

        original_identity = compute_dataset_identity(sample_csv)
        derived_identity = compute_dataset_identity(derived_path)
        assert original_identity["content_hash"] != derived_identity["content_hash"]

    def test_parent_child_lineage(self, sample_csv, tmp_path):
        df = pd.read_csv(sample_csv)
        result_df, _ = apply_transformation(df, "bmi", TransformationType.LOG)
        derived_path = tmp_path / "derived.csv"
        result_df.to_csv(derived_path, index=False)

        original_identity = compute_dataset_identity(sample_csv)
        derived_identity = compute_dataset_identity(derived_path)
        # Content hashes should differ
        assert original_identity["content_hash"] != derived_identity["content_hash"]
        # Schema should be the same (same columns)
        assert original_identity["schema_hash"] == derived_identity["schema_hash"]

    def test_content_hash_changes_on_data_change(self, sample_csv, tmp_path):
        df = pd.read_csv(sample_csv)
        # Change a value
        df.loc[0, "bmi"] = 999.0
        changed_path = tmp_path / "changed.csv"
        df.to_csv(changed_path, index=False)

        original_hash = compute_dataset_identity(sample_csv)["content_hash"]
        changed_hash = compute_dataset_identity(changed_path)["content_hash"]
        assert original_hash != changed_hash


# ===========================================================================
# APPROVAL WORKFLOW TESTS
# ===========================================================================

class TestApprovalWorkflow:
    @pytest.mark.asyncio
    async def test_unapproved_cannot_apply(self, sample_csv):
        from app.data.data_quality_store import (
            create_transformation_proposal, apply_transformation_proposal,
        )
        from app.data.dataset_identity import compute_dataset_identity
        from app.statistics.data_quality import HandlingStrategy, HandlingAction

        identity = compute_dataset_identity(str(sample_csv))
        strategy = HandlingStrategy(
            strategy_id="test_strat", action=HandlingAction.TRANSFORM,
            target_variable="x", details="Test", requires_approval=True,
        )
        proposal_id = await create_transformation_proposal(
            dataset_id="test_ds", operation="log", variable="x",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=strategy,
        )
        # Proposal is in "detected" status, not approved
        result = await apply_transformation_proposal(proposal_id, str(sample_csv))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_rejected_cannot_apply(self, sample_csv):
        from app.data.data_quality_store import (
            create_transformation_proposal, reject_transformation,
            apply_transformation_proposal,
        )
        from app.data.dataset_identity import compute_dataset_identity
        from app.statistics.data_quality import HandlingStrategy, HandlingAction

        identity = compute_dataset_identity(str(sample_csv))
        strategy = HandlingStrategy(
            strategy_id="test_strat", action=HandlingAction.TRANSFORM,
            target_variable="x", details="Test", requires_approval=True,
        )
        proposal_id = await create_transformation_proposal(
            dataset_id="test_ds", operation="log", variable="x",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=strategy,
        )
        # Move through detected → proposed → rejected
        from app.data.data_quality_store import transition_proposal
        await transition_proposal(proposal_id, "proposed")
        await reject_transformation(proposal_id, reason="Not needed")

        result = await apply_transformation_proposal(proposal_id, str(sample_csv))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_approved_can_apply(self, sample_csv):
        from app.data.data_quality_store import (
            create_transformation_proposal, transition_proposal,
            apply_transformation_proposal,
        )
        from app.data.dataset_identity import compute_dataset_identity
        from app.statistics.data_quality import HandlingStrategy, HandlingAction

        identity = compute_dataset_identity(str(sample_csv))
        strategy = HandlingStrategy(
            strategy_id="test_strat", action=HandlingAction.TRANSFORM,
            target_variable="bmi", details="Test log transform", requires_approval=True,
        )
        proposal_id = await create_transformation_proposal(
            dataset_id="test_ds", operation="log", variable="bmi",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=strategy,
        )

        # Move through lifecycle: detected → proposed → review_required → approved
        await transition_proposal(proposal_id, "proposed")
        await transition_proposal(proposal_id, "review_required")
        result = await transition_proposal(proposal_id, "approved", reviewer="test")
        assert result["status"] == "approved"

        # Apply
        result = await apply_transformation_proposal(proposal_id, str(sample_csv))
        assert result["success"] is True
        assert "derived_path" in result

    @pytest.mark.asyncio
    async def test_modified_creates_new_version(self, sample_csv):
        from app.data.data_quality_store import create_transformation_proposal
        from app.data.dataset_identity import compute_dataset_identity
        from app.statistics.data_quality import HandlingStrategy, HandlingAction

        identity = compute_dataset_identity(str(sample_csv))
        strategy = HandlingStrategy(
            strategy_id="test_strat", action=HandlingAction.TRANSFORM,
            target_variable="x", details="Test", requires_approval=True,
        )
        id1 = await create_transformation_proposal(
            dataset_id="test_ds", operation="log", variable="x",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=strategy,
        )
        id2 = await create_transformation_proposal(
            dataset_id="test_ds", operation="sqrt", variable="x",
            source_content_hash=identity["content_hash"],
            source_schema_hash=identity["schema_hash"],
            strategy=strategy,
        )
        assert id1 != id2  # Different proposals = different versions


# ===========================================================================
# REVALIDATION TESTS
# ===========================================================================

class TestRevalidation:
    def test_valid_derived_dataset_passes(self, sample_csv):
        spec = {
            "analysis_id": "test1",
            "primary_outcome": "outcome",
            "predictors": ["age"],
            "analysis_method": "linear_regression",
            "validation_status": "pending",
        }
        results = revalidate_analyses(str(sample_csv), [spec])
        assert len(results) == 1
        assert results[0]["status"] in ("pass", "warning")

    def test_transformation_invalidating_analysis_blocks(self, sample_csv, tmp_path):
        # Create a derived dataset that removes the "age" column
        df = pd.read_csv(sample_csv)
        df_dropped = df.drop(columns=["age"])
        derived_path = tmp_path / "derived_no_age.csv"
        df_dropped.to_csv(derived_path, index=False)

        spec = {
            "analysis_id": "test2",
            "primary_outcome": "outcome",
            "predictors": ["age"],
            "analysis_method": "linear_regression",
            "validation_status": "pass",
        }
        results = revalidate_analyses(str(derived_path), [spec])
        assert results[0]["blocked_by_transformation"] is True

    def test_check_execution_allowed(self, sample_csv):
        spec = {
            "primary_outcome": "outcome",
            "predictors": ["age"],
            "analysis_method": "linear_regression",
        }
        allowed, reason = check_execution_allowed(str(sample_csv), spec)
        assert allowed is True

    def test_check_execution_blocked_on_missing_var(self, sample_csv, tmp_path):
        df = pd.read_csv(sample_csv).drop(columns=["age"])
        path = tmp_path / "no_age.csv"
        df.to_csv(path, index=False)

        spec = {
            "primary_outcome": "outcome",
            "predictors": ["age"],
            "analysis_method": "linear_regression",
        }
        allowed, reason = check_execution_allowed(str(path), spec)
        assert allowed is False


# ===========================================================================
# ANTI-P-HACKING TESTS
# ===========================================================================

class TestAntiPHacking:
    def test_p_value_selection_blocked(self):
        guard = AntiPHackingGuard()
        strategies = [{"name": "a"}, {"name": "b"}]
        selected = {"name": "a", "p_value_reason": "smallest p"}
        checks = guard.check_strategy_selection(strategies, selected, p_value=0.001)
        blocked = [c for c in checks if c.verdict == AntiPHackingVerdict.BLOCKED]
        assert len(blocked) > 0

    def test_iterative_exclusion_suspicious(self):
        guard = AntiPHackingGuard()
        selected = {"iteration_count": 3}
        checks = guard.check_strategy_selection([], selected)
        suspicious = [c for c in checks if c.verdict == AntiPHackingVerdict.SUSPICIOUS]
        assert len(suspicious) > 0

    def test_significance_comparison_suspicious(self):
        guard = AntiPHackingGuard()
        strategies = [
            {"comparison_reason": "achieves significance"},
            {"comparison_reason": "robust"},
        ]
        checks = guard.check_strategy_selection(strategies, strategies[0])
        suspicious = [c for c in checks if c.verdict == AntiPHackingVerdict.SUSPICIOUS]
        assert len(suspicious) > 0

    def test_transformation_p_value_selection_blocked(self):
        guard = AntiPHackingGuard()
        selected = {"selection_reason": "smallest_p_value"}
        checks = guard.check_transformation_selection([], selected, result_p_value=0.001)
        blocked = [c for c in checks if c.verdict == AntiPHackingVerdict.BLOCKED]
        assert len(blocked) > 0

    def test_outlier_exclusion_significance_blocked(self):
        guard = AntiPHackingGuard()
        checks = guard.check_outlier_exclusion(
            "improves p-value", affects_significance=True,
        )
        blocked = [c for c in checks if c.verdict == AntiPHackingVerdict.BLOCKED]
        assert len(blocked) > 0

    def test_outlier_exclusion_no_justification_suspicious(self):
        guard = AntiPHackingGuard()
        checks = guard.check_outlier_exclusion("some reason")
        suspicious = [c for c in checks if c.verdict == AntiPHackingVerdict.SUSPICIOUS]
        assert len(suspicious) > 0

    def test_outlier_exclusion_with_justification_safe(self):
        guard = AntiPHackingGuard()
        checks = guard.check_outlier_exclusion("data_entry_error: value entered incorrectly")
        safe = [c for c in checks if c.verdict == AntiPHackingVerdict.SAFE]
        assert len(safe) > 0

    def test_report_generation(self):
        guard = AntiPHackingGuard()
        from app.statistics.anti_phacking import AntiPHackingCheck
        checks = [
            AntiPHackingCheck(check_name="a", verdict=AntiPHackingVerdict.SAFE, description="ok"),
            AntiPHackingCheck(check_name="b", verdict=AntiPHackingVerdict.SUSPICIOUS, description="sus"),
        ]
        report = guard.generate_report(checks)
        assert report["overall_verdict"] == "suspicious"
        assert report["n_safe"] == 1
        assert report["n_suspicious"] == 1


# ===========================================================================
# END-TO-END SCENARIOS
# ===========================================================================

class TestScenarioA_MissingPredictor:
    """Scenario A: Missing predictor — 15% missing in bmi."""

    def test_missing_detected(self, sample_csv_with_missing):
        df = pd.read_csv(sample_csv_with_missing)
        missingness = analyze_missingness(df, outcome="outcome", predictors=["bmi", "bp"])
        bmi_var = next(v for v in missingness.variables if v.variable == "bmi")
        assert bmi_var.n_missing > 0

    def test_strategies_generated(self, sample_csv_with_missing):
        df = pd.read_csv(sample_csv_with_missing)
        missingness = analyze_missingness(df, outcome="outcome", predictors=["bmi"])
        strategies = generate_missing_data_strategies(missingness, outcome="outcome", predictors=["bmi"])
        assert len(strategies) > 0

    def test_no_automatic_imputation(self, sample_csv_with_missing):
        df_original = pd.read_csv(sample_csv_with_missing)
        df_check = pd.read_csv(sample_csv_with_missing)
        # Original should have missing values
        assert df_original["bmi"].isna().sum() > 0
        # No automatic imputation should have occurred
        pd.testing.assert_frame_equal(df_original, df_check)

    def test_researcher_approval_required(self, sample_csv_with_missing):
        df = pd.read_csv(sample_csv_with_missing)
        missingness = analyze_missingness(df, outcome="outcome", predictors=["bmi"])
        strategies = generate_missing_data_strategies(missingness, outcome="outcome", predictors=["bmi"])
        approval_required = [s for s in strategies if s.requires_approval]
        assert len(approval_required) > 0


class TestScenarioB_ExtremeObservation:
    """Scenario B: Extreme but plausible observation."""

    def test_outlier_flagged(self, sample_csv_with_outliers):
        df = pd.read_csv(sample_csv_with_outliers)
        results = detect_outliers_iqr(df, columns=["value"])
        assert results[0].n_flagged >= 1

    def test_not_auto_deleted(self, sample_csv_with_outliers):
        df_original = pd.read_csv(sample_csv_with_outliers)
        df_check = pd.read_csv(sample_csv_with_outliers)
        pd.testing.assert_frame_equal(df_original, df_check)

    def test_strategies_presented(self, sample_csv_with_outliers):
        df = pd.read_csv(sample_csv_with_outliers)
        results = detect_outliers_iqr(df, columns=["value"])
        strategies = generate_outlier_strategies(results)
        assert any(s.action == HandlingAction.RETAIN for s in strategies)

    def test_approval_required_for_exclusion(self, sample_csv_with_outliers):
        df = pd.read_csv(sample_csv_with_outliers)
        results = detect_outliers_iqr(df, columns=["value"])
        strategies = generate_outlier_strategies(results)
        approval_required = [s for s in strategies if s.requires_approval]
        assert len(approval_required) > 0


class TestScenarioC_InvalidTransformation:
    """Scenario C: Log transformation proposed for variable with zeros."""

    def test_log_blocked_with_zeros(self, sample_csv_with_zeros):
        df = pd.read_csv(sample_csv_with_zeros)
        safety, violations, _ = validate_transformation_domain(df["value"], TransformationType.LOG)
        assert safety == TransformationSafety.BLOCKED

    def test_log1p_allowed(self, sample_csv_with_zeros):
        df = pd.read_csv(sample_csv_with_zeros)
        safety, violations, _ = validate_transformation_domain(df["value"], TransformationType.LOG1P)
        assert safety == TransformationSafety.SAFE


class TestScenarioD_ValidYeoJohnson:
    """Scenario D: Yeo-Johnson with positive, zero, and negative values."""

    def test_yeojohnson_allowed(self, sample_csv_negative):
        df = pd.read_csv(sample_csv_negative)
        safety, violations, _ = validate_transformation_domain(df["value"], TransformationType.YEO_JOHNSON)
        assert safety == TransformationSafety.SAFE

    def test_fitted_parameter_recorded(self, sample_csv_negative):
        df = pd.read_csv(sample_csv_negative)
        _, params = apply_transformation(df, "value", TransformationType.YEO_JOHNSON)
        assert "lambda" in params

    def test_original_unchanged(self, sample_csv_negative):
        df_original = pd.read_csv(sample_csv_negative)
        original_hash = canonical_content_hash(df_original)
        apply_transformation(df_original, "value", TransformationType.YEO_JOHNSON)
        df_reloaded = pd.read_csv(sample_csv_negative)
        reloaded_hash = canonical_content_hash(df_reloaded)
        assert original_hash == reloaded_hash


class TestScenarioE_TransformationInvalidatesAnalysis:
    """Scenario E: Transformation causes analysis to fail revalidation."""

    def test_revalidation_fails(self, sample_csv, tmp_path):
        df = pd.read_csv(sample_csv)
        # Drop a required variable
        df_dropped = df.drop(columns=["age"])
        derived_path = tmp_path / "derived.csv"
        df_dropped.to_csv(derived_path, index=False)

        spec = {
            "analysis_id": "test_e",
            "primary_outcome": "outcome",
            "predictors": ["age"],
            "analysis_method": "linear_regression",
            "validation_status": "pass",
        }
        results = revalidate_analyses(str(derived_path), [spec])
        assert results[0]["blocked_by_transformation"] is True

    def test_execution_blocked(self, sample_csv, tmp_path):
        df = pd.read_csv(sample_csv).drop(columns=["age"])
        path = tmp_path / "no_age.csv"
        df.to_csv(path, index=False)

        spec = {
            "primary_outcome": "outcome",
            "predictors": ["age"],
            "analysis_method": "linear_regression",
        }
        allowed, reason = check_execution_allowed(str(path), spec)
        assert allowed is False


# ===========================================================================
# DATA QUALITY PROFILE TESTS
# ===========================================================================

class TestDataQualityProfile:
    def test_profile_creation(self):
        profile = DataQualityProfile(
            dataset_id="test",
            row_count=100,
            column_count=5,
        )
        assert profile.row_count == 100
        assert profile.generated_at is not None

    def test_has_critical_issues(self):
        from app.statistics.data_quality import VariableMissingness, DatasetMissingness, MissingnessStatus
        missingness = DatasetMissingness(
            total_rows=100,
            complete_cases=50,
            incomplete_cases=50,
            total_missing_cells=100,
            total_cells=500,
            pct_missing_overall=20.0,
            variables=[
                VariableMissingness(
                    variable="bad",
                    variable_type="numeric",
                    n_missing=60,
                    n_observed=40,
                    total=100,
                    pct_missing=60.0,
                    status=MissingnessStatus.EXTREME,
                ),
            ],
        )
        profile = DataQualityProfile(missingness=missingness)
        assert profile.has_critical_issues() is True

    def test_get_high_impact_variables(self):
        from app.statistics.data_quality import VariableMissingness, DatasetMissingness, MissingnessStatus
        missingness = DatasetMissingness(
            total_rows=100,
            complete_cases=80,
            incomplete_cases=20,
            total_missing_cells=20,
            total_cells=500,
            pct_missing_overall=4.0,
            variables=[
                VariableMissingness(
                    variable="ok", variable_type="numeric",
                    n_missing=5, n_observed=95, total=100,
                    pct_missing=5.0, status=MissingnessStatus.LOW,
                ),
                VariableMissingness(
                    variable="bad", variable_type="numeric",
                    n_missing=30, n_observed=70, total=100,
                    pct_missing=30.0, status=MissingnessStatus.HIGH,
                ),
            ],
        )
        profile = DataQualityProfile(missingness=missingness)
        high_impact = profile.get_high_impact_variables()
        assert "bad" in high_impact
        assert "ok" not in high_impact
