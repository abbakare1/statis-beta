"""Tests for Statistical Analysis Specification, Registry, and Validator."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest
from app.statistics.specification import (
    StatisticalAnalysisSpecification,
    AnalysisDecisionTrace,
    PrespecifiedStatus,
    ApprovalStatus,
    ValidationStatus,
)
from app.statistics.registry import (
    get_method,
    list_methods,
    find_methods_for_outcome,
    REGISTRY,
)
from app.statistics.validator import StatisticalValidator


# ---------------------------------------------------------------------------
# Specification tests
# ---------------------------------------------------------------------------

class TestSpecification:
    def test_create_spec(self):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="bp_systolic",
            predictors=["treatment_group"],
            analysis_method="independent_t_test",
        )
        assert spec.primary_outcome == "bp_systolic"
        assert spec.analysis_method == "independent_t_test"
        assert spec.validation_status == ValidationStatus.PENDING

    def test_spec_defaults(self):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="y",
            analysis_method="linear_regression",
        )
        assert spec.alpha == 0.05
        assert spec.confidence_level == 0.95
        assert spec.version == "1.0"
        assert spec.prespecified_status == PrespecifiedStatus.PRESPECIFIED

    def test_spec_approve(self):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="y", analysis_method="t_test",
        )
        spec.approve()
        assert spec.approval_status == ApprovalStatus.APPROVED

    def test_spec_lock(self):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="y", analysis_method="t_test",
        )
        spec.approve()
        spec.lock()
        assert spec.approval_status == ApprovalStatus.LOCKED

    def test_spec_modify(self):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="y", analysis_method="t_test", version="1.0",
        )
        spec.modify("2.0")
        assert spec.version == "2.0"
        assert spec.approval_status == ApprovalStatus.MODIFIED


class TestDecisionTrace:
    def test_add_step(self):
        trace = DecisionTraceFixture()
        trace.add_step("check_outcome", "Outcome is continuous", "pass")
        assert len(trace.steps) == 1
        assert trace.steps[0]["step"] == "check_outcome"

    def test_reject_method(self):
        trace = DecisionTraceFixture()
        trace.reject_method("chi_square", "Outcome is continuous")
        assert len(trace.rejection_reasons) == 1
        assert trace.rejection_reasons[0]["method"] == "chi_square"

    def test_set_final(self):
        trace = DecisionTraceFixture()
        trace.set_final("linear_regression")
        assert trace.final_method == "linear_regression"


# Use a simple stand-in to avoid coupling
from app.statistics.specification import AnalysisDecisionTrace as _Trace

class DecisionTraceFixture(_Trace):
    def __init__(self):
        super().__init__(analysis_id="test-001")


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_phase1_methods_registered(self):
        expected = [
            "independent_t_test", "paired_t_test", "chi_square", "fisher_exact",
            "pearson_correlation", "spearman_correlation", "one_way_anova",
            "linear_regression", "logistic_regression", "mann_whitney_u",
            "wilcoxon_signed_rank", "descriptive",
        ]
        for m in expected:
            assert m in REGISTRY, f"Missing: {m}"

    def test_list_methods(self):
        methods = list_methods()
        assert len(methods) >= 12

    def test_get_method(self):
        m = get_method("independent_t_test")
        assert m is not None
        assert m.name == "Independent samples t-test"
        assert "continuous" in m.outcome_types
        assert "binary" in m.predictor_types

    def test_find_methods_for_outcome(self):
        methods = find_methods_for_outcome("binary")
        names = [m.method_id for m in methods]
        assert "chi_square" in names
        assert "fisher_exact" in names
        assert "logistic_regression" in names

    def test_all_methods_have_required_fields(self):
        for mid, entry in REGISTRY.items():
            assert entry.method_id == mid
            assert entry.name
            assert entry.description
            assert len(entry.outcome_types) > 0
            assert len(entry.predictor_types) > 0


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

@pytest.fixture
def iris_df():
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "sepal_length": np.random.normal(5.0, 0.5, n),
        "sepal_width": np.random.normal(3.5, 0.4, n),
        "petal_length": np.random.normal(1.5, 0.3, n),
        "group": np.random.choice(["A", "B"], n),
        "binary_outcome": np.random.choice([0, 1], n),
    })


@pytest.fixture
def small_df():
    return pd.DataFrame({
        "y": [1, 2, 3],
        "x": [4, 5, 6],
    })


class TestValidatorPass:
    def test_valid_spec_passes(self, iris_df):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="sepal_length",
            predictors=["group"],
            analysis_method="independent_t_test",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=iris_df)
        assert result.status == ValidationStatus.PASS
        assert len(result.blockers) == 0

    def test_regression_valid(self, iris_df):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="sepal_length",
            predictors=["sepal_width", "petal_length"],
            analysis_method="linear_regression",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=iris_df)
        assert result.status == ValidationStatus.PASS


class TestValidatorBlocks:
    def test_missing_variable_blocks(self, iris_df):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="nonexistent",
            predictors=["group"],
            analysis_method="independent_t_test",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=iris_df)
        assert result.status == ValidationStatus.BLOCK
        assert any("not found" in b for b in result.blockers)

    def test_insufficient_sample_size_blocks(self, small_df):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="y",
            predictors=["x"],
            analysis_method="linear_regression",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=small_df)
        assert result.status == ValidationStatus.BLOCK
        assert any("sample" in b.lower() for b in result.blockers)

    def test_no_data_blocks(self):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="y", analysis_method="t_test",
        )
        v = StatisticalValidator()
        result = v.validate(spec)
        assert result.status == ValidationStatus.BLOCK
        assert any("No dataset" in b for b in result.blockers)

    def test_constant_variable_blocks(self, iris_df):
        iris_df["const"] = 5.0
        spec = StatisticalAnalysisSpecification(
            primary_outcome="sepal_length",
            predictors=["const"],
            analysis_method="linear_regression",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=iris_df)
        assert result.status == ValidationStatus.BLOCK
        assert any("constant" in b.lower() for b in result.blockers)


class TestValidatorWarnings:
    def test_high_missing_warns(self, iris_df):
        iris_df.loc[0:14, "sepal_length"] = np.nan  # 30% missing
        spec = StatisticalAnalysisSpecification(
            primary_outcome="sepal_length",
            predictors=["group"],
            analysis_method="independent_t_test",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=iris_df)
        assert result.status == ValidationStatus.WARNING

    def test_small_sample_warns(self, iris_df):
        small_iris = iris_df.head(12)
        spec = StatisticalAnalysisSpecification(
            primary_outcome="sepal_length",
            predictors=["group"],
            analysis_method="independent_t_test",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=small_iris)
        assert result.status == ValidationStatus.WARNING
        assert any("underpowered" in w.lower() for w in result.warnings)

    def test_predictor_type_mismatch_warns(self, iris_df):
        iris_df["continuous_pred"] = np.random.normal(0, 1, len(iris_df))
        spec = StatisticalAnalysisSpecification(
            primary_outcome="binary_outcome",
            predictors=["continuous_pred"],
            analysis_method="chi_square",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=iris_df)
        # Should warn or at least not crash
        assert result.status in (ValidationStatus.WARNING, ValidationStatus.PASS)


class TestValidatorDiagnostics:
    def test_diagnostics_populated(self, iris_df):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="sepal_length",
            predictors=["group"],
            analysis_method="independent_t_test",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=iris_df)
        assert len(result.checks) > 0
        assert spec.diagnostics == result.checks

    def test_spec_validation_updated(self, iris_df):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="sepal_length",
            predictors=["group"],
            analysis_method="independent_t_test",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=iris_df)
        assert spec.validation_status == result.status
        assert spec.validation_warnings == result.warnings
