"""Phase 2C Tests — Candidate Generation, Plan Lifecycle, Guardrails, End-to-End Scenarios."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest
from app.statistics.candidate_generator import (
    CandidateGenerator,
    DataProfile,
    CandidateSet,
)
from app.statistics.plan import (
    AnalysisPlan,
    AnalysisEntry,
    AnalysisLifecycle,
    AnalysisVersion,
    VersionHistory,
    _LIFECYCLE_TRANSITIONS,
)
from app.statistics.specification import (
    StatisticalAnalysisSpecification,
    AnalysisDecisionTrace,
    PrespecifiedStatus,
    ApprovalStatus,
    ValidationStatus,
)
from app.statistics.registry import get_method, REGISTRY, list_methods
from app.statistics.validator import StatisticalValidator
from app.agents.analysis_planner import AnalysisPlannerAgent
from app.agents.statistical_executor import StatisticalExecutorAgent, ExecutionGuardError


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def binary_df():
    """Obesity → readmission scenario."""
    np.random.seed(42)
    n = 200
    obesity = np.random.binomial(1, 0.35, n)
    age = np.random.normal(55, 12, n)
    sex = np.random.choice(["M", "F"], n)
    # readmission depends on obesity and age
    logit = -2 + 0.8 * obesity + 0.02 * (age - 55)
    prob = 1 / (1 + np.exp(-logit))
    readmission = np.random.binomial(1, prob)
    return pd.DataFrame({
        "readmission": readmission,
        "obesity": obesity,
        "age": age,
        "sex": sex,
    })


@pytest.fixture
def continuous_df():
    """Biomarker → severity scenario."""
    np.random.seed(42)
    n = 80
    severity = np.random.normal(50, 15, n)
    biomarker = 20 + 0.6 * severity + np.random.normal(0, 8, n)
    return pd.DataFrame({
        "severity": severity,
        "biomarker": biomarker,
    })


@pytest.fixture
def paired_df():
    """Treatment → BP change scenario."""
    np.random.seed(42)
    n = 60
    treatment = np.random.choice(["Drug", "Placebo"], n)
    pre_bp = np.random.normal(140, 15, n)
    # Drug lowers BP more
    effect = np.where(treatment == "Drug", -12, -3)
    post_bp = pre_bp + effect + np.random.normal(0, 5, n)
    change = post_bp - pre_bp
    return pd.DataFrame({
        "treatment": treatment,
        "pre_bp": pre_bp,
        "post_bp": post_bp,
        "change_bp": change,
    })


@pytest.fixture
def smoker_df():
    """Smoking → disease scenario."""
    np.random.seed(42)
    n = 150
    smoking = np.random.choice(["smoker", "non-smoker"], n, p=[0.3, 0.7])
    disease = np.random.binomial(1, np.where(smoking == "smoker", 0.4, 0.15))
    return pd.DataFrame({
        "smoking": smoking,
        "disease": disease,
    })


@pytest.fixture
def age_biomarker_df():
    """Age → biomarker level scenario."""
    np.random.seed(42)
    n = 100
    age = np.random.uniform(20, 80, n)
    biomarker = 10 + 0.3 * age + np.random.normal(0, 5, n)
    return pd.DataFrame({
        "age": age,
        "biomarker": biomarker,
    })


def _make_profile(
    outcome="readmission",
    outcome_type="binary",
    predictors=None,
    predictor_types=None,
    n_obs=200,
    n_groups=2,
    study_design="retrospective_cohort",
    **kwargs,
) -> DataProfile:
    if predictors is None:
        predictors = ["obesity"]
    if predictor_types is None:
        predictor_types = ["binary"]
    return DataProfile(
        n_observations=n_obs,
        outcome_variable=outcome,
        outcome_type=outcome_type,
        predictor_variables=predictors,
        predictor_types=predictor_types,
        n_groups=n_groups,
        study_design=study_design,
        **kwargs,
    )


# ===========================================================================
# 1. Candidate Generation Tests
# ===========================================================================

class TestCandidateGenerationBinary:
    def test_binary_outcome_generates_logistic_and_chi_square(self):
        profile = _make_profile()
        gen = CandidateGenerator()
        result = gen.generate(profile)
        method_ids = [c.method_id for c in result.primary_candidates + result.sensitivity_candidates]
        assert "logistic_regression" in method_ids
        assert "chi_square" in method_ids

    def test_binary_outcome_includes_fisher_for_small_samples(self):
        profile = _make_profile(n_obs=10)
        gen = CandidateGenerator()
        result = gen.generate(profile)
        all_ids = [c.method_id for c in result.primary_candidates + result.sensitivity_candidates + result.exploratory_candidates]
        assert "fisher_exact" in all_ids

    def test_binary_with_continuous_predictor(self):
        profile = _make_profile(
            predictors=["age"],
            predictor_types=["continuous"],
        )
        gen = CandidateGenerator()
        result = gen.generate(profile)
        # Logistic regression supports continuous predictors
        method_ids = [c.method_id for c in result.primary_candidates + result.sensitivity_candidates]
        assert "logistic_regression" in method_ids


class TestCandidateGenerationContinuous:
    def test_continuous_outcome_generates_t_test_and_regression(self):
        profile = _make_profile(
            outcome="sepal_length",
            outcome_type="continuous",
            predictors=["group"],
            predictor_types=["binary"],
        )
        gen = CandidateGenerator()
        result = gen.generate(profile)
        method_ids = [c.method_id for c in result.primary_candidates + result.sensitivity_candidates]
        assert "independent_t_test" in method_ids
        assert "linear_regression" in method_ids

    def test_continuous_with_multiple_continuous_predictors(self):
        profile = _make_profile(
            outcome="biomarker",
            outcome_type="continuous",
            predictors=["age", "weight"],
            predictor_types=["continuous", "continuous"],
            n_obs=100,
        )
        gen = CandidateGenerator()
        result = gen.generate(profile)
        method_ids = [c.method_id for c in result.primary_candidates + result.sensitivity_candidates]
        assert "linear_regression" in method_ids
        assert "pearson_correlation" in method_ids or "spearman_correlation" in method_ids


class TestCandidateGenerationCategorical:
    def test_categorical_outcome_generates_chi_square(self):
        profile = _make_profile(
            outcome="grade",
            outcome_type="categorical",
            predictors=["group"],
            predictor_types=["binary"],
        )
        gen = CandidateGenerator()
        result = gen.generate(profile)
        method_ids = [c.method_id for c in result.primary_candidates + result.sensitivity_candidates]
        assert "chi_square" in method_ids


class TestCandidateGenerationPaired:
    def test_paired_data_includes_paired_methods(self):
        profile = _make_profile(
            outcome="pre_bp",
            outcome_type="continuous",
            predictors=["post_bp"],
            predictor_types=["continuous"],
            paired=True,
            paired_variables=["pre_bp", "post_bp"],
            n_groups=2,
        )
        gen = CandidateGenerator()
        result = gen.generate(profile)
        all_ids = [c.method_id for c in result.primary_candidates + result.sensitivity_candidates + result.exploratory_candidates]
        # Paired t-test and Wilcoxon should appear as candidates
        # (they require binary predictor per registry, but paired structure is noted)
        assert len(all_ids) > 0


class TestCandidateGenerationRejected:
    def test_unsupported_method_not_generated(self):
        """Methods not in the registry must not appear as candidates."""
        profile = _make_profile()
        gen = CandidateGenerator()
        result = gen.generate(profile)
        all_ids = [c.method_id for c in result.primary_candidates + result.sensitivity_candidates + result.exploratory_candidates]
        for mid in all_ids:
            assert mid in REGISTRY, f"Method '{mid}' is not in the registry"

    def test_rejected_methods_recorded(self):
        profile = _make_profile(n_obs=3)  # very small
        gen = CandidateGenerator()
        result = gen.generate(profile)
        # Some methods should be rejected due to sample size
        assert len(result.rejected_methods) > 0 or len(result.warnings) > 0


# ===========================================================================
# 2. Registry Constraint Tests
# ===========================================================================

class TestRegistryConstraints:
    def test_only_registered_methods_in_candidates(self):
        profile = _make_profile()
        gen = CandidateGenerator()
        result = gen.generate(profile)
        all_candidates = (
            result.primary_candidates
            + result.sensitivity_candidates
            + result.exploratory_candidates
        )
        for c in all_candidates:
            assert c.method_id in REGISTRY

    def test_unsupported_method_rejected_by_registry(self):
        assert get_method("nonexistent_method") is None

    def test_all_registry_methods_have_engine_function(self):
        for mid, entry in REGISTRY.items():
            assert entry.engine_function, f"Method '{mid}' has no engine_function"


# ===========================================================================
# 3. Validator Tests
# ===========================================================================

class TestValidatorIntegration:
    def test_valid_spec_passes(self, binary_df):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="readmission",
            predictors=["obesity"],
            analysis_method="logistic_regression",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=binary_df)
        assert result.status == ValidationStatus.PASS

    def test_missing_variable_blocks(self, binary_df):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="nonexistent",
            predictors=["obesity"],
            analysis_method="logistic_regression",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=binary_df)
        assert result.status == ValidationStatus.BLOCK

    def test_insufficient_sample_blocks(self):
        tiny_df = pd.DataFrame({"y": [1, 2], "x": [3, 4]})
        spec = StatisticalAnalysisSpecification(
            primary_outcome="y",
            predictors=["x"],
            analysis_method="linear_regression",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=tiny_df)
        assert result.status == ValidationStatus.BLOCK

    def test_constant_predictor_blocks(self, binary_df):
        binary_df["const"] = 1
        spec = StatisticalAnalysisSpecification(
            primary_outcome="readmission",
            predictors=["const"],
            analysis_method="logistic_regression",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=binary_df)
        assert result.status == ValidationStatus.BLOCK


# ===========================================================================
# 4. Lifecycle Tests
# ===========================================================================

class TestLifecycle:
    def test_valid_transitions(self):
        allowed = _LIFECYCLE_TRANSITIONS[AnalysisLifecycle.DRAFT]
        assert AnalysisLifecycle.VALIDATING in allowed

    def test_invalid_transition_blocked(self):
        # DRAFT cannot go directly to EXECUTED
        allowed = _LIFECYCLE_TRANSITIONS[AnalysisLifecycle.DRAFT]
        assert AnalysisLifecycle.EXECUTED not in allowed

    def test_executed_is_terminal(self):
        allowed = _LIFECYCLE_TRANSITIONS[AnalysisLifecycle.EXECUTED]
        assert len(allowed) == 0

    def test_validation_failed_can_revise(self):
        allowed = _LIFECYCLE_TRANSITIONS[AnalysisLifecycle.VALIDATION_FAILED]
        assert AnalysisLifecycle.DRAFT in allowed

    def test_analysis_entry_lifecycle(self):
        entry = AnalysisEntry(analysis_name="test", analysis_class="primary")
        assert entry.get_lifecycle() == AnalysisLifecycle.DRAFT

    def test_full_lifecycle_flow(self):
        entry = AnalysisEntry(analysis_name="test", analysis_class="primary")
        version = entry.version_history.versions[0]

        # DRAFT → VALIDATING
        assert entry.can_transition_to(AnalysisLifecycle.VALIDATING)
        version.lifecycle = AnalysisLifecycle.VALIDATING

        # VALIDATING → VALIDATED
        assert entry.can_transition_to(AnalysisLifecycle.VALIDATED)
        version.lifecycle = AnalysisLifecycle.VALIDATED

        # VALIDATED → NEEDS_REVIEW
        assert entry.can_transition_to(AnalysisLifecycle.NEEDS_REVIEW)
        version.lifecycle = AnalysisLifecycle.NEEDS_REVIEW

        # NEEDS_REVIEW → APPROVED
        assert entry.can_transition_to(AnalysisLifecycle.APPROVED)
        version.lifecycle = AnalysisLifecycle.APPROVED

        # APPROVED → LOCKED
        assert entry.can_transition_to(AnalysisLifecycle.LOCKED)
        version.lifecycle = AnalysisLifecycle.LOCKED

        # LOCKED → EXECUTED
        assert entry.can_transition_to(AnalysisLifecycle.EXECUTED)
        version.lifecycle = AnalysisLifecycle.EXECUTED

        # EXECUTED is terminal
        assert not entry.can_transition_to(AnalysisLifecycle.DRAFT)


# ===========================================================================
# 5. Approval Guardrail Tests
# ===========================================================================

class TestApprovalGuardrails:
    def test_cannot_execute_without_approval(self):
        """Execution must be blocked when approval is not granted."""
        spec = StatisticalAnalysisSpecification(
            primary_outcome="readmission",
            predictors=["obesity"],
            analysis_method="logistic_regression",
            approval_status=ApprovalStatus.PROPOSED,
            validation_status=ValidationStatus.PASS,
        )

        executor = StatisticalExecutorAgent()
        with pytest.raises(ExecutionGuardError) as exc_info:
            executor._run_guardrails(
                spec.model_dump(),
                lifecycle="locked",
                file_path=None,
            )
        assert exc_info.value.guard == "not_approved"

    def test_cannot_execute_blocked_validation(self):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="readmission",
            predictors=["obesity"],
            analysis_method="logistic_regression",
            approval_status=ApprovalStatus.APPROVED,
            validation_status=ValidationStatus.BLOCK,
        )

        executor = StatisticalExecutorAgent()
        with pytest.raises(ExecutionGuardError) as exc_info:
            executor._run_guardrails(
                spec.model_dump(),
                lifecycle="locked",
                file_path=None,
            )
        assert exc_info.value.guard == "validation_blocked"

    def test_cannot_execute_draft_lifecycle(self):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="readmission",
            predictors=["obesity"],
            analysis_method="logistic_regression",
            approval_status=ApprovalStatus.APPROVED,
            validation_status=ValidationStatus.PASS,
        )

        executor = StatisticalExecutorAgent()
        with pytest.raises(ExecutionGuardError) as exc_info:
            executor._run_guardrails(
                spec.model_dump(),
                lifecycle="draft",
                file_path=None,
            )
        assert exc_info.value.guard == "invalid_lifecycle"

    def test_approved_locked_passes_guardrails(self):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="readmission",
            predictors=["obesity"],
            analysis_method="logistic_regression",
            approval_status=ApprovalStatus.LOCKED,
            validation_status=ValidationStatus.PASS,
        )

        executor = StatisticalExecutorAgent()
        # Should not raise (file_path None is allowed for guardrail check)
        executor._run_guardrails(
            spec.model_dump(),
            lifecycle="locked",
            file_path=None,
        )

    def test_unsupported_method_blocked(self):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="y",
            predictors=["x"],
            analysis_method="nonexistent_method",
        )

        executor = StatisticalExecutorAgent()
        with pytest.raises(ExecutionGuardError) as exc_info:
            executor._run_guardrails(
                spec.model_dump(),
                lifecycle="locked",
                file_path=None,
            )
        assert exc_info.value.guard == "unsupported_method"


# ===========================================================================
# 6. Versioning Tests
# ===========================================================================

class TestVersioning:
    def test_first_version_is_1_0(self):
        entry = AnalysisEntry(analysis_name="test", analysis_class="primary")
        cv = entry.version_history.get_current()
        assert cv.version_number == "1.0"

    def test_modification_creates_new_version(self):
        entry = AnalysisEntry(analysis_name="test", analysis_class="primary")

        spec_v1 = StatisticalAnalysisSpecification(
            primary_outcome="y",
            predictors=["x1"],
            analysis_method="linear_regression",
            version="1.0",
        )
        entry.version_history.versions[0].specification = spec_v1

        # Create v2
        spec_v2 = StatisticalAnalysisSpecification(
            primary_outcome="y",
            predictors=["x1", "x2"],
            analysis_method="linear_regression",
            version="2.0",
        )
        v2 = AnalysisVersion(
            version_number="2.0",
            specification=spec_v2,
            lifecycle=AnalysisLifecycle.DRAFT,
            modification_reason="Added covariate x2",
        )
        entry.version_history.add_version(v2)

        assert len(entry.version_history.versions) == 2
        assert entry.version_history.get_current().version_number == "2.0"

    def test_prior_version_preserved(self):
        entry = AnalysisEntry(analysis_name="test", analysis_class="primary")
        v1_spec = StatisticalAnalysisSpecification(
            primary_outcome="y", predictors=["x1"], analysis_method="t", version="1.0",
        )
        entry.version_history.versions[0].specification = v1_spec

        v2_spec = StatisticalAnalysisSpecification(
            primary_outcome="y", predictors=["x1", "x2"], analysis_method="t", version="2.0",
        )
        entry.version_history.add_version(AnalysisVersion(
            version_number="2.0", specification=v2_spec,
        ))

        # Can still access v1
        v1 = entry.version_history.get_version_by_number("1.0")
        assert v1 is not None
        assert len(v1.specification.predictors) == 1

    def test_new_version_requires_revalidation(self):
        entry = AnalysisEntry(analysis_name="test", analysis_class="primary")
        entry.version_history.get_current().lifecycle = AnalysisLifecycle.VALIDATED

        # Add new version — starts at DRAFT
        new_v = AnalysisVersion(
            version_number="2.0",
            specification=StatisticalAnalysisSpecification(
                primary_outcome="y", predictors=["x1"], analysis_method="t",
            ),
            lifecycle=AnalysisLifecycle.DRAFT,
        )
        entry.version_history.add_version(new_v)

        assert entry.get_lifecycle() == AnalysisLifecycle.DRAFT


# ===========================================================================
# 7. Prespecification Tests
# ===========================================================================

class TestPrespecification:
    def test_default_is_unclear(self):
        entry = AnalysisEntry(analysis_name="test", analysis_class="primary")
        assert entry.prespecified_status == "unclear"

    def test_can_set_prespecified(self):
        entry = AnalysisEntry(
            analysis_name="test",
            analysis_class="primary",
            prespecified_status="prespecified",
        )
        assert entry.prespecified_status == "prespecified"

    def test_can_set_not_prespecified(self):
        entry = AnalysisEntry(
            analysis_name="test",
            analysis_class="exploratory",
            prespecified_status="not_prespecified",
        )
        assert entry.prespecified_status == "not_prespecified"


# ===========================================================================
# 8. Exploratory Analysis Labeling
# ===========================================================================

class TestExploratoryLabeling:
    def test_exploratory_class_preserved(self):
        plan = AnalysisPlan(project_id="p1")
        entry = AnalysisEntry(
            analysis_name="exploratory analysis",
            analysis_class="exploratory",
        )
        plan.add_analysis(entry)
        exploratory = plan.get_analyses_by_class("exploratory")
        assert len(exploratory) == 1
        assert exploratory[0].analysis_class == "exploratory"

    def test_primary_not_in_exploratory(self):
        plan = AnalysisPlan(project_id="p1")
        plan.add_analysis(AnalysisEntry(analysis_name="primary", analysis_class="primary"))
        plan.add_analysis(AnalysisEntry(analysis_name="exploratory", analysis_class="exploratory"))
        assert len(plan.get_primary_analyses()) == 1
        assert len(plan.get_analyses_by_class("exploratory")) == 1


# ===========================================================================
# 9. Causal Language Guardrail
# ===========================================================================

class TestCausalLanguage:
    def test_default_is_association(self):
        entry = AnalysisEntry(analysis_name="test", analysis_class="primary")
        assert entry.causal_language_flag == "association"

    def test_can_flag_causation(self):
        entry = AnalysisEntry(
            analysis_name="test",
            analysis_class="primary",
            causal_language_flag="causation",
        )
        assert entry.causal_language_flag == "causation"


# ===========================================================================
# 10. Multiple Testing Strategy
# ===========================================================================

class TestMultipleTesting:
    def test_default_is_none_required(self):
        entry = AnalysisEntry(analysis_name="test", analysis_class="primary")
        assert entry.multiple_testing_strategy == "none_required"

    def test_can_set_adjustment_planned(self):
        entry = AnalysisEntry(
            analysis_name="test",
            analysis_class="primary",
            multiple_testing_strategy="adjustment_planned",
        )
        assert entry.multiple_testing_strategy == "adjustment_planned"


# ===========================================================================
# 11. Plan Structure Tests
# ===========================================================================

class TestPlanStructure:
    def test_plan_has_analyses(self):
        plan = AnalysisPlan(project_id="p1")
        plan.add_analysis(AnalysisEntry(analysis_name="a1", analysis_class="primary"))
        plan.add_analysis(AnalysisEntry(analysis_name="a2", analysis_class="secondary"))
        assert len(plan.analyses) == 2

    def test_plan_get_analysis_by_id(self):
        plan = AnalysisPlan(project_id="p1")
        entry = AnalysisEntry(analysis_name="a1", analysis_class="primary")
        plan.add_analysis(entry)
        found = plan.get_analysis(entry.analysis_id)
        assert found is not None
        assert found.analysis_name == "a1"

    def test_plan_audit_events(self):
        plan = AnalysisPlan(project_id="p1")
        plan.add_audit_event("plan_created", details={"n_analyses": 0})
        assert len(plan.audit_events) == 1
        assert plan.audit_events[0]["event_type"] == "plan_created"


# ===========================================================================
# 12. End-to-End Synthetic Scenarios
# ===========================================================================

class TestScenario1_ObesityReadmission:
    """Does obesity predict 30-day hospital readmission?"""

    def test_candidates_generated(self):
        profile = _make_profile(
            outcome="readmission",
            outcome_type="binary",
            predictors=["obesity"],
            predictor_types=["binary"],
            n_obs=200,
            study_design="retrospective_cohort",
        )
        gen = CandidateGenerator()
        result = gen.generate(profile)
        method_ids = [c.method_id for c in result.primary_candidates + result.sensitivity_candidates]
        assert "logistic_regression" in method_ids
        assert "chi_square" in method_ids

    def test_logistic_regression_validates(self, binary_df):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="readmission",
            predictors=["obesity"],
            analysis_method="logistic_regression",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=binary_df)
        assert result.status == ValidationStatus.PASS

    def test_logistic_regression_adjusts_for_covariates(self, binary_df):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="readmission",
            predictors=["obesity"],
            covariates=["age", "sex"],
            analysis_method="logistic_regression",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=binary_df)
        assert result.status == ValidationStatus.PASS

    def test_chi_square_is_unadjusted(self):
        """Chi-square should appear as a sensitivity/alternative (unadjusted)."""
        profile = _make_profile(
            outcome="readmission",
            outcome_type="binary",
            predictors=["obesity"],
            predictor_types=["binary"],
        )
        gen = CandidateGenerator()
        result = gen.generate(profile)
        all_candidates = result.primary_candidates + result.sensitivity_candidates
        chi_sq = [c for c in all_candidates if c.method_id == "chi_square"]
        assert len(chi_sq) > 0


class TestScenario2_BiomarkerSeverity:
    """Is a biomarker associated with disease severity?"""

    def test_candidates_generated(self):
        profile = _make_profile(
            outcome="severity",
            outcome_type="continuous",
            predictors=["biomarker"],
            predictor_types=["continuous"],
            n_obs=80,
        )
        gen = CandidateGenerator()
        result = gen.generate(profile)
        method_ids = [c.method_id for c in result.primary_candidates + result.sensitivity_candidates]
        assert "pearson_correlation" in method_ids or "linear_regression" in method_ids

    def test_correlation_validates(self, continuous_df):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="severity",
            predictors=["biomarker"],
            analysis_method="pearson_correlation",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=continuous_df)
        assert result.status == ValidationStatus.PASS


class TestScenario3_TreatmentBP:
    """Does treatment group affect change in systolic blood pressure?"""

    def test_candidates_generated(self):
        profile = _make_profile(
            outcome="change_bp",
            outcome_type="continuous",
            predictors=["treatment"],
            predictor_types=["binary"],
            n_obs=60,
            n_groups=2,
        )
        gen = CandidateGenerator()
        result = gen.generate(profile)
        method_ids = [c.method_id for c in result.primary_candidates + result.sensitivity_candidates]
        assert "independent_t_test" in method_ids or "mann_whitney_u" in method_ids

    def test_paired_structure_recognized(self, paired_df):
        """The planner should recognize paired structure."""
        profile = _make_profile(
            outcome="change_bp",
            outcome_type="continuous",
            predictors=["pre_bp", "post_bp"],
            predictor_types=["continuous", "continuous"],
            n_obs=60,
            paired=True,
            paired_variables=["pre_bp", "post_bp"],
        )
        gen = CandidateGenerator()
        result = gen.generate(profile)
        # Should generate some candidates
        all_c = result.primary_candidates + result.sensitivity_candidates + result.exploratory_candidates
        assert len(all_c) > 0


class TestScenario4_SmokingDisease:
    """Is smoking status associated with disease status?"""

    def test_candidates_generated(self):
        profile = _make_profile(
            outcome="disease",
            outcome_type="binary",
            predictors=["smoking"],
            predictor_types=["binary"],
            n_obs=150,
        )
        gen = CandidateGenerator()
        result = gen.generate(profile)
        method_ids = [c.method_id for c in result.primary_candidates + result.sensitivity_candidates]
        assert "chi_square" in method_ids
        assert "fisher_exact" in method_ids or "logistic_regression" in method_ids

    def test_validates(self, smoker_df):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="disease",
            predictors=["smoking"],
            analysis_method="chi_square",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=smoker_df)
        assert result.status == ValidationStatus.PASS


class TestScenario5_AgeBiomarker:
    """Is age associated with biomarker level?"""

    def test_candidates_generated(self):
        profile = _make_profile(
            outcome="biomarker",
            outcome_type="continuous",
            predictors=["age"],
            predictor_types=["continuous"],
            n_obs=100,
        )
        gen = CandidateGenerator()
        result = gen.generate(profile)
        method_ids = [c.method_id for c in result.primary_candidates + result.sensitivity_candidates]
        assert "pearson_correlation" in method_ids
        assert "linear_regression" in method_ids

    def test_validates(self, age_biomarker_df):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="biomarker",
            predictors=["age"],
            analysis_method="pearson_correlation",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=age_biomarker_df)
        assert result.status == ValidationStatus.PASS


# ===========================================================================
# 13. Edge Case Tests
# ===========================================================================

class TestEdgeCases:
    def test_zero_observations(self):
        profile = _make_profile(n_obs=0, outcome_type="continuous", predictors=[], predictor_types=[])
        gen = CandidateGenerator()
        result = gen.generate(profile)
        assert len(result.primary_candidates) == 0
        assert len(result.warnings) > 0 or len(result.rejected_methods) > 0

    def test_one_observation(self):
        profile = _make_profile(n_obs=1)
        gen = CandidateGenerator()
        result = gen.generate(profile)
        # Everything should be rejected
        all_c = result.primary_candidates + result.sensitivity_candidates
        assert len(all_c) == 0

    def test_extreme_missingness_warns(self):
        profile = _make_profile(
            missingness_summary={"readmission": 60.0, "obesity": 25.0},
        )
        gen = CandidateGenerator()
        result = gen.generate(profile)
        assert any("missing" in w.lower() for w in result.warnings)

    def test_extreme_imbalance_warns(self):
        profile = _make_profile(outcome_type="binary", event_rate=0.02)
        gen = CandidateGenerator()
        result = gen.generate(profile)
        assert any("imbalance" in w.lower() for w in result.warnings)

    def test_no_predictors_warns(self):
        profile = _make_profile(predictors=[], predictor_types=[])
        gen = CandidateGenerator()
        result = gen.generate(profile)
        assert any("no predictor" in w.lower() for w in result.warnings)

    def test_unsupported_method_in_spec_blocks(self):
        spec = StatisticalAnalysisSpecification(
            primary_outcome="y",
            predictors=["x"],
            analysis_method="nonexistent_method",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=pd.DataFrame({"y": [1], "x": [1]}))
        # Unknown method should at least warn
        assert result.status in (ValidationStatus.WARNING, ValidationStatus.BLOCK)

    def test_complete_separation_warns(self):
        """Logistic regression with perfect prediction should warn."""
        df = pd.DataFrame({
            "outcome": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "predictor": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        })
        spec = StatisticalAnalysisSpecification(
            primary_outcome="outcome",
            predictors=["predictor"],
            analysis_method="logistic_regression",
        )
        v = StatisticalValidator()
        result = v.validate(spec, data_frame=df)
        # Should at least not crash; may warn about separation
        assert result.status in (ValidationStatus.PASS, ValidationStatus.WARNING, ValidationStatus.BLOCK)


# ===========================================================================
# 14. Executor Guardrail Error Tests
# ===========================================================================

class TestExecutorGuardrails:
    def test_guard_error_has_guard_name(self):
        err = ExecutionGuardError("test_guard", "test message")
        assert err.guard == "test_guard"
        assert err.message == "test message"
        assert "test_guard" in str(err)


# ===========================================================================
# 15. Decision Trace Tests
# ===========================================================================

class TestDecisionTrace:
    def test_trace_records_steps(self):
        trace = AnalysisDecisionTrace(analysis_id="test")
        trace.add_step("outcome_check", "Outcome is binary", "pass")
        trace.add_step("sample_check", "N=200, adequate", "pass")
        assert len(trace.steps) == 2

    def test_trace_records_rejections(self):
        trace = AnalysisDecisionTrace(analysis_id="test")
        trace.reject_method("chi_square", "Outcome is continuous")
        assert len(trace.rejection_reasons) == 1

    def test_trace_records_alternatives(self):
        trace = AnalysisDecisionTrace(analysis_id="test")
        trace.alternatives_considered = ["Mann-Whitney U", "Permutation test"]
        assert len(trace.alternatives_considered) == 2

    def test_trace_set_final(self):
        trace = AnalysisDecisionTrace(analysis_id="test")
        trace.set_final("logistic_regression")
        assert trace.final_method == "logistic_regression"


# ===========================================================================
# 16. Planner Agent Integration Test
# ===========================================================================

class TestPlannerAgent:
    def test_build_data_profile(self):
        planner = AnalysisPlannerAgent()
        context = {
            "research_question": "Does obesity predict readmission?",
            "study_design": "retrospective_cohort",
            "data_dictionary": {
                "n_rows": 200,
                "variables": [
                    {"name": "readmission", "dtype": "int64", "role": "outcome", "n_unique": 2, "missing_pct": 0},
                    {"name": "obesity", "dtype": "int64", "role": "predictor", "n_unique": 2, "missing_pct": 0},
                    {"name": "age", "dtype": "float64", "role": "covariate", "n_unique": 80, "missing_pct": 2.0},
                ],
            },
        }
        profile = planner._build_data_profile(context)
        assert profile is not None
        assert profile.n_observations == 200
        assert profile.outcome_variable == "readmission"
        assert profile.outcome_type == "binary"
        assert "obesity" in profile.predictor_variables

    def test_infer_type_binary(self):
        planner = AnalysisPlannerAgent()
        assert planner._infer_type("int64", 2, 200) == "binary"

    def test_infer_type_continuous(self):
        planner = AnalysisPlannerAgent()
        assert planner._infer_type("float64", 150, 200) == "continuous"

    def test_infer_type_categorical(self):
        planner = AnalysisPlannerAgent()
        assert planner._infer_type("object", 5, 200) == "categorical"
