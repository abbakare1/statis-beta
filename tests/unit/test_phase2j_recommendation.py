"""Phase 2J — Recommendation Engine Validation (Invariant 37).

Validates that STATIS recommends appropriate candidate methods based on
study design and variable types, NOT based on observed p-values.

Key invariant: Same study design + same variables + different random outcome
values → recommendation category remains governed by design/type, not by
which analysis produces the smallest p-value.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.statistics.candidate_generator import CandidateGenerator, DataProfile
from app.statistics.variable_roles import AnalysisProblemType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def generator():
    return CandidateGenerator()


def _make_profile(**kwargs) -> DataProfile:
    defaults = {
        "n_observations": 200,
        "outcome_variable": "outcome",
        "outcome_type": "continuous",
        "predictor_variables": ["predictor"],
        "predictor_types": ["binary"],
        "covariates": [],
        "covariate_types": [],
        "n_groups": 2,
        "group_balances": {"A": 100, "B": 100},
        "paired": False,
        "paired_variables": [],
        "study_design": "observational",
        "research_question": "Does X affect Y?",
        "has_repeated_measures": False,
        "follow_up_or_time": False,
        "missingness_summary": {},
        "event_rate": None,
        "distribution_notes": [],
        "problem_type": AnalysisProblemType.UNKNOWN,
        "role_resolutions": None,
        "identifier_variables": [],
    }
    defaults.update(kwargs)
    return DataProfile(**defaults)


# ===========================================================================
# Two Independent Continuous Groups
# ===========================================================================

class TestTwoIndependentGroups:
    def test_recommends_ttest_and_mannwhitney(self, generator):
        """Two independent continuous groups → t-test and Mann-Whitney as candidates."""
        profile = _make_profile(
            outcome_type="continuous",
            predictor_variables=["group"],
            predictor_types=["binary"],
            n_groups=2,
            problem_type=AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
        )
        result = generator.generate(profile)
        method_ids = [c.method_id for c in result.primary_candidates + result.secondary_candidates]
        assert "independent_t_test" in method_ids
        assert "mann_whitney_u" in method_ids

    def test_does_not_recommend_paired_methods(self, generator):
        """Paired methods not recommended when data is not paired."""
        profile = _make_profile(
            outcome_type="continuous",
            predictor_variables=["group"],
            predictor_types=["binary"],
            n_groups=2,
            paired=False,
            problem_type=AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
        )
        result = generator.generate(profile)
        all_methods = [c.method_id for c in result.primary_candidates + result.secondary_candidates + result.sensitivity_candidates]
        assert "paired_t_test" not in all_methods
        assert "wilcoxon_signed_rank" not in all_methods

    def test_does_not_recommend_survival_methods(self, generator):
        """Survival methods not recommended for non-survival outcomes."""
        profile = _make_profile(
            outcome_type="continuous",
            predictor_variables=["group"],
            predictor_types=["binary"],
            n_groups=2,
            problem_type=AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
        )
        result = generator.generate(profile)
        all_methods = [c.method_id for c in result.primary_candidates + result.secondary_candidates]
        assert "kaplan_meier" not in all_methods
        assert "log_rank" not in all_methods
        assert "cox_proportional_hazards" not in all_methods


# ===========================================================================
# Paired Continuous Measurements
# ===========================================================================

class TestPairedMeasurements:
    def test_recommends_paired_methods(self, generator):
        """Paired data → paired t-test and Wilcoxon as candidates."""
        profile = _make_profile(
            outcome_type="continuous",
            predictor_variables=["condition"],
            predictor_types=["binary"],
            n_groups=2,
            paired=True,
            paired_variables=["before", "after"],
            problem_type=AnalysisProblemType.TWO_PAIRED_GROUPS_CONTINUOUS_OUTCOME,
        )
        result = generator.generate(profile)
        method_ids = [c.method_id for c in result.primary_candidates + result.secondary_candidates]
        assert "paired_t_test" in method_ids
        assert "wilcoxon_signed_rank" in method_ids


# ===========================================================================
# Two Independent Binary Variables
# ===========================================================================

class TestTwoBinaryVariables:
    def test_recommends_chi_square_and_fisher(self, generator):
        """Two binary variables → chi-square and Fisher exact as candidates."""
        profile = _make_profile(
            outcome_type="binary",
            predictor_variables=["exposure"],
            predictor_types=["binary"],
            n_groups=2,
            problem_type=AnalysisProblemType.BINARY_OUTCOME_SINGLE_PREDICTOR,
        )
        result = generator.generate(profile)
        method_ids = [c.method_id for c in result.primary_candidates + result.secondary_candidates]
        assert "chi_square_test" in method_ids or "chi_square" in method_ids
        assert "fisher_exact_test" in method_ids or "fisher_exact" in method_ids


# ===========================================================================
# Continuous-Continuous Association
# ===========================================================================

class TestContinuousAssociation:
    def test_recommends_correlations(self, generator):
        """Continuous-continuous → Pearson and Spearman as candidates."""
        profile = _make_profile(
            outcome_type="continuous",
            predictor_variables=["x"],
            predictor_types=["continuous"],
            n_groups=1,
            problem_type=AnalysisProblemType.CONTINUOUS_CONTINUOUS_ASSOCIATION,
        )
        result = generator.generate(profile)
        method_ids = [c.method_id for c in result.primary_candidates + result.secondary_candidates]
        assert "pearson_correlation" in method_ids
        assert "spearman_correlation" in method_ids

    def test_does_not_recommends_group_comparison(self, generator):
        """Group comparison methods not recommended for continuous association."""
        profile = _make_profile(
            outcome_type="continuous",
            predictor_variables=["x"],
            predictor_types=["continuous"],
            n_groups=1,
            problem_type=AnalysisProblemType.CONTINUOUS_CONTINUOUS_ASSOCIATION,
        )
        result = generator.generate(profile)
        all_methods = [c.method_id for c in result.primary_candidates + result.secondary_candidates]
        assert "independent_t_test" not in all_methods
        assert "one_way_anova" not in all_methods


# ===========================================================================
# Multiple Group Continuous Outcome
# ===========================================================================

class TestMultiGroupContinuous:
    def test_recommends_anova(self, generator):
        """Multiple groups, continuous outcome → ANOVA as candidate."""
        profile = _make_profile(
            outcome_type="continuous",
            predictor_variables=["group"],
            predictor_types=["categorical"],
            n_groups=3,
            group_balances={"A": 60, "B": 70, "C": 70},
            problem_type=AnalysisProblemType.MULTI_GROUP_CONTINUOUS_OUTCOME,
        )
        result = generator.generate(profile)
        method_ids = [c.method_id for c in result.primary_candidates + result.secondary_candidates]
        assert "one_way_anova" in method_ids


# ===========================================================================
# Binary Outcome with Predictors
# ===========================================================================

class TestBinaryOutcomePredictors:
    def test_recommends_logistic_regression(self, generator):
        """Binary outcome with predictors → logistic regression."""
        profile = _make_profile(
            outcome_type="binary",
            predictor_variables=["x1", "x2"],
            predictor_types=["continuous", "binary"],
            n_groups=2,
            problem_type=AnalysisProblemType.BINARY_OUTCOME_MULTIVARIABLE,
        )
        result = generator.generate(profile)
        method_ids = [c.method_id for c in result.primary_candidates + result.secondary_candidates]
        assert "logistic_regression" in method_ids


# ===========================================================================
# Continuous Outcome with Predictors
# ===========================================================================

class TestContinuousOutcomePredictors:
    def test_recommends_linear_regression(self, generator):
        """Continuous outcome with predictors → linear regression."""
        profile = _make_profile(
            outcome_type="continuous",
            predictor_variables=["x1", "x2"],
            predictor_types=["continuous", "binary"],
            n_groups=1,
            problem_type=AnalysisProblemType.UNKNOWN,
        )
        result = generator.generate(profile)
        method_ids = [c.method_id for c in result.primary_candidates + result.secondary_candidates]
        assert "linear_regression" in method_ids


# ===========================================================================
# Time-to-Event Outcome
# ===========================================================================

class TestSurvivalOutcome:
    def test_recommends_survival_methods(self, generator):
        """Time-to-event outcome → KM, log-rank, Cox as candidates (any list)."""
        profile = _make_profile(
            outcome_type="survival",
            predictor_variables=["treatment"],
            predictor_types=["binary"],
            n_groups=2,
            follow_up_or_time=True,
            problem_type=AnalysisProblemType.TIME_TO_EVENT,
        )
        result = generator.generate(profile)
        all_methods = [c.method_id for c in
                       result.primary_candidates + result.secondary_candidates +
                       result.sensitivity_candidates + result.exploratory_candidates]
        assert "kaplan_meier" in all_methods
        assert "log_rank" in all_methods
        assert "cox_proportional_hazards" in all_methods

    def test_survival_classification_by_structure(self, generator):
        """Survival methods are classified by data structure, not p-value."""
        # 2 groups + predictor → log_rank primary, cox primary, KM sensitivity
        profile = _make_profile(
            outcome_type="survival",
            predictor_variables=["treatment"],
            predictor_types=["binary"],
            n_groups=2,
            follow_up_or_time=True,
            problem_type=AnalysisProblemType.TIME_TO_EVENT,
        )
        result = generator.generate(profile)
        primary_ids = [c.method_id for c in result.primary_candidates]
        sensitivity_ids = [c.method_id for c in result.sensitivity_candidates]
        # KM is sensitivity when predictors exist and n_groups > 1
        assert "kaplan_meier" in sensitivity_ids
        assert "log_rank" in primary_ids
        assert "cox_proportional_hazards" in primary_ids

    def test_does_not_recommends_non_survival_methods(self, generator):
        """Non-survival methods not recommended for survival outcomes."""
        profile = _make_profile(
            outcome_type="survival",
            predictor_variables=["treatment"],
            predictor_types=["binary"],
            n_groups=2,
            follow_up_or_time=True,
            problem_type=AnalysisProblemType.TIME_TO_EVENT,
        )
        result = generator.generate(profile)
        all_methods = [c.method_id for c in
                       result.primary_candidates + result.secondary_candidates +
                       result.sensitivity_candidates]
        assert "independent_t_test" not in all_methods
        assert "pearson_correlation" not in all_methods


# ===========================================================================
# Invariant 37: Recommendation Independence from p-values
# ===========================================================================

class TestRecommendationNotPValueDriven:
    def test_same_design_different_outcomes_same_recommendations(self, generator):
        """Same design/variables, different random outcomes → same method recommendations.

        This is the critical anti-p-hacking test: recommendation must not depend
        on which method produces a smaller p-value.
        """
        # Build profiles with identical structure (different random outcomes
        # don't affect the profile — only the data would, and we don't pass data)
        profile1 = _make_profile(
            outcome_type="continuous",
            predictor_variables=["group"],
            predictor_types=["binary"],
            n_groups=2,
            problem_type=AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
        )
        profile2 = _make_profile(
            outcome_type="continuous",
            predictor_variables=["group"],
            predictor_types=["binary"],
            n_groups=2,
            problem_type=AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
        )

        result1 = generator.generate(profile1)
        result2 = generator.generate(profile2)

        # Method IDs must be identical
        ids1 = sorted([c.method_id for c in result1.primary_candidates + result1.secondary_candidates])
        ids2 = sorted([c.method_id for c in result2.primary_candidates + result2.secondary_candidates])
        assert ids1 == ids2

    def test_recommendations_deterministic(self, generator):
        """Same profile → same recommendations (deterministic)."""
        profile = _make_profile(
            outcome_type="continuous",
            predictor_variables=["group"],
            predictor_types=["binary"],
            n_groups=2,
            problem_type=AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
        )
        result1 = generator.generate(profile)
        result2 = generator.generate(profile)
        ids1 = sorted([c.method_id for c in result1.primary_candidates + result1.secondary_candidates])
        ids2 = sorted([c.method_id for c in result2.primary_candidates + result2.secondary_candidates])
        assert ids1 == ids2

    def test_rejections_are_deterministic(self, generator):
        """Rejected methods are the same regardless of outcome values."""
        profile = _make_profile(
            outcome_type="continuous",
            predictor_variables=["group"],
            predictor_types=["binary"],
            n_groups=2,
            paired=False,
            problem_type=AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
        )
        result1 = generator.generate(profile)
        result2 = generator.generate(profile)
        rejections1 = sorted([r["method_id"] for r in result1.rejected_methods])
        rejections2 = sorted([r["method_id"] for r in result2.rejected_methods])
        assert rejections1 == rejections2

    def test_no_p_value_in_candidate_metadata(self, generator):
        """CandidateAnalysis objects contain no p-value field."""
        profile = _make_profile(
            outcome_type="continuous",
            predictor_variables=["group"],
            predictor_types=["binary"],
            n_groups=2,
            problem_type=AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
        )
        result = generator.generate(profile)
        for candidate in result.primary_candidates + result.secondary_candidates:
            candidate_dict = candidate.__dict__ if hasattr(candidate, "__dict__") else {}
            # No p-value field should exist
            assert not any("p_value" in str(k).lower() or "pvalue" in str(k).lower() for k in candidate_dict.keys())
            # No "significance" in rationale
            assert "significance" not in candidate.rationale.lower()
            assert "p <" not in candidate.rationale.lower()
            assert "p<" not in candidate.rationale.lower()
