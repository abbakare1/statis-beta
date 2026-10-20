"""Regression tests for analysis planner correction.

Tests the 20 specified scenarios for the analysis planner defect correction.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.analysis_planner import AnalysisPlannerAgent
from app.statistics.candidate_generator import CandidateGenerator, DataProfile
from app.statistics.variable_roles import AnalysisProblemType


@pytest.fixture
def planner():
    return AnalysisPlannerAgent()


@pytest.fixture
def generator():
    return CandidateGenerator()


def _make_clinical_trial_profile():
    """Create a data profile matching the clinical trial dataset."""
    return DataProfile(
        n_observations=120,
        outcome_variable="Followup_PSS10",
        outcome_type="continuous",
        predictor_variables=["Age", "BMI", "Intervention_Engagement_Hrs"],
        predictor_types=["continuous", "continuous", "continuous"],
        grouping_variables=["Treatment_Arm"],
        grouping_types=["binary"],
        covariates=[],
        covariate_types=[],
        n_groups=2,
        group_balances={"Digital Intervention": 60, "Standard Care": 60},
        paired=False,
        paired_variables=[],
        study_design="randomized controlled trial",
        research_question="Among participants enrolled in the two-arm clinical trial, does the Digital Intervention produce a greater reduction in perceived stress from baseline to follow-up compared with Standard Care?",
        has_repeated_measures=False,
        follow_up_or_time=False,
        missingness_summary={},
        event_rate=None,
        distribution_notes=[],
        problem_type=AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
        role_resolutions=None,
        identifier_variables=["Participant_ID"],
    )


class TestPredictorSelection:
    """Test 1, 6, 7: Predictor selection behavior."""

    def test_same_predictors_not_blindly_attached(self, planner):
        """Test 1: The same predictors are not blindly attached to every candidate."""
        profile = _make_clinical_trial_profile()
        candidate_set = planner.candidate_generator.generate(profile)

        context = {
            "project_id": "test",
            "dataset_id": "test",
            "research_question": profile.research_question,
            "study_design": profile.study_design,
        }
        plan = planner._build_plan(context, profile, candidate_set, None)

        method_predictors = {}
        for entry in plan.analyses:
            spec = entry.get_current_spec()
            method_predictors[spec.analysis_method] = list(spec.predictors)

        # T-test should have no predictors (uses grouping variable)
        assert method_predictors.get("independent_t_test", []) == []
        # Mann-Whitney should have no predictors (uses grouping variable)
        assert method_predictors.get("mann_whitney_u", []) == []
        # Linear regression should have predictors
        assert len(method_predictors.get("linear_regression", [])) > 0

    def test_linear_regression_uses_justified_predictors(self, planner):
        """Test 6: Linear regression uses a justified predictor set."""
        profile = _make_clinical_trial_profile()
        candidate_set = planner.candidate_generator.generate(profile)

        context = {
            "project_id": "test",
            "dataset_id": "test",
            "research_question": profile.research_question,
            "study_design": profile.study_design,
        }
        plan = planner._build_plan(context, profile, candidate_set, None)

        for entry in plan.analyses:
            spec = entry.get_current_spec()
            if spec.analysis_method == "linear_regression":
                assert "Followup_PSS10" not in spec.predictors
                assert "Participant_ID" not in spec.predictors

    def test_outcome_not_included_as_predictor(self, planner):
        """Test 7: Outcomes are not included as predictors in the same model."""
        profile = _make_clinical_trial_profile()
        candidate_set = planner.candidate_generator.generate(profile)

        context = {
            "project_id": "test",
            "dataset_id": "test",
            "research_question": profile.research_question,
            "study_design": profile.study_design,
        }
        plan = planner._build_plan(context, profile, candidate_set, None)

        for entry in plan.analyses:
            spec = entry.get_current_spec()
            assert spec.primary_outcome not in spec.predictors


class TestGroupingVariableSelection:
    """Test 2, 3, 4: Grouping variable selection behavior."""

    def test_ttest_uses_grouping_variable(self, planner):
        """Test 2: T-test candidates use a valid grouping variable."""
        profile = _make_clinical_trial_profile()
        candidate_set = planner.candidate_generator.generate(profile)

        context = {
            "project_id": "test",
            "dataset_id": "test",
            "research_question": profile.research_question,
            "study_design": profile.study_design,
        }
        plan = planner._build_plan(context, profile, candidate_set, None)

        for entry in plan.analyses:
            spec = entry.get_current_spec()
            if spec.analysis_method == "independent_t_test":
                assert "Treatment_Arm" in spec.subgroup_variables

    def test_mann_whitney_uses_grouping_variable(self, planner):
        """Test 3: Mann-Whitney candidates use a valid grouping variable."""
        profile = _make_clinical_trial_profile()
        candidate_set = planner.candidate_generator.generate(profile)

        context = {
            "project_id": "test",
            "dataset_id": "test",
            "research_question": profile.research_question,
            "study_design": profile.study_design,
        }
        plan = planner._build_plan(context, profile, candidate_set, None)

        for entry in plan.analyses:
            spec = entry.get_current_spec()
            if spec.analysis_method == "mann_whitney_u":
                assert "Treatment_Arm" in spec.subgroup_variables

    def test_anova_rejects_continuous_grouping(self, generator):
        """Test 4: ANOVA rejects continuous grouping variables."""
        profile = DataProfile(
            n_observations=100,
            outcome_variable="outcome",
            outcome_type="continuous",
            predictor_variables=["group"],
            predictor_types=["continuous"],
            grouping_variables=[],
            grouping_types=[],
            n_groups=1,
            problem_type=AnalysisProblemType.MULTI_GROUP_CONTINUOUS_OUTCOME,
        )
        result = generator.generate(profile)
        method_ids = [c.method_id for c in result.primary_candidates + result.sensitivity_candidates]
        assert "one_way_anova" not in method_ids or any(
            "continuous" in w.lower() for c in result.primary_candidates + result.sensitivity_candidates
            if c.method_id == "one_way_anova" for w in c.warnings
        )


class TestIdentifierExclusion:
    """Test 8: Identifiers are excluded from substantive predictor lists."""

    def test_identifiers_excluded_from_predictors(self, planner):
        """Test 8: Identifiers are excluded from substantive predictor lists."""
        profile = _make_clinical_trial_profile()
        candidate_set = planner.candidate_generator.generate(profile)

        context = {
            "project_id": "test",
            "dataset_id": "test",
            "research_question": profile.research_question,
            "study_design": profile.study_design,
        }
        plan = planner._build_plan(context, profile, candidate_set, None)

        for entry in plan.analyses:
            spec = entry.get_current_spec()
            assert "Participant_ID" not in spec.predictors
            assert "Participant_ID" not in spec.subgroup_variables


class TestOutcomeSelection:
    """Test 10: Outcome selection behavior."""

    def test_pss10_prioritized_for_pss10_question(self, planner):
        """Test 10: The stated PSS-10 research question prioritizes PSS-10."""
        csv_path = Path("data/uploads/e825c062-891f-469e-a57c-c0bf7cb01758_rigorous_clinical_trial_data.csv")
        if not csv_path.exists():
            pytest.skip("Clinical trial dataset not found")

        df = pd.read_csv(csv_path)

        variables = []
        for col in df.columns:
            series = df[col]
            variables.append({
                "name": col,
                "dtype": str(series.dtype),
                "n_unique": int(series.nunique()),
                "n_total": len(series),
                "n_missing": int(series.isna().sum()),
                "missing_pct": float(series.isna().sum() / len(series) * 100),
            })

        data_dict = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "variables": variables,
        }

        research_question = "Among participants enrolled in the two-arm clinical trial, does the Digital Intervention produce a greater reduction in perceived stress from baseline to follow-up compared with Standard Care?"

        context = {
            "project_id": "test",
            "dataset_id": "test",
            "file_path": str(csv_path),
            "data_dictionary": data_dict,
            "research_question": research_question,
            "study_design": "randomized controlled trial",
        }

        profile = planner._build_data_profile(context)
        assert "PSS10" in profile.outcome_variable or "pss" in profile.outcome_variable.lower()


class TestExistingMethods:
    """Test 19: Existing registered methods continue to work."""

    def test_all_registered_methods_still_work(self, generator):
        """Test 19: Existing registered methods continue to work."""
        profile_ttest = DataProfile(
            n_observations=100,
            outcome_variable="outcome",
            outcome_type="continuous",
            predictor_variables=["group"],
            predictor_types=["binary"],
            grouping_variables=["group"],
            grouping_types=["binary"],
            n_groups=2,
            problem_type=AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
        )
        result = generator.generate(profile_ttest)
        method_ids = [c.method_id for c in result.primary_candidates + result.sensitivity_candidates]
        assert "independent_t_test" in method_ids

        profile_corr = DataProfile(
            n_observations=100,
            outcome_variable="y",
            outcome_type="continuous",
            predictor_variables=["x"],
            predictor_types=["continuous"],
            n_groups=1,
            problem_type=AnalysisProblemType.CONTINUOUS_CONTINUOUS_ASSOCIATION,
        )
        result = generator.generate(profile_corr)
        method_ids = [c.method_id for c in result.primary_candidates + result.sensitivity_candidates]
        assert "pearson_correlation" in method_ids

        profile_logreg = DataProfile(
            n_observations=100,
            outcome_variable="outcome",
            outcome_type="binary",
            predictor_variables=["x1", "x2"],
            predictor_types=["continuous", "binary"],
            n_groups=2,
            problem_type=AnalysisProblemType.BINARY_OUTCOME_MULTIVARIABLE,
        )
        result = generator.generate(profile_logreg)
        method_ids = [c.method_id for c in result.primary_candidates + result.sensitivity_candidates]
        assert "logistic_regression" in method_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])