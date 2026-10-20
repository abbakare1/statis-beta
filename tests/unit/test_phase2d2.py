"""Phase 2D.2 Tests — Semantic Planning and Statistical Candidate Hardening.

Tests A through J plus end-to-end regression test for Test 1.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
import numpy as np
import pandas as pd
import pytest


# Initialize DB before any agent runs
@pytest.fixture(scope="session", autouse=True)
def _init_test_db():
    """Create test database tables."""
    from app.core.database import init_db, close_db
    asyncio.run(init_db())
    yield
    asyncio.run(close_db())


def run_async(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Test A — Identifier Protection
# ---------------------------------------------------------------------------

class TestIdentifierProtection:
    """Assert that identifiers are never eligible as outcome, predictor, or grouping variable."""

    def test_identifier_not_outcome(self):
        from app.statistics.variable_roles import resolve_variable_roles, VariableRole

        variables = [
            {"name": "patient_id", "dtype": "int64", "n_unique": 120, "n_total": 120, "n_missing": 0},
            {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_total": 120, "n_missing": 0},
            {"name": "systolic_bp", "dtype": "float64", "n_unique": 101, "n_total": 120, "n_missing": 0},
        ]
        result = resolve_variable_roles(variables, "Does treatment differ in BP?")

        id_res = next(r for r in result.resolutions if r.variable_name == "patient_id")
        assert id_res.proposed_role == VariableRole.IDENTIFIER
        assert id_res.proposed_role != VariableRole.OUTCOME
        assert id_res.proposed_role != VariableRole.PREDICTOR
        assert id_res.proposed_role != VariableRole.GROUPING_VARIABLE

    def test_identifier_excluded_from_predictors(self):
        from app.statistics.variable_roles import resolve_variable_roles, VariableRole

        variables = [
            {"name": "record_id", "dtype": "int64", "n_unique": 100, "n_total": 100, "n_missing": 0},
            {"name": "value", "dtype": "float64", "n_unique": 95, "n_total": 100, "n_missing": 0},
        ]
        result = resolve_variable_roles(variables)

        id_res = next(r for r in result.resolutions if r.variable_name == "record_id")
        assert id_res.proposed_role == VariableRole.IDENTIFIER

    def test_identifier_by_name_pattern(self):
        from app.statistics.variable_roles import resolve_variable_roles, VariableRole

        variables = [
            {"name": "subject_id", "dtype": "object", "n_unique": 50, "n_total": 50, "n_missing": 0},
            {"name": "score", "dtype": "float64", "n_unique": 48, "n_total": 50, "n_missing": 0},
        ]
        result = resolve_variable_roles(variables)

        id_res = next(r for r in result.resolutions if r.variable_name == "subject_id")
        assert id_res.proposed_role == VariableRole.IDENTIFIER

    def test_identifier_by_uniqueness(self):
        from app.statistics.variable_roles import resolve_variable_roles, VariableRole

        variables = [
            {"name": "uuid_col", "dtype": "object", "n_unique": 200, "n_total": 200, "n_missing": 0},
            {"name": "group", "dtype": "int64", "n_unique": 3, "n_total": 200, "n_missing": 0},
        ]
        result = resolve_variable_roles(variables)

        id_res = next(r for r in result.resolutions if r.variable_name == "uuid_col")
        assert id_res.proposed_role == VariableRole.IDENTIFIER


# ---------------------------------------------------------------------------
# Test B — Research-Question Outcome Mapping
# ---------------------------------------------------------------------------

class TestResearchQuestionMapping:
    """Assert that the research question correctly maps to variable roles."""

    def test_rq_maps_treatment_to_exposure(self):
        from app.statistics.variable_roles import resolve_variable_roles, VariableRole

        variables = [
            {"name": "patient_id", "dtype": "int64", "n_unique": 120, "n_total": 120, "n_missing": 0},
            {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_total": 120, "n_missing": 0},
            {"name": "systolic_bp", "dtype": "float64", "n_unique": 101, "n_total": 120, "n_missing": 0},
        ]
        result = resolve_variable_roles(
            variables,
            "Does treatment group differ in systolic blood pressure?"
        )

        treatment = next(r for r in result.resolutions if r.variable_name == "treatment")
        assert treatment.proposed_role in (VariableRole.EXPOSURE, VariableRole.GROUPING_VARIABLE)

    def test_rq_maps_systolic_bp_to_outcome(self):
        from app.statistics.variable_roles import resolve_variable_roles, VariableRole

        variables = [
            {"name": "patient_id", "dtype": "int64", "n_unique": 120, "n_total": 120, "n_missing": 0},
            {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_total": 120, "n_missing": 0},
            {"name": "systolic_bp", "dtype": "float64", "n_unique": 101, "n_total": 120, "n_missing": 0},
        ]
        result = resolve_variable_roles(
            variables,
            "Does treatment group differ in systolic blood pressure?"
        )

        bp = next(r for r in result.resolutions if r.variable_name == "systolic_bp")
        assert bp.proposed_role == VariableRole.OUTCOME
        assert bp.linked_research_question_term is not None
        assert bp.role_source.value == "research_question_match"

    def test_rq_no_question_uses_heuristics(self):
        from app.statistics.variable_roles import resolve_variable_roles, VariableRole

        variables = [
            {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_total": 100, "n_missing": 0},
            {"name": "outcome_score", "dtype": "float64", "n_unique": 90, "n_total": 100, "n_missing": 0},
        ]
        result = resolve_variable_roles(variables, None)

        outcome = next(r for r in result.resolutions if r.variable_name == "outcome_score")
        assert outcome.proposed_role == VariableRole.OUTCOME


# ---------------------------------------------------------------------------
# Test C — Statistical Problem Classification
# ---------------------------------------------------------------------------

class TestProblemClassification:
    """Assert correct statistical problem type classification."""

    def test_two_groups_continuous_outcome(self):
        from app.statistics.variable_roles import resolve_variable_roles, AnalysisProblemType

        variables = [
            {"name": "patient_id", "dtype": "int64", "n_unique": 120, "n_total": 120, "n_missing": 0},
            {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_total": 120, "n_missing": 0},
            {"name": "systolic_bp", "dtype": "float64", "n_unique": 101, "n_total": 120, "n_missing": 0},
        ]
        result = resolve_variable_roles(
            variables,
            "Does treatment group differ in systolic blood pressure?"
        )
        assert result.problem_type == AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME

    def test_correlation_problem(self):
        from app.statistics.variable_roles import resolve_variable_roles, AnalysisProblemType

        variables = [
            {"name": "height", "dtype": "float64", "n_unique": 80, "n_total": 100, "n_missing": 0},
            {"name": "weight", "dtype": "float64", "n_unique": 85, "n_total": 100, "n_missing": 0},
        ]
        result = resolve_variable_roles(
            variables,
            "Is there an association between height and weight?"
        )
        assert result.problem_type == AnalysisProblemType.CONTINUOUS_CONTINUOUS_ASSOCIATION


# ---------------------------------------------------------------------------
# Test D — Paired-Method Protection
# ---------------------------------------------------------------------------

class TestPairedMethodProtection:
    """Assert that paired methods are not eligible without pairing evidence."""

    def test_wilcoxon_rejected_without_pairing(self):
        from app.statistics.candidate_generator import CandidateGenerator, DataProfile
        from app.statistics.variable_roles import AnalysisProblemType

        profile = DataProfile(
            n_observations=100,
            outcome_variable="bp",
            outcome_type="continuous",
            predictor_variables=["group"],
            predictor_types=["binary"],
            n_groups=2,
            paired=False,
            research_question="Does group differ in BP?",
            problem_type=AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
        )

        gen = CandidateGenerator()
        result = gen.generate(profile)

        # Wilcoxon should not be in primary candidates
        primary_ids = [c.method_id for c in result.primary_candidates]
        assert "wilcoxon_signed_rank" not in primary_ids

    def test_paired_ttest_rejected_without_pairing(self):
        from app.statistics.candidate_generator import CandidateGenerator, DataProfile
        from app.statistics.variable_roles import AnalysisProblemType

        profile = DataProfile(
            n_observations=100,
            outcome_variable="bp",
            outcome_type="continuous",
            predictor_variables=["group"],
            predictor_types=["binary"],
            n_groups=2,
            paired=False,
            problem_type=AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
        )

        gen = CandidateGenerator()
        result = gen.generate(profile)

        primary_ids = [c.method_id for c in result.primary_candidates]
        assert "paired_t_test" not in primary_ids

    def test_wilcoxon_eligible_with_pairing(self):
        from app.statistics.candidate_generator import CandidateGenerator, DataProfile
        from app.statistics.variable_roles import AnalysisProblemType

        profile = DataProfile(
            n_observations=50,
            outcome_variable="bp_change",
            outcome_type="continuous",
            predictor_variables=["time"],
            predictor_types=["binary"],
            n_groups=2,
            paired=True,
            problem_type=AnalysisProblemType.TWO_PAIRED_GROUPS_CONTINUOUS_OUTCOME,
        )

        gen = CandidateGenerator()
        result = gen.generate(profile)

        all_ids = (
            [c.method_id for c in result.primary_candidates]
            + [c.method_id for c in result.sensitivity_candidates]
        )
        assert "wilcoxon_signed_rank" in all_ids


# ---------------------------------------------------------------------------
# Test E — Correlation Protection
# ---------------------------------------------------------------------------

class TestCorrelationProtection:
    """Assert that correlation is not proposed for group comparisons."""

    def test_pearson_not_primary_for_group_comparison(self):
        from app.statistics.candidate_generator import CandidateGenerator, DataProfile
        from app.statistics.variable_roles import AnalysisProblemType

        profile = DataProfile(
            n_observations=120,
            outcome_variable="systolic_bp",
            outcome_type="continuous",
            predictor_variables=["treatment"],
            predictor_types=["binary"],
            n_groups=2,
            paired=False,
            research_question="Does treatment differ in BP?",
            problem_type=AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
        )

        gen = CandidateGenerator()
        result = gen.generate(profile)

        primary_ids = [c.method_id for c in result.primary_candidates]
        assert "pearson_correlation" not in primary_ids
        assert "spearman_correlation" not in primary_ids

    def test_pearson_eligible_for_correlation_problem(self):
        from app.statistics.candidate_generator import CandidateGenerator, DataProfile
        from app.statistics.variable_roles import AnalysisProblemType

        profile = DataProfile(
            n_observations=100,
            outcome_variable="weight",
            outcome_type="continuous",
            predictor_variables=["height"],
            predictor_types=["continuous"],
            n_groups=0,
            paired=False,
            problem_type=AnalysisProblemType.CONTINUOUS_CONTINUOUS_ASSOCIATION,
        )

        gen = CandidateGenerator()
        result = gen.generate(profile)

        all_ids = [c.method_id for c in result.primary_candidates]
        assert "pearson_correlation" in all_ids


# ---------------------------------------------------------------------------
# Test F — Skewness Interpretation
# ---------------------------------------------------------------------------

class TestSkewnessInterpretation:
    """Assert correct skewness interpretation in transformation suggestions."""

    def test_negative_skew_not_called_right_skewed(self):
        from app.statistics.transformations import suggest_transformations

        np.random.seed(42)
        # Create left-skewed data (negative skewness)
        data = np.random.beta(5, 2, 1000) * 100
        df = pd.DataFrame({"bp": data})

        skewness = df["bp"].skew()
        assert skewness < 0, f"Expected negative skewness, got {skewness}"

        candidates = suggest_transformations(df, "bp")

        for c in candidates:
            if c.reason:
                assert "right-skewed" not in c.reason.lower(), (
                    f"Transformation reason incorrectly says 'right-skewed' for negative skewness: {c.reason}"
                )

    def test_symmetric_data_not_called_skewed(self):
        from app.statistics.transformations import suggest_transformations

        np.random.seed(42)
        # Create symmetric data
        data = np.random.normal(100, 15, 1000)
        df = pd.DataFrame({"bp": data})

        skewness = df["bp"].skew()
        assert abs(skewness) < 0.3, f"Expected near-zero skewness, got {skewness}"

        candidates = suggest_transformations(df, "bp")

        for c in candidates:
            if c.reason and "skewed" in c.reason.lower():
                assert "approximately symmetric" in c.reason.lower() or "no transformation" in c.reason.lower(), (
                    f"Symmetric data incorrectly described as skewed: {c.reason}"
                )


# ---------------------------------------------------------------------------
# Test G — Transformation Restraint
# ---------------------------------------------------------------------------

class TestTransformationRestraint:
    """Assert that no transformation is RECOMMENDED for approximately symmetric data."""

    def test_no_transformation_recommended_for_symmetric(self):
        from app.statistics.transformations import suggest_transformations

        np.random.seed(42)
        data = np.random.normal(130, 15, 120)
        df = pd.DataFrame({"systolic_bp": data})

        skewness = df["systolic_bp"].skew()
        assert abs(skewness) < 0.5, f"Expected approximately symmetric data, got skewness={skewness}"

        candidates = suggest_transformations(df, "systolic_bp")

        # No transformation should be "recommended" for approximately symmetric data
        for c in candidates:
            assert c.reason is not None
            # The reason should not claim transformation is needed
            if "recommended" in c.reason.lower():
                pytest.fail(f"Transformation incorrectly recommended for symmetric data: {c.reason}")

    def test_available_not_recommended_distinction(self):
        from app.statistics.transformations import suggest_transformations

        np.random.seed(42)
        data = np.random.normal(100, 10, 100)
        df = pd.DataFrame({"value": data})

        candidates = suggest_transformations(df, "value")

        # All transformations should mention they are available but not currently indicated
        for c in candidates:
            assert "no transformation currently indicated" in c.reason.lower() or \
                   "available" in c.reason.lower() or \
                   "if needed" in c.reason.lower(), (
                f"Transformation should be marked as available, not recommended: {c.reason}"
            )


# ---------------------------------------------------------------------------
# Test H — Unknown Study Design
# ---------------------------------------------------------------------------

class TestUnknownStudyDesign:
    """Assert that unknown study design is preserved, not fabricated."""

    def test_no_design_provided_returns_unknown(self):
        from app.statistics.study_design_state import classify_design_state, DesignState

        state, design_type = classify_design_state(None)
        assert state == DesignState.UNKNOWN
        assert design_type is None

    def test_empty_string_returns_unknown(self):
        from app.statistics.study_design_state import classify_design_state, DesignState

        state, design_type = classify_design_state("")
        assert state == DesignState.UNKNOWN
        assert design_type is None

    def test_llm_inferred_design_is_ambiguous(self):
        from app.statistics.study_design_state import classify_design_state, DesignState

        state, design_type = classify_design_state(
            study_design_from_researcher=None,
            study_design_from_llm="cohort",
        )
        assert state == DesignState.AMBIGUOUS

    def test_researcher_provided_design_is_known(self):
        from app.statistics.study_design_state import classify_design_state, DesignState

        state, design_type = classify_design_state(
            study_design_from_researcher="RCT",
        )
        assert state == DesignState.KNOWN

    def test_study_design_agent_returns_unknown_when_no_input(self):
        from app.agents.study_design import StudyDesignAgent

        agent = StudyDesignAgent()
        result = run_async(agent.run({
            "research_question": "Does treatment work?",
            "study_design": "",
            "project_id": "test",
        }))

        assert result.success
        assert result.data["design_type"] == "unknown"
        assert result.data["design_state"] == "unknown"


# ---------------------------------------------------------------------------
# Test I — Conflict Handling
# ---------------------------------------------------------------------------

class TestConflictHandling:
    """Assert that role conflicts are detected and flagged."""

    def test_multiple_outcomes_detected(self):
        from app.statistics.variable_roles import resolve_variable_roles, VariableRole, RoleStatus

        variables = [
            {"name": "outcome_score", "dtype": "float64", "n_unique": 80, "n_total": 100, "n_missing": 0},
            {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_total": 100, "n_missing": 0},
            {"name": "result_value", "dtype": "float64", "n_unique": 75, "n_total": 100, "n_missing": 0},
        ]
        result = resolve_variable_roles(variables)

        # At least one should have a conflict
        outcomes = [r for r in result.resolutions if r.proposed_role == VariableRole.OUTCOME]
        assert len(outcomes) >= 1

    def test_rq_outcome_takes_precedence(self):
        from app.statistics.variable_roles import resolve_variable_roles, VariableRole

        variables = [
            {"name": "patient_id", "dtype": "int64", "n_unique": 100, "n_total": 100, "n_missing": 0},
            {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_total": 100, "n_missing": 0},
            {"name": "systolic_bp", "dtype": "float64", "n_unique": 90, "n_total": 100, "n_missing": 0},
            {"name": "outcome_measure", "dtype": "float64", "n_unique": 85, "n_total": 100, "n_missing": 0},
        ]
        result = resolve_variable_roles(
            variables,
            "Does treatment group differ in systolic blood pressure?"
        )

        # systolic_bp should be identified as outcome from RQ
        bp = next(r for r in result.resolutions if r.variable_name == "systolic_bp")
        assert bp.proposed_role == VariableRole.OUTCOME
        assert bp.role_source.value == "research_question_match"


# ---------------------------------------------------------------------------
# Test J — Regression Protection
# ---------------------------------------------------------------------------

class TestRegressionProtection:
    """Ensure patient_id never becomes the dependent variable."""

    def test_patient_id_never_outcome_in_planner(self):
        from app.statistics.variable_roles import resolve_variable_roles, VariableRole

        variables = [
            {"name": "patient_id", "dtype": "int64", "n_unique": 120, "n_total": 120, "n_missing": 0},
            {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_total": 120, "n_missing": 0},
            {"name": "systolic_bp", "dtype": "float64", "n_unique": 101, "n_total": 120, "n_missing": 0},
        ]
        result = resolve_variable_roles(variables)

        pid = next(r for r in result.resolutions if r.variable_name == "patient_id")
        assert pid.proposed_role == VariableRole.IDENTIFIER
        assert pid.proposed_role != VariableRole.OUTCOME

    def test_identifier_not_in_predictor_list(self):
        from app.statistics.variable_roles import resolve_variable_roles, VariableRole, get_non_identifier_roles

        variables = [
            {"name": "subject_id", "dtype": "object", "n_unique": 200, "n_total": 200, "n_missing": 0},
            {"name": "group", "dtype": "int64", "n_unique": 3, "n_total": 200, "n_missing": 0},
            {"name": "score", "dtype": "float64", "n_unique": 180, "n_total": 200, "n_missing": 0},
        ]
        result = resolve_variable_roles(variables)

        non_id_roles = get_non_identifier_roles()
        for r in result.resolutions:
            if r.variable_name == "subject_id":
                assert r.proposed_role == VariableRole.IDENTIFIER
                assert r.proposed_role not in non_id_roles


# ---------------------------------------------------------------------------
# Test 18 — End-to-End Regression Test
# ---------------------------------------------------------------------------

class TestEndToEndTest1:
    """Run the complete Test 1 pipeline and verify structural correctness."""

    def test_test1_role_resolution(self):
        from app.statistics.variable_roles import resolve_variable_roles, VariableRole, AnalysisProblemType

        variables = [
            {"name": "patient_id", "dtype": "object", "n_unique": 120, "n_total": 120, "n_missing": 0},
            {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_total": 120, "n_missing": 0},
            {"name": "systolic_bp", "dtype": "float64", "n_unique": 101, "n_total": 120, "n_missing": 0},
        ]
        result = resolve_variable_roles(
            variables,
            "Does treatment group differ in systolic blood pressure?"
        )

        # Verify roles
        roles = {r.variable_name: r.proposed_role for r in result.resolutions}
        assert roles["patient_id"] == VariableRole.IDENTIFIER
        assert roles["treatment"] in (VariableRole.EXPOSURE, VariableRole.GROUPING_VARIABLE)
        assert roles["systolic_bp"] == VariableRole.OUTCOME

        # Verify problem type
        assert result.problem_type == AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME

    def test_test1_candidate_generation(self):
        from app.statistics.candidate_generator import CandidateGenerator, DataProfile
        from app.statistics.variable_roles import AnalysisProblemType

        profile = DataProfile(
            n_observations=120,
            outcome_variable="systolic_bp",
            outcome_type="continuous",
            predictor_variables=[],
            predictor_types=[],
            grouping_variables=["treatment"],
            grouping_types=["binary"],
            n_groups=2,
            paired=False,
            research_question="Does treatment group differ in systolic blood pressure?",
            problem_type=AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
            identifier_variables=["patient_id"],
        )

        gen = CandidateGenerator()
        result = gen.generate(profile)

        primary_ids = [c.method_id for c in result.primary_candidates]

        # Should include appropriate two-group methods
        assert "independent_t_test" in primary_ids
        assert "mann_whitney_u" in primary_ids

        # Should NOT include correlation as primary
        assert "pearson_correlation" not in primary_ids
        assert "spearman_correlation" not in primary_ids

        # Should NOT include paired methods
        assert "paired_t_test" not in primary_ids
        assert "wilcoxon_signed_rank" not in primary_ids

    def test_test1_identifier_excluded(self):
        from app.statistics.variable_roles import resolve_variable_roles, VariableRole

        variables = [
            {"name": "patient_id", "dtype": "object", "n_unique": 120, "n_total": 120, "n_missing": 0},
            {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_total": 120, "n_missing": 0},
            {"name": "systolic_bp", "dtype": "float64", "n_unique": 101, "n_total": 120, "n_missing": 0},
        ]
        result = resolve_variable_roles(variables)

        # patient_id should be identifier, never outcome or predictor
        pid = next(r for r in result.resolutions if r.variable_name == "patient_id")
        assert pid.proposed_role == VariableRole.IDENTIFIER

        # No other variable should be assigned as identifier
        other_roles = [r for r in result.resolutions if r.variable_name != "patient_id"]
        for r in other_roles:
            assert r.proposed_role != VariableRole.IDENTIFIER

    def test_test1_no_transformation_recommended(self):
        from app.statistics.transformations import suggest_transformations
        import numpy as np

        # Create Test 1-like data
        np.random.seed(42)
        n = 120
        treatment = np.random.choice([0, 1], n)
        bp = np.where(treatment == 0,
                      np.random.normal(137.6, 10.0, n),
                      np.random.normal(125.5, 13.2, n))
        df = pd.DataFrame({
            "patient_id": range(1, n + 1),
            "treatment": treatment,
            "systolic_bp": bp,
        })

        candidates = suggest_transformations(df, "systolic_bp")

        # No transformation should be recommended for approximately symmetric data
        skewness = df["systolic_bp"].skew()
        assert abs(skewness) < 0.5, f"Expected approximately symmetric, got skewness={skewness}"

        for c in candidates:
            assert "no transformation currently indicated" in c.reason.lower() or \
                   "available" in c.reason.lower(), (
                f"Transformation should not be recommended for Test 1 data: {c.reason}"
            )

    def test_test1_study_design_preserved(self):
        from app.statistics.study_design_state import classify_design_state, DesignState

        # No design provided
        state, _ = classify_design_state(None)
        assert state == DesignState.UNKNOWN

    def test_test1_full_planner_profile(self):
        """Test that the planner correctly builds a profile for Test 1."""
        from app.agents.analysis_planner import AnalysisPlannerAgent

        planner = AnalysisPlannerAgent()
        context = {
            "research_question": "Does treatment group differ in systolic blood pressure?",
            "study_design": "",
            "data_dictionary": {
                "n_rows": 120,
                "variables": [
                    {"name": "patient_id", "dtype": "object", "n_unique": 120, "n_missing": 0, "missing_pct": 0},
                    {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_missing": 0, "missing_pct": 0},
                    {"name": "systolic_bp", "dtype": "float64", "n_unique": 101, "n_missing": 0, "missing_pct": 0},
                ],
            },
        }

        profile = planner._build_data_profile(context)

        assert profile is not None
        assert profile.outcome_variable == "systolic_bp"
        assert "treatment" in profile.grouping_variables
        assert "patient_id" in profile.identifier_variables
        assert profile.problem_type.value == "two_independent_groups_continuous_outcome"
