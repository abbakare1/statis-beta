"""Candidate Generator — deterministic, registry-driven method selection.

This module NEVER uses the LLM. It queries the method registry to produce
a list of plausible statistical methods based on:

1. Variable role resolution (what is the outcome, what is the exposure)
2. Statistical problem type (what question is being asked)
3. Data characteristics (sample size, variable types, distributions)

The LLM may later add methodological rationale, but the candidate list
itself is entirely deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.statistics.registry import (
    REGISTRY,
    get_method,
    MethodEntry,
    find_methods_for_outcome,
)
from app.statistics.variable_roles import (
    VariableRole,
    VariableRoleResolutionSet,
    AnalysisProblemType,
)


@dataclass
class DataProfile:
    """Structured summary of dataset characteristics for candidate generation.

    This is what the planner passes to the generator. It contains metadata
    about the dataset, NOT the raw data itself.
    """
    n_observations: int
    outcome_variable: str
    outcome_type: str  # continuous, binary, categorical, ordinal
    predictor_variables: list[str] = field(default_factory=list)
    predictor_types: list[str] = field(default_factory=list)  # parallel to predictor_variables
    grouping_variables: list[str] = field(default_factory=list)
    grouping_types: list[str] = field(default_factory=list)  # parallel to grouping_variables
    covariates: list[str] = field(default_factory=list)
    covariate_types: list[str] = field(default_factory=list)
    n_groups: int = 0
    group_balances: dict[str, int] = field(default_factory=dict)
    paired: bool = False
    paired_variables: list[str] = field(default_factory=list)
    study_design: str | None = None
    research_question: str | None = None
    has_repeated_measures: bool = False
    follow_up_or_time: bool = False
    missingness_summary: dict[str, float] = field(default_factory=dict)
    event_rate: float | None = None  # for binary outcomes
    distribution_notes: list[str] = field(default_factory=list)
    # Phase 2D.2 additions
    problem_type: AnalysisProblemType = AnalysisProblemType.UNKNOWN
    role_resolutions: VariableRoleResolutionSet | None = None
    identifier_variables: list[str] = field(default_factory=list)


@dataclass
class CandidateAnalysis:
    """A single candidate analysis method with registry-derived metadata."""
    method_id: str
    method_name: str
    description: str
    outcome_type_compatible: bool
    predictor_type_compatible: bool
    design_compatible: bool
    sample_size_adequate: bool
    min_observations_required: int
    assumptions: list[str]
    effect_measure: str
    alternatives: list[str]
    sensitivity_methods: list[str]
    common_misuse: list[str]
    limitations: list[str]
    confidence: str = "high"  # high, moderate, low
    warnings: list[str] = field(default_factory=list)
    rationale: str = ""
    # Phase 2D.2 additions
    problem_type_compatible: bool = True
    paired_required: bool = False
    paired_available: bool = False


@dataclass
class CandidateSet:
    """The complete set of candidate analyses for a given data profile."""
    data_profile: DataProfile
    primary_candidates: list[CandidateAnalysis] = field(default_factory=list)
    secondary_candidates: list[CandidateAnalysis] = field(default_factory=list)
    sensitivity_candidates: list[CandidateAnalysis] = field(default_factory=list)
    exploratory_candidates: list[CandidateAnalysis] = field(default_factory=list)
    rejected_methods: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Phase 2D.2 additions
    problem_type: AnalysisProblemType = AnalysisProblemType.UNKNOWN


class CandidateGenerator:
    """Generate candidate statistical analyses from data profile and registry.

    This is entirely deterministic — no LLM calls.

    Phase 2D.2: Now considers:
    - Variable role resolution (identifiers excluded)
    - Statistical problem type (method family matching)
    - Paired/structural requirements
    """

    def generate(self, profile: DataProfile) -> CandidateSet:
        """Produce a structured CandidateSet from the data profile."""
        result = CandidateSet(data_profile=profile, problem_type=profile.problem_type)

        # Find all registry methods compatible with outcome type
        compatible_methods = find_methods_for_outcome(profile.outcome_type)

        # Filter to only registered methods that match predictor types
        viable_methods: list[tuple[MethodEntry, list[str]]] = []

        for method in compatible_methods:
            # Skip descriptive — it's always available but not a candidate for hypothesis testing
            if method.method_id == "descriptive":
                continue

            # Check if paired method is appropriate
            if self._is_paired_method(method) and not profile.paired:
                continue

            # Check if method requires paired data
            if self._requires_pairing(method) and not profile.paired:
                continue

            # Check predictor type compatibility
            pred_warnings = self._check_predictor_compatibility(method, profile)

            # Check sample size
            sample_ok = profile.n_observations >= method.min_observations

            # Check group requirements
            groups_ok = profile.n_groups >= method.min_groups if method.min_groups > 1 else True

            # Check problem type compatibility
            problem_ok = self._check_problem_type_compatibility(method, profile)

            # Survival-specific filtering: only include survival methods when
            # outcome type is "survival" AND problem type is TIME_TO_EVENT
            if method.method_id in ("kaplan_meier", "log_rank", "cox_proportional_hazards"):
                if profile.outcome_type != "survival":
                    problem_ok = False
                elif profile.problem_type != AnalysisProblemType.TIME_TO_EVENT:
                    problem_ok = False

            if sample_ok and groups_ok and problem_ok:
                viable_methods.append((method, pred_warnings))

        # Classify candidates
        for method, pred_warnings in viable_methods:
            candidate = self._build_candidate(method, profile, pred_warnings)
            self._classify_candidate(candidate, profile, result)

        # Survival-specific classification adjustments
        if profile.problem_type == AnalysisProblemType.TIME_TO_EVENT:
            self._adjust_survival_classification(profile, result)

        # Add rejected methods for traceability
        self._record_rejections(compatible_methods, viable_methods, profile, result)

        # Generate warnings
        self._generate_warnings(profile, result)

        return result

    def _is_paired_method(self, method: MethodEntry) -> bool:
        """Check if a method is specifically for paired data."""
        paired_methods = {"paired_t_test", "wilcoxon_signed_rank"}
        return method.method_id in paired_methods

    def _requires_pairing(self, method: MethodEntry) -> bool:
        """Check if a method requires paired/repeated measures."""
        return self._is_paired_method(method)

    def _check_problem_type_compatibility(
        self, method: MethodEntry, profile: DataProfile
    ) -> bool:
        """Check if a method is compatible with the identified problem type.

        This prevents proposing correlation for group comparisons, etc.
        """
        problem = profile.problem_type

        if problem == AnalysisProblemType.UNKNOWN:
            return True  # If problem is unknown, allow all methods

        # Survival/time-to-event problems
        if problem == AnalysisProblemType.TIME_TO_EVENT:
            survival_methods = {
                "kaplan_meier", "log_rank", "cox_proportional_hazards", "descriptive",
            }
            return method.method_id in survival_methods

        # Group comparison problems
        if problem in (
            AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME,
            AnalysisProblemType.TWO_PAIRED_GROUPS_CONTINUOUS_OUTCOME,
            AnalysisProblemType.MULTI_GROUP_CONTINUOUS_OUTCOME,
        ):
            # Only allow methods appropriate for group comparisons
            group_methods = {
                "independent_t_test", "paired_t_test", "mann_whitney_u",
                "wilcoxon_signed_rank", "one_way_anova", "chi_square",
                "fisher_exact", "linear_regression", "logistic_regression",
                "descriptive",
            }
            return method.method_id in group_methods

        # Correlation/association problems
        if problem == AnalysisProblemType.CONTINUOUS_CONTINUOUS_ASSOCIATION:
            correlation_methods = {
                "pearson_correlation", "spearman_correlation",
                "linear_regression", "descriptive",
            }
            return method.method_id in correlation_methods

        # Binary outcome problems
        if problem in (
            AnalysisProblemType.BINARY_OUTCOME_SINGLE_PREDICTOR,
            AnalysisProblemType.BINARY_OUTCOME_MULTIVARIABLE,
        ):
            binary_methods = {
                "chi_square", "fisher_exact", "logistic_regression",
                "mann_whitney_u", "descriptive",
            }
            return method.method_id in binary_methods

        return True

    def _check_predictor_compatibility(
        self, method: MethodEntry, profile: DataProfile
    ) -> list[str]:
        """Check if predictor/grouping types are compatible with the method.

        Returns a list of warning strings (empty = fully compatible).
        """
        warnings = []

        # Identify group comparison methods that expect a grouping variable
        group_comparison_methods = {
            "independent_t_test", "paired_t_test", "mann_whitney_u",
            "wilcoxon_signed_rank", "one_way_anova", "chi_square",
            "fisher_exact", "log_rank",
        }

        # Identify correlation methods that expect a single predictor
        correlation_methods = {"pearson_correlation", "spearman_correlation"}

        if method.method_id in group_comparison_methods:
            # Check grouping variable types
            # Backward compatibility: if grouping_variables is empty, use
            # predictor_variables for binary/categorical predictors
            effective_grouping_types = profile.grouping_types
            if not effective_grouping_types:
                effective_grouping_types = [
                    ptype for ptype in profile.predictor_types
                    if ptype in ("binary", "categorical")
                ]

            if not effective_grouping_types:
                warnings.append(
                    f"No grouping variable available for {method.name}"
                )
            for gtype in effective_grouping_types:
                if "any" in method.predictor_types or gtype in method.predictor_types:
                    continue
                warnings.append(
                    f"Grouping variable type '{gtype}' not directly supported by {method.name} "
                    f"(supports: {method.predictor_types})"
                )
        elif method.method_id in correlation_methods:
            # Check first predictor type
            if not profile.predictor_types:
                warnings.append(
                    f"No predictor variable available for {method.name}"
                )
            elif profile.predictor_types[0] not in method.predictor_types and "any" not in method.predictor_types:
                warnings.append(
                    f"Predictor type '{profile.predictor_types[0]}' not directly supported by {method.name} "
                    f"(supports: {method.predictor_types})"
                )
        else:
            # Regression methods: check all predictor types
            if not profile.predictor_types:
                return warnings

            for ptype in profile.predictor_types:
                if "any" in method.predictor_types or ptype in method.predictor_types:
                    continue
                warnings.append(
                    f"Predictor type '{ptype}' not directly supported by {method.name} "
                    f"(supports: {method.predictor_types})"
                )
        return warnings

    def _build_candidate(
        self,
        method: MethodEntry,
        profile: DataProfile,
        pred_warnings: list[str],
    ) -> CandidateAnalysis:
        """Build a CandidateAnalysis from a MethodEntry and DataProfile."""
        # Determine outcome compatibility
        outcome_compatible = (
            "any" in method.outcome_types
            or profile.outcome_type in method.outcome_types
        )

        # Determine design compatibility
        design_compatible = True
        if profile.study_design and "any" not in method.study_designs:
            design_key = profile.study_design.lower().replace(" ", "_")
            design_compatible = design_key in method.study_designs

        # Sample size check
        sample_adequate = profile.n_observations >= method.min_observations

        warnings = list(pred_warnings)
        if not sample_adequate:
            warnings.append(
                f"Sample size ({profile.n_observations}) below minimum "
                f"({method.min_observations}) for {method.name}"
            )
        if not design_compatible:
            warnings.append(
                f"Study design '{profile.study_design}' not in compatible list "
                f"for {method.name}"
            )

        confidence = "high"
        if warnings:
            confidence = "moderate"
        if not sample_adequate or not design_compatible:
            confidence = "low"

        paired_required = self._requires_pairing(method)
        paired_available = profile.paired

        return CandidateAnalysis(
            method_id=method.method_id,
            method_name=method.name,
            description=method.description,
            outcome_type_compatible=outcome_compatible,
            predictor_type_compatible=len(pred_warnings) == 0,
            design_compatible=design_compatible,
            sample_size_adequate=sample_adequate,
            min_observations_required=method.min_observations,
            assumptions=method.assumptions,
            effect_measure=method.effect_measure,
            alternatives=method.alternative_methods,
            sensitivity_methods=method.sensitivity_methods,
            common_misuse=method.common_misuse,
            limitations=method.limitations,
            confidence=confidence,
            warnings=warnings,
            paired_required=paired_required,
            paired_available=paired_available,
        )

    def _classify_candidate(
        self,
        candidate: CandidateAnalysis,
        profile: DataProfile,
        result: CandidateSet,
    ):
        """Classify a candidate into primary/secondary/sensitivity/exploratory.

        Classification is based on structural characteristics of the method
        relative to the data profile. The LLM may later override this.
        """
        # Primary: methods that directly test the association between outcome and predictors
        if (
            candidate.outcome_type_compatible
            and candidate.predictor_type_compatible
            and candidate.sample_size_adequate
            and candidate.problem_type_compatible
        ):
            result.primary_candidates.append(candidate)

        # Sensitivity: alternative methods for the same outcome/predictor structure
        elif candidate.outcome_type_compatible and candidate.sample_size_adequate:
            result.sensitivity_candidates.append(candidate)

        # Exploratory: methods with warnings or incompatible aspects
        else:
            result.exploratory_candidates.append(candidate)

    def _adjust_survival_classification(
        self,
        profile: DataProfile,
        result: CandidateSet,
    ):
        """Adjust survival method classification based on data structure.

        Survival methods are reclassified based on:
        - Kaplan-Meier: primary for ungrouped or single-group analyses
        - Log-rank: primary when comparing exactly 2 groups
        - Cox PH: primary when there are covariates to adjust for
        """
        has_predictors = len(profile.predictor_variables) > 0
        n_groups = profile.n_groups

        survival_ids = {"kaplan_meier", "log_rank", "cox_proportional_hazards"}

        # Separate survival from non-survival candidates
        survival_candidates = []
        non_survival_primary = []
        non_survival_other = []

        for c in result.primary_candidates:
            if c.method_id in survival_ids:
                survival_candidates.append(c)
            else:
                non_survival_primary.append(c)

        for c in result.sensitivity_candidates + result.exploratory_candidates:
            if c.method_id not in survival_ids:
                non_survival_other.append(c)

        # Clear and rebuild
        result.primary_candidates = list(non_survival_primary)
        result.sensitivity_candidates = list(non_survival_other)
        result.exploratory_candidates = []

        for c in survival_candidates:
            mid = c.method_id

            if mid == "kaplan_meier":
                # KM is primary when no predictors or single group
                if not has_predictors or n_groups <= 1:
                    result.primary_candidates.append(c)
                else:
                    result.sensitivity_candidates.append(c)

            elif mid == "log_rank":
                # Log-rank is primary when comparing exactly 2 groups
                if n_groups == 2 and has_predictors:
                    result.primary_candidates.append(c)
                else:
                    result.sensitivity_candidates.append(c)

            elif mid == "cox_proportional_hazards":
                # Cox is primary when there are covariates to adjust for
                if has_predictors and len(profile.predictor_variables) >= 1:
                    result.primary_candidates.append(c)
                else:
                    result.sensitivity_candidates.append(c)

    def _record_rejections(
        self,
        compatible_methods: list[MethodEntry],
        viable_methods: list[tuple[MethodEntry, list[str]]],
        profile: DataProfile,
        result: CandidateSet,
    ):
        """Record why methods were not included as candidates."""
        viable_ids = {m.method_id for m, _ in viable_methods}

        for method in compatible_methods:
            if method.method_id in viable_ids:
                continue
            if method.method_id == "descriptive":
                continue

            reasons = []
            if profile.n_observations < method.min_observations:
                reasons.append(
                    f"Insufficient sample size ({profile.n_observations} < {method.min_observations})"
                )
            if profile.n_groups < method.min_groups and method.min_groups > 1:
                reasons.append(
                    f"Insufficient groups ({profile.n_groups} < {method.min_groups})"
                )
            if self._requires_pairing(method) and not profile.paired:
                reasons.append("Requires paired/repeated measures data")
            if not self._check_problem_type_compatibility(method, profile):
                reasons.append(
                    f"Not compatible with problem type '{profile.problem_type.value}'"
                )

            result.rejected_methods.append({
                "method_id": method.method_id,
                "method_name": method.name,
                "reasons": reasons if reasons else ["Not compatible with data profile"],
            })

    def _generate_warnings(self, profile: DataProfile, result: CandidateSet):
        """Generate dataset-level warnings."""
        if profile.n_observations < 10:
            result.warnings.append(
                f"Very small sample size ({profile.n_observations}). "
                "Most methods will be underpowered."
            )

        if profile.outcome_type == "binary" and profile.event_rate is not None:
            if profile.event_rate < 0.1 or profile.event_rate > 0.9:
                result.warnings.append(
                    f"Extreme outcome imbalance (event rate={profile.event_rate:.2%}). "
                    "Consider exact methods or penalized regression."
                )

        high_missing = [
            var for var, pct in profile.missingness_summary.items()
            if pct > 20
        ]
        if high_missing:
            result.warnings.append(
                f"High missingness (>20%) in: {', '.join(high_missing)}"
            )

        if not profile.predictor_variables and not profile.grouping_variables:
            result.warnings.append("No predictor or grouping variables specified.")

        # Warn if identifiers are in predictor or grouping lists
        if profile.identifier_variables:
            pred_overlap = set(profile.predictor_variables) & set(profile.identifier_variables)
            grp_overlap = set(profile.grouping_variables) & set(profile.identifier_variables)
            if pred_overlap:
                result.warnings.append(
                    f"Identifier variables in predictor list: {', '.join(pred_overlap)}"
                )
            if grp_overlap:
                result.warnings.append(
                    f"Identifier variables in grouping variable list: {', '.join(grp_overlap)}"
                )

        # Warn about problem type
        if profile.problem_type == AnalysisProblemType.UNKNOWN:
            result.warnings.append(
                "Statistical problem type could not be determined. "
                "Results may include inappropriate methods."
            )

        # Survival-specific warnings
        if profile.problem_type == AnalysisProblemType.TIME_TO_EVENT:
            if profile.outcome_type != "survival":
                result.warnings.append(
                    "Problem type is TIME_TO_EVENT but outcome type is not 'survival'. "
                    "Survival methods may not be appropriate."
                )
            if profile.event_rate is not None:
                if profile.event_rate < 0.1:
                    result.warnings.append(
                        f"Low event rate ({profile.event_rate:.2%}) for survival analysis. "
                        "Results may be underpowered."
                    )
                elif profile.event_rate > 0.95:
                    result.warnings.append(
                        f"Very high event rate ({profile.event_rate:.2%}) for survival analysis. "
                        "Consider checking for data issues."
                    )
