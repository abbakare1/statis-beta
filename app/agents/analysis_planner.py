"""Analysis Planner Agent — two-stage analysis planning with registry constraints.

Stage A (deterministic): CandidateGenerator produces method candidates from registry.
Stage B (LLM): Provides methodological rationale, classifies candidates, identifies concerns.
Stage C (deterministic): StatisticalValidator validates every candidate.

The AI must NEVER be the final authority on whether an analysis is valid.
"""

from __future__ import annotations

import json

from app.agents.base import BaseAgent, AgentResult
from app.memory.store import store_memory
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
)
from app.statistics.specification import (
    StatisticalAnalysisSpecification,
    AnalysisDecisionTrace,
    PrespecifiedStatus,
    ApprovalStatus,
    ValidationStatus,
)
from app.statistics.registry import get_method
from app.statistics.validator import StatisticalValidator
from app.statistics.variable_roles import (
    VariableRole,
    VariableRoleResolutionSet,
    AnalysisProblemType,
    resolve_variable_roles,
)
from app.statistics.study_design_state import classify_design_state, DesignState


SYSTEM_PROMPT = """You are a senior biostatistician providing methodological guidance.

You will receive:
1. A set of CANDIDATE statistical methods already identified by the system registry
2. Dataset metadata (variable types, sample sizes, data quality)

Your task is to:
1. Provide a concise methodological rationale for each candidate
2. Classify each candidate as primary, secondary, sensitivity, or exploratory
3. Identify any methodological concerns not captured by the registry
4. Recommend the primary analysis

CRITICAL CONSTRAINTS:
- You MUST only work with the candidates provided. Do NOT invent new methods.
- You MUST NOT select a method because it might produce p < 0.05.
- You MUST NOT select a method because it produces larger effect sizes.
- You MUST distinguish association from causation for observational studies.
- You MUST flag if prespecification status is unclear.

Return a JSON object with:
{
  "primary_analysis_id": "<method_id of recommended primary>",
  "classifications": {
    "<method_id>": {
      "class": "primary|secondary|sensitivity|exploratory",
      "rationale": "why this method is appropriate",
      "concerns": ["methodological concerns"]
    }
  },
  "prespecification_assessment": "prespecified|not_prespecified|unclear",
  "causal_language_flag": "association|causation",
  "multiple_testing_strategy": "none_required|prespecified_limited|adjustment_planned|exploratory_unadjusted|review_required",
  "general_warnings": ["..."]
}"""


class AnalysisPlannerAgent(BaseAgent):
    name = "analysis_planner"
    requires_llm = True
    description = "Generate a validated statistical analysis plan using registry constraints."

    def __init__(self, model_provider=None):
        super().__init__(model_provider=model_provider)
        self.candidate_generator = CandidateGenerator()
        self.validator = StatisticalValidator()

    async def _execute(self, context: dict) -> AgentResult:
        provider = self.get_provider(context)
        project_id = context.get("project_id")

        # ================================================================
        # Stage A: Build data profile (deterministic)
        # ================================================================
        profile = self._build_data_profile(context)
        if profile is None:
            return AgentResult(
                success=False,
                data={"error": "Insufficient data profile information in context."},
                warnings=["Cannot generate candidates without data profile."],
            )

        # ================================================================
        # Stage B: Generate candidates from registry (deterministic)
        # ================================================================
        candidate_set = self.candidate_generator.generate(profile)

        # Build the candidate summary for the LLM
        candidate_summary = self._format_candidates_for_llm(candidate_set)

        # ================================================================
        # Stage C: LLM provides rationale and classification
        # ================================================================
        llm_result = None
        if await provider.is_available():
            llm_result = await self._get_llm_guidance(
                provider, context, candidate_summary
            )

        # ================================================================
        # Stage D: Build AnalysisPlan with entries (deterministic + LLM rationale)
        # ================================================================
        plan = self._build_plan(
            context, profile, candidate_set, llm_result
        )

        # ================================================================
        # Stage E: Validate every candidate (deterministic)
        # ================================================================
        file_path = context.get("file_path")
        self._validate_all_entries(plan, file_path)

        # ================================================================
        # Store in memory
        # ================================================================
        if project_id:
            await store_memory(
                memory_type="project",
                category="analysis_plan",
                key="full_plan",
                value=plan.model_dump_json()[:10000],
                source="deterministic_generation",
                confidence="high",
                project_id=project_id,
            )

        total = len(plan.analyses)
        return AgentResult(
            success=True,
            data={
                "plan_id": plan.plan_id,
                "n_analyses": total,
                "primary": len(plan.get_primary_analyses()),
                "analyses": [self._entry_summary(a) for a in plan.analyses],
                "warnings": plan.warnings,
                "rejected_methods": [
                    r["method_name"] for r in candidate_set.rejected_methods
                ],
            },
            warnings=plan.warnings,
            confidence="high",
        )

    # ----------------------------------------------------------------
    # Data profile construction
    # ----------------------------------------------------------------

    def _build_data_profile(self, context: dict) -> DataProfile | None:
        """Build a DataProfile from the agent context.

        Phase 2D.2: Uses variable role resolution instead of hardcoded heuristics.
        Never falls back to first-variable-as-outcome.
        """
        data_dict = context.get("data_dictionary", {})
        variables = data_dict.get("variables", [])

        if not variables:
            return None

        n_obs = data_dict.get("n_rows", 0)
        if n_obs == 0:
            for v in variables:
                n_obs = max(n_obs, v.get("n_unique", 0) + v.get("n_missing", 0))

        # Resolve variable roles using the new resolver
        research_question = context.get("research_question")
        # Ensure variables have n_total for correct identifier detection
        for v in variables:
            if "n_total" not in v:
                v["n_total"] = n_obs
        role_resolutions = resolve_variable_roles(variables, research_question)

        # Extract outcomes, grouping variables, and predictors from resolutions
        # Grouping variables (binary/categorical exposures) are separated from
        # regression predictors to enable method-appropriate specifications.
        outcome_var = None
        outcome_type = None
        outcome_candidates = []  # Collect all outcome candidates for priority selection
        grouping_vars = []
        grouping_types = []
        predictor_vars = []
        predictor_types = []
        covariate_vars = []
        identifier_vars = []

        for resolution in role_resolutions.resolutions:
            var_info = next((v for v in variables if v.get("name") == resolution.variable_name), None)
            if not var_info:
                continue

            name = var_info.get("name", "")
            dtype = var_info.get("dtype", "unknown")
            n_unique = var_info.get("n_unique", 0)
            var_type = self._infer_type(dtype, n_unique, n_obs)

            pass

            if resolution.proposed_role == VariableRole.IDENTIFIER:
                identifier_vars.append(name)
            elif resolution.proposed_role == VariableRole.OUTCOME:
                outcome_candidates.append((name, var_type))
            elif resolution.proposed_role in (
                VariableRole.EXPOSURE, VariableRole.GROUPING_VARIABLE,
            ):
                # Binary/categorical variables used for group comparisons
                grouping_vars.append(name)
                grouping_types.append(var_type)
            elif resolution.proposed_role == VariableRole.PREDICTOR:
                # Continuous or multi-level variables for regression
                # Exclude variables that look like outcomes of different measures
                # (e.g., cortisol when PSS-10 is the primary outcome)
                # Exclude engagement variables (post-assignment mediators)
                if not self._looks_like_outcome(name) and not self._looks_like_engagement(name):
                    predictor_vars.append(name)
                    predictor_types.append(var_type)
            elif resolution.proposed_role in (VariableRole.COVARIATE, VariableRole.CONFOUNDER_CANDIDATE):
                covariate_vars.append(name)

        # Select the best outcome variable from candidates
        # Priority: follow-up measurements > baseline measurements > others
        if outcome_candidates:
            # Sort by priority: follow-up first, then baseline, then others
            def outcome_priority(name):
                lower = name.lower()
                if "followup" in lower or "follow_up" in lower or "post" in lower:
                    return 0  # Highest priority
                elif "baseline" in lower or "pre" in lower:
                    return 1  # Lower priority
                else:
                    return 2  # Default
            outcome_candidates.sort(key=lambda x: outcome_priority(x[0]))
            outcome_var, outcome_type = outcome_candidates[0]

            # Check if there's a baseline measurement for the primary outcome
            # and add it as a covariate for baseline adjustment
            baseline_var = self._find_baseline_for_outcome(outcome_var, outcome_candidates)
            if baseline_var and baseline_var not in covariate_vars:
                covariate_vars.append(baseline_var)

        # NO FALLBACK to first-variable-as-outcome.
        # If no outcome identified, return None — the pipeline must handle this.
        if outcome_var is None:
            return None

        # Count groups from grouping variables
        n_groups = 0
        for v in variables:
            if v.get("name") in grouping_vars:
                nu = v.get("n_unique", 0)
                if 2 <= nu <= 20:
                    n_groups = max(n_groups, nu)

        # Missingness summary
        missingness = {}
        for v in variables:
            pct = v.get("missing_pct", 0)
            if pct > 0:
                missingness[v.get("name", "")] = pct

        # Event rate for binary outcomes
        event_rate = None
        if outcome_type == "binary":
            for v in variables:
                if v.get("name") == outcome_var:
                    n_unique = v.get("n_unique", 0)
                    if n_unique == 2:
                        event_rate = 0.5
                    break

        # Determine study design state
        study_design_str = context.get("study_design")
        design_state, design_type = classify_design_state(
            study_design_from_researcher=study_design_str,
        )

        return DataProfile(
            n_observations=n_obs,
            outcome_variable=outcome_var,
            outcome_type=outcome_type or "continuous",
            predictor_variables=predictor_vars,
            predictor_types=predictor_types,
            grouping_variables=grouping_vars,
            grouping_types=grouping_types,
            covariates=covariate_vars,
            n_groups=n_groups,
            study_design=study_design_str,
            research_question=research_question,
            missingness_summary=missingness,
            event_rate=event_rate,
            problem_type=role_resolutions.problem_type,
            role_resolutions=role_resolutions,
            identifier_variables=identifier_vars,
        )

    def _find_baseline_for_outcome(self, outcome_var: str, outcome_candidates: list) -> str | None:
        """Find the baseline measurement for a given outcome variable.

        For example, if the outcome is Followup_PSS10, look for Baseline_PSS10.
        """
        import re
        outcome_lower = outcome_var.lower()

        # Look for a baseline version of the outcome
        for candidate_name, _ in outcome_candidates:
            candidate_lower = candidate_name.lower()
            if "baseline" in candidate_lower or "pre" in candidate_lower:
                # Check if it's the same measure as the outcome
                # by comparing the non-baseline/followup parts
                outcome_clean = re.sub(r"(followup|follow_up|post|baseline|pre)", "", outcome_lower)
                candidate_clean = re.sub(r"(followup|follow_up|post|baseline|pre)", "", candidate_lower)
                if outcome_clean == candidate_clean:
                    return candidate_name

        return None

    def _looks_like_engagement(self, name: str) -> bool:
        """Heuristic: does this variable name look like an engagement/mediator variable?

        Engagement variables are post-assignment and may be mediators. They should
        not be included in the primary treatment-effect model.
        """
        import re
        lower = name.lower()
        engagement_patterns = [
            r"engagement", r"adherence", r"compliance", r"participation",
            r"hours", r"minutes", r"duration", r"frequency",
        ]
        return any(re.search(p, lower) for p in engagement_patterns)

    def _looks_like_outcome(self, name: str) -> bool:
        """Heuristic: does this variable name look like an outcome measure?

        Used to exclude potential outcome variables from the predictor list
        when they are not the primary outcome. For example, cortisol variables
        should not be predictors when PSS-10 is the primary outcome.
        """
        import re
        lower = name.lower()
        outcome_patterns = [
            r"outcome", r"result", r"response", r"endpoint",
            r"score", r"concentration", r"pressure",
            r"temperature", r"rate", r"improvement", r"change",
            r"biomarker", r"marker",
            r"pss", r"stress", r"anxiety", r"depression", r"pain",
            r"cortisol", r"glucose", r"cholesterol",
        ]
        return any(re.search(p, lower) for p in outcome_patterns)

    def _infer_type(self, dtype: str, n_unique: int, n_obs: int) -> str:
        """Infer variable type from dtype and cardinality.

        Accepts both raw pandas dtypes ("int64", "float64", "object") and the
        data-dictionary agent's normalized dtype vocabulary ("numeric", "text",
        "categorical", "boolean", "datetime"). Without the normalized names a
        binary numeric outcome was typed "continuous" and the planner proposed
        t-tests instead of logistic/Fisher for binary outcomes.
        """
        dtype = (dtype or "unknown").lower()
        if dtype in ("object", "str", "category", "categorical", "bool", "boolean", "text"):
            if n_unique <= 2:
                return "binary"
            elif n_unique <= 10:
                return "categorical"
            else:
                return "categorical"
        elif dtype in (
            "int64", "int32", "int", "float64", "float32", "float",
            "numeric", "number", "double", "decimal",
        ):
            if n_unique <= 2:
                return "binary"
            elif n_unique <= 10 and n_obs > 0 and n_unique / n_obs < 0.05:
                return "categorical"
            else:
                return "continuous"
        return "continuous"

    # ----------------------------------------------------------------
    # LLM guidance
    # ----------------------------------------------------------------

    def _format_candidates_for_llm(self, candidate_set: CandidateSet) -> str:
        """Format candidate set as a structured prompt for the LLM."""
        lines = []
        lines.append("CANDIDATE ANALYSIS METHODS (from registry):")
        lines.append("")

        all_candidates = (
            candidate_set.primary_candidates
            + candidate_set.sensitivity_candidates
            + candidate_set.exploratory_candidates
        )

        for i, c in enumerate(all_candidates, 1):
            lines.append(f"--- Candidate {i}: {c.method_id} ---")
            lines.append(f"Name: {c.method_name}")
            lines.append(f"Description: {c.description}")
            lines.append(f"Effect measure: {c.effect_measure}")
            lines.append(f"Assumptions: {'; '.join(c.assumptions)}")
            lines.append(f"Alternatives: {', '.join(c.alternatives)}")
            lines.append(f"Common misuse: {'; '.join(c.common_misuse[:2])}")
            if c.warnings:
                lines.append(f"Warnings: {'; '.join(c.warnings)}")
            lines.append(f"Confidence: {c.confidence}")
            lines.append("")

        if candidate_set.rejected_methods:
            lines.append("REJECTED METHODS:")
            for r in candidate_set.rejected_methods:
                lines.append(f"  - {r['method_name']}: {'; '.join(r['reasons'])}")
            lines.append("")

        if candidate_set.warnings:
            lines.append(f"DATA WARNINGS: {'; '.join(candidate_set.warnings)}")

        return "\n".join(lines)

    async def _get_llm_guidance(
        self, provider, context: dict, candidate_summary: str
    ) -> dict | None:
        """Get methodological guidance from the LLM."""
        context_parts = []
        context_parts.append(f"Research Question: {context.get('research_question', 'N/A')}")
        context_parts.append(f"Study Design: {context.get('study_design', 'N/A')}")
        context_parts.append(f"\n{candidate_summary}")

        user_prompt = "\n".join(context_parts)
        user_prompt += "\n\nProvide methodological guidance for these candidates."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            result = await provider.chat_json(messages, temperature=0.0)
            return result
        except Exception as e:
            return None

    # ----------------------------------------------------------------
    # Plan construction
    # ----------------------------------------------------------------

    def _build_plan(
        self,
        context: dict,
        profile: DataProfile,
        candidate_set: CandidateSet,
        llm_result: dict | None,
    ) -> AnalysisPlan:
        """Build the AnalysisPlan from candidates and LLM guidance."""
        plan = AnalysisPlan(
            project_id=context.get("project_id", ""),
            research_question=profile.research_question,
            study_design=profile.study_design,
        )

        # Merge all candidates
        all_candidates = (
            candidate_set.primary_candidates
            + candidate_set.sensitivity_candidates
            + candidate_set.exploratory_candidates
        )

        # Get LLM classifications
        classifications = {}
        if llm_result:
            classifications = llm_result.get("classifications", {})

        # Build AnalysisEntry for each candidate
        for candidate in all_candidates:
            # Determine class from LLM or default to registry classification
            llm_class = classifications.get(candidate.method_id, {})
            analysis_class = llm_class.get("class", self._default_class(candidate))
            rationale = llm_class.get("rationale", "")
            concerns = llm_class.get("concerns", [])

            # Build the specification
            spec = self._build_spec(context, profile, candidate, analysis_class, rationale)

            # Build decision trace
            trace = self._build_trace(candidate, profile, analysis_class, concerns)

            # Determine prespecification
            prespec = "unclear"
            if llm_result:
                prespec = llm_result.get("prespecified_assessment", "unclear")

            # Causal language flag
            causal_flag = "association"
            if llm_result:
                causal_flag = llm_result.get("causal_language_flag", "association")

            # Multiple testing
            mt_strategy = "none_required"
            if llm_result:
                mt_strategy = llm_result.get("multiple_testing_strategy", "none_required")

            entry = AnalysisEntry(
                analysis_name=candidate.method_name,
                analysis_class=analysis_class,
                prespecified_status=prespec,
                causal_language_flag=causal_flag,
                multiple_testing_strategy=mt_strategy,
                decision_trace=trace,
            )

            # Add first version
            version = AnalysisVersion(
                version_number="1.0",
                specification=spec,
                lifecycle=AnalysisLifecycle.DRAFT,
            )
            entry.version_history.add_version(version)

            plan.add_analysis(entry)

        # Add plan-level warnings
        plan.warnings = list(candidate_set.warnings)
        if llm_result:
            plan.warnings.extend(llm_result.get("general_warnings", []))

        # Audit event
        plan.add_audit_event(
            "plan_created",
            details={"n_analyses": len(plan.analyses)},
        )

        return plan

    def _default_class(self, candidate) -> str:
        """Determine default classification from candidate characteristics."""
        if candidate.confidence == "high" and candidate.sample_size_adequate:
            return "primary"
        elif candidate.confidence == "moderate":
            return "sensitivity"
        else:
            return "exploratory"

    def _build_spec(
        self,
        context: dict,
        profile: DataProfile,
        candidate,
        analysis_class: str,
        rationale: str,
    ) -> StatisticalAnalysisSpecification:
        """Build a StatisticalAnalysisSpecification from a candidate.

        Method-specific specification:
        - Group comparison methods (t-test, Mann-Whitney, ANOVA) use a single
          grouping variable in subgroup_variables, not a list of predictors.
        - Regression methods use the full predictor list.
        - Correlation methods use a single predictor.
        """
        method_id = candidate.method_id

        # Identify group comparison methods that expect a single grouping variable
        group_comparison_methods = {
            "independent_t_test", "paired_t_test", "mann_whitney_u",
            "wilcoxon_signed_rank", "one_way_anova", "chi_square",
            "fisher_exact", "log_rank",
        }

        # Identify correlation methods that expect a single predictor
        correlation_methods = {"pearson_correlation", "spearman_correlation"}

        # Determine predictors and subgroup variables based on method type
        if method_id in group_comparison_methods:
            # Use grouping variable, not regression predictors
            # Backward compatibility: if grouping_variables is empty, use
            # binary/categorical predictors as grouping variables
            subgroup_vars = profile.grouping_variables
            if not subgroup_vars:
                for i, ptype in enumerate(profile.predictor_types):
                    if ptype in ("binary", "categorical"):
                        subgroup_vars.append(profile.predictor_variables[i])
            predictors = []
        elif method_id in correlation_methods:
            # Use a single predictor
            predictors = profile.predictor_variables[:1] if profile.predictor_variables else []
            subgroup_vars = []
        else:
            # Regression methods: use full predictor list plus grouping variables
            # (treatment assignment) as predictors
            predictors = list(profile.predictor_variables)
            # Add grouping variables (e.g., Treatment_Arm) as predictors for regression
            for gv in profile.grouping_variables:
                if gv not in predictors:
                    predictors.append(gv)
            subgroup_vars = []

        return StatisticalAnalysisSpecification(
            project_id=context.get("project_id"),
            dataset_id=context.get("dataset_id"),
            research_question=profile.research_question,
            study_design=profile.study_design,
            primary_outcome=profile.outcome_variable,
            predictors=predictors,
            covariates=profile.covariates,
            subgroup_variables=subgroup_vars,
            analysis_method=method_id,
            method_category=analysis_class,
            rationale=rationale,
            assumptions=[
                {"name": a, "description": a, "status": "untested"}
                for a in candidate.assumptions
            ],
            alpha=0.05,
            confidence_level=0.95,
        )

    def _build_trace(
        self, candidate, profile, analysis_class, concerns
    ) -> AnalysisDecisionTrace:
        """Build a structured decision trace."""
        trace = AnalysisDecisionTrace(analysis_id=candidate.method_id)

        trace.add_step(
            "outcome_type",
            f"Outcome '{profile.outcome_variable}' is {profile.outcome_type}",
            "info",
        )
        trace.add_step(
            "sample_size",
            f"Dataset has {profile.n_observations} observations",
            "info" if candidate.sample_size_adequate else "warning",
        )
        trace.add_step(
            "registry_match",
            f"Method '{candidate.method_id}' supports outcome type '{profile.outcome_type}'",
            "pass",
        )

        if profile.study_design:
            trace.add_step(
                "design_compatibility",
                f"Study design '{profile.study_design}' — "
                f"{'compatible' if candidate.design_compatible else 'not in compatible list'}",
                "pass" if candidate.design_compatible else "warning",
            )

        for concern in concerns:
            trace.add_step("methodological_concern", concern, "warning")

        trace.set_final(candidate.method_id)
        trace.alternatives_considered = candidate.alternatives

        return trace

    # ----------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------

    def _validate_all_entries(self, plan: AnalysisPlan, file_path: str | None):
        """Validate every analysis entry in the plan."""
        import pandas as pd
        from pathlib import Path

        df = None
        if file_path:
            try:
                p = Path(file_path)
                if p.suffix == ".csv":
                    df = pd.read_csv(p)
                elif p.suffix == ".tsv":
                    df = pd.read_csv(p, sep="\t")
                elif p.suffix in (".xlsx", ".xls"):
                    df = pd.read_excel(p)
                elif p.suffix == ".parquet":
                    df = pd.read_parquet(p)
            except Exception:
                pass

        for entry in plan.analyses:
            spec = entry.get_current_spec()
            if spec is None:
                continue

            cv = entry.version_history.get_current()
            cv.lifecycle = AnalysisLifecycle.VALIDATING

            if df is not None:
                result = self.validator.validate(spec, data_frame=df)
            else:
                result = self.validator.validate(spec)

            cv.validation_status = result.status
            cv.validation_warnings = result.warnings
            cv.validation_blockers = result.blockers

            if result.status == ValidationStatus.BLOCK:
                cv.lifecycle = AnalysisLifecycle.VALIDATION_FAILED
            elif result.status == ValidationStatus.WARNING:
                cv.lifecycle = AnalysisLifecycle.NEEDS_REVIEW
            else:
                cv.lifecycle = AnalysisLifecycle.VALIDATED

            # Audit event
            plan.add_audit_event(
                "candidate_validated",
                analysis_id=entry.analysis_id,
                details={
                    "method": spec.analysis_method,
                    "validation_status": result.status.value,
                    "n_warnings": len(result.warnings),
                    "n_blockers": len(result.blockers),
                },
            )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------

    def _entry_summary(self, entry: AnalysisEntry) -> dict:
        """Produce a concise summary of an AnalysisEntry.

        Includes everything downstream consumers need to review and execute
        the analysis: the persisted plan is what /api/analysis-plan/{id}/execute
        reconstructs the specification from, and the frontend renders
        predictors/covariates/warnings from this same dict.
        """
        cv = entry.version_history.get_current()
        spec = cv.specification if cv else None

        return {
            "analysis_id": entry.analysis_id,
            "analysis_name": entry.analysis_name,
            "analysis_class": entry.analysis_class,
            "method": spec.analysis_method if spec else None,
            "outcome": spec.primary_outcome if spec else None,
            "predictors": list(spec.predictors) if spec else [],
            "covariates": list(spec.covariates) if spec else [],
            "subgroup_variables": list(spec.subgroup_variables) if spec else [],
            "rationale": spec.rationale if spec else "",
            "lifecycle": cv.lifecycle.value if cv else "unknown",
            "validation": cv.validation_status.value if cv else "unknown",
            "approval": cv.approval_status.value if cv else "unknown",
            "prespecified": entry.prespecified_status,
            "warnings": list(cv.validation_warnings) if cv else [],
            "validation_blockers": list(cv.validation_blockers) if cv else [],
            "n_warnings": len(cv.validation_warnings) if cv else 0,
            "n_blockers": len(cv.validation_blockers) if cv else 0,
        }
