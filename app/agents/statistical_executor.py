"""Statistical Executor Agent — run deterministic analyses via the statistical engine.

Guardrails:
- Must receive a valid StatisticalAnalysisSpecification
- Specification must be LOCKED or EXECUTED lifecycle state
- Approval must be APPROVED or LOCKED
- Dataset must exist and match the specification's dataset_id
- All required variables must be present
"""

from __future__ import annotations

import json
from pathlib import Path

from app.agents.base import BaseAgent, AgentResult
from app.statistics import engine as stat_engine
from app.statistics.specification import (
    StatisticalAnalysisSpecification,
    ValidationStatus,
    ApprovalStatus,
)
from app.statistics.plan import AnalysisLifecycle
from app.statistics.registry import get_method


class ExecutionGuardError(Exception):
    """Raised when an execution guardrail is violated."""
    def __init__(self, guard: str, message: str):
        self.guard = guard
        self.message = message
        super().__init__(f"Guard '{guard}': {message}")


class StatisticalExecutorAgent(BaseAgent):
    name = "statistical_executor"
    requires_llm = False
    description = "Execute a specified statistical analysis and return structured results."

    async def _execute(self, context: dict) -> AgentResult:
        # ================================================================
        # Guardrail checks
        # ================================================================
        spec_dict = context.get("specification")
        lifecycle = context.get("lifecycle", "")
        file_path = context.get("file_path")

        # If a full specification is provided, run guardrails
        if spec_dict:
            try:
                self._run_guardrails(spec_dict, lifecycle, file_path)
            except ExecutionGuardError as e:
                return AgentResult(
                    success=False,
                    data={
                        "error": f"Execution blocked: {e.message}",
                        "guard": e.guard,
                        "specification": spec_dict.get("analysis_method", "unknown"),
                    },
                    warnings=[f"Guard violated: {e.guard}"],
                )

            # Extract analysis parameters from specification
            spec = StatisticalAnalysisSpecification(**spec_dict) if isinstance(spec_dict, dict) else spec_dict
            analysis_type = spec.analysis_method
            context = self._spec_to_context(spec, file_path or context.get("file_path"), context)
        else:
            # Legacy path: direct context-based execution
            analysis_type = context.get("analysis_type", "")

        file_path_obj = Path(context.get("file_path", ""))

        dispatch = {
            "descriptive": self._run_descriptive,
            "independent_t_test": self._run_independent_t_test,
            "paired_t_test": self._run_paired_t_test,
            "chi_square": self._run_chi_square,
            "fisher_exact": self._run_fisher_exact,
            "correlation": self._run_correlation,
            "pearson_correlation": lambda fp, ctx: self._run_correlation(fp, {**ctx, "method": "pearson"}),
            "spearman_correlation": lambda fp, ctx: self._run_correlation(fp, {**ctx, "method": "spearman"}),
            "one_way_anova": self._run_one_way_anova,
            "logistic_regression": self._run_logistic_regression,
            "linear_regression": self._run_linear_regression,
            "mann_whitney_u": self._run_mann_whitney,
            "wilcoxon_signed_rank": self._run_wilcoxon,
            "power_analysis": self._run_power_analysis,
        }

        handler = dispatch.get(analysis_type)
        if handler is None:
            return AgentResult(
                success=False,
                data={"error": f"Unknown analysis type: {analysis_type}"},
                warnings=[f"Unsupported analysis: {analysis_type}"],
            )

        try:
            result = handler(file_path_obj, context)
            return AgentResult(
                success=True,
                data=result,
                warnings=result.get("warnings", []),
                confidence="high",
            )
        except Exception as e:
            return AgentResult(
                success=False,
                data={"error": f"Analysis failed: {e}"},
                warnings=[str(e)],
            )

    # ----------------------------------------------------------------
    # Guardrails
    # ----------------------------------------------------------------

    def _run_guardrails(self, spec_dict: dict, lifecycle: str, file_path: str | None):
        """Enforce execution guardrails. Raises ExecutionGuardError on failure."""
        spec = spec_dict if isinstance(spec_dict, dict) else spec_dict

        # 1. Method must be in registry
        method_id = spec.get("analysis_method", "") if isinstance(spec, dict) else getattr(spec, "analysis_method", "")
        method_entry = get_method(method_id)
        if method_entry is None:
            raise ExecutionGuardError(
                "unsupported_method",
                f"Method '{method_id}' is not in the statistical method registry.",
            )

        # 2. Lifecycle must be LOCKED or EXECUTED
        if lifecycle and lifecycle not in (AnalysisLifecycle.LOCKED, AnalysisLifecycle.EXECUTED):
            raise ExecutionGuardError(
                "invalid_lifecycle",
                f"Analysis lifecycle is '{lifecycle}'. "
                f"Execution requires LOCKED or EXECUTED state.",
            )

        # 3. Approval must be APPROVED or LOCKED
        approval = spec.get("approval_status", "") if isinstance(spec, dict) else getattr(spec, "approval_status", "")
        if approval and approval not in (ApprovalStatus.APPROVED, ApprovalStatus.LOCKED):
            raise ExecutionGuardError(
                "not_approved",
                f"Analysis approval status is '{approval}'. "
                f"Execution requires APPROVED or LOCKED status.",
            )

        # 4. Validation must not be BLOCK
        validation = spec.get("validation_status", "") if isinstance(spec, dict) else getattr(spec, "validation_status", "")
        if validation == ValidationStatus.BLOCK:
            raise ExecutionGuardError(
                "validation_blocked",
                "Analysis has validation BLOCK status. Cannot execute.",
            )

        # 5. Dataset must exist
        if file_path:
            p = Path(file_path)
            if not p.exists():
                raise ExecutionGuardError(
                    "dataset_missing",
                    f"Dataset file not found: {file_path}",
                )

        # 6. Dataset identity verification
        if file_path and Path(file_path).exists():
            self._verify_dataset_identity(spec, Path(file_path))

        # 7. Variables must exist in dataset (if file available)
        if file_path and Path(file_path).exists():
            self._check_variables_exist(spec, Path(file_path))

    def _verify_dataset_identity(self, spec: dict, file_path: Path):
        """Verify dataset identity matches the approved specification."""
        from app.data.dataset_identity import compute_dataset_identity, verify_dataset_identity

        expected_hash = spec.get("dataset_hash", "") if isinstance(spec, dict) else getattr(spec, "dataset_hash", "")
        expected_content_hash = spec.get("dataset_content_hash", None)
        expected_schema_hash = spec.get("dataset_schema_hash", None)
        expected_dataset_id = spec.get("dataset_id", None) if isinstance(spec, dict) else getattr(spec, "dataset_id", None)

        # If no hash is bound to the spec, skip verification (legacy path)
        if not expected_content_hash and not expected_hash:
            return

        identity = compute_dataset_identity(file_path)
        content_hash_to_check = expected_content_hash or expected_hash

        if content_hash_to_check and identity["content_hash"] != content_hash_to_check:
            raise ExecutionGuardError(
                "DATASET_INTEGRITY_MISMATCH",
                f"Dataset content hash does not match approved specification. "
                f"Expected: {content_hash_to_check[:16]}..., "
                f"Observed: {identity['content_hash'][:16]}... "
                f"The dataset may have been modified. Create a new analysis version "
                f"to re-approve against the current dataset.",
            )

        if expected_schema_hash and identity["schema_hash"] != expected_schema_hash:
            raise ExecutionGuardError(
                "DATASET_SCHEMA_MISMATCH",
                f"Dataset schema hash does not match approved specification. "
                f"Expected: {expected_schema_hash[:16]}..., "
                f"Observed: {identity['schema_hash'][:16]}... "
                f"The dataset columns or types may have changed.",
            )

    def _check_variables_exist(self, spec: dict, file_path: Path):
        """Check that required variables exist in the dataset."""
        import pandas as pd

        try:
            suffix = file_path.suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(file_path)
            elif suffix == ".tsv":
                df = pd.read_csv(file_path, sep="\t")
            elif suffix in (".xlsx", ".xls"):
                df = pd.read_excel(file_path)
            elif suffix == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                return
        except Exception:
            return

        outcome = spec.get("primary_outcome", "") if isinstance(spec, dict) else getattr(spec, "primary_outcome", "")
        predictors = spec.get("predictors", []) if isinstance(spec, dict) else getattr(spec, "predictors", [])
        covariates = spec.get("covariates", []) if isinstance(spec, dict) else getattr(spec, "covariates", [])

        required = [outcome] + predictors + covariates
        df_cols = set(df.columns)

        missing = [v for v in required if v and v not in df_cols]
        if missing:
            raise ExecutionGuardError(
                "missing_variables",
                f"Required variables not found in dataset: {', '.join(missing)}",
            )

    def _spec_to_context(self, spec: StatisticalAnalysisSpecification, file_path: str, base_context: dict) -> dict:
        """Convert a specification to executor context."""
        ctx = {
            "file_path": file_path,
            "analysis_type": spec.analysis_method,
            "outcome": spec.primary_outcome,
            "predictors": spec.predictors,
            "covariates": spec.covariates,
        }

        # Method-specific mappings
        if spec.analysis_method == "independent_t_test":
            # Grouping variable is in subgroup_variables, not predictors
            ctx["group"] = spec.subgroup_variables[0] if spec.subgroup_variables else None
        elif spec.analysis_method == "paired_t_test":
            if len(spec.predictors) >= 2:
                ctx["outcome1"] = spec.predictors[0]
                ctx["outcome2"] = spec.predictors[1]
        elif spec.analysis_method in ("chi_square", "fisher_exact"):
            # For chi-square, variable1 is the grouping variable, variable2 is the outcome
            if spec.subgroup_variables:
                ctx["variable1"] = spec.subgroup_variables[0]
                ctx["variable2"] = spec.primary_outcome
            elif len(spec.predictors) >= 1:
                ctx["variable1"] = spec.predictors[0]
                ctx["variable2"] = spec.primary_outcome
        elif spec.analysis_method in ("pearson_correlation", "spearman_correlation"):
            if len(spec.predictors) >= 1:
                ctx["x"] = spec.predictors[0]
                ctx["y"] = spec.primary_outcome
                ctx["method"] = spec.analysis_method.replace("_correlation", "")
        elif spec.analysis_method == "one_way_anova":
            # Grouping variable is in subgroup_variables, not predictors
            ctx["group"] = spec.subgroup_variables[0] if spec.subgroup_variables else None
        elif spec.analysis_method == "mann_whitney_u":
            # Grouping variable is in subgroup_variables, not predictors
            ctx["group"] = spec.subgroup_variables[0] if spec.subgroup_variables else None
        elif spec.analysis_method == "wilcoxon_signed_rank":
            if len(spec.predictors) >= 2:
                ctx["outcome1"] = spec.predictors[0]
                ctx["outcome2"] = spec.predictors[1]

        return ctx

    # ----------------------------------------------------------------
    # Analysis handlers (unchanged from Phase 1)
    # ----------------------------------------------------------------

    def _run_descriptive(self, fp: Path, ctx: dict) -> dict:
        return stat_engine.descriptive_statistics(
            fp, columns=ctx.get("columns"),
        )

    def _run_independent_t_test(self, fp: Path, ctx: dict) -> dict:
        return stat_engine.independent_t_test(
            fp,
            outcome=ctx["outcome"],
            group=ctx["group"],
            group1=ctx.get("group1"),
            group2=ctx.get("group2"),
            equal_var=ctx.get("equal_var", False),
            alternative=ctx.get("alternative", "two-sided"),
        )

    def _run_paired_t_test(self, fp: Path, ctx: dict) -> dict:
        return stat_engine.paired_t_test(
            fp,
            outcome1=ctx["outcome1"],
            outcome2=ctx["outcome2"],
            alternative=ctx.get("alternative", "two-sided"),
        )

    def _run_chi_square(self, fp: Path, ctx: dict) -> dict:
        return stat_engine.chi_square_test(
            fp,
            variable1=ctx["variable1"],
            variable2=ctx["variable2"],
        )

    def _run_fisher_exact(self, fp: Path, ctx: dict) -> dict:
        return stat_engine.fisher_exact_test(
            fp,
            variable1=ctx["variable1"],
            variable2=ctx["variable2"],
        )

    def _run_correlation(self, fp: Path, ctx: dict) -> dict:
        return stat_engine.correlation_test(
            fp,
            x=ctx["x"],
            y=ctx["y"],
            method=ctx.get("method", "pearson"),
        )

    def _run_one_way_anova(self, fp: Path, ctx: dict) -> dict:
        return stat_engine.one_way_anova(
            fp,
            outcome=ctx["outcome"],
            group=ctx["group"],
        )

    def _run_logistic_regression(self, fp: Path, ctx: dict) -> dict:
        return stat_engine.logistic_regression(
            fp,
            outcome=ctx["outcome"],
            predictors=ctx["predictors"],
            covariates=ctx.get("covariates"),
        )

    def _run_linear_regression(self, fp: Path, ctx: dict) -> dict:
        return stat_engine.linear_regression(
            fp,
            outcome=ctx["outcome"],
            predictors=ctx["predictors"],
            covariates=ctx.get("covariates"),
        )

    def _run_mann_whitney(self, fp: Path, ctx: dict) -> dict:
        return stat_engine.mann_whitney_u(
            fp,
            outcome=ctx["outcome"],
            group=ctx["group"],
            group1=ctx.get("group1"),
            group2=ctx.get("group2"),
            alternative=ctx.get("alternative", "two-sided"),
        )

    def _run_wilcoxon(self, fp: Path, ctx: dict) -> dict:
        return stat_engine.wilcoxon_signed_rank(
            fp,
            outcome1=ctx["outcome1"],
            outcome2=ctx["outcome2"],
            alternative=ctx.get("alternative", "two-sided"),
        )

    def _run_power_analysis(self, fp: Path, ctx: dict) -> dict:
        return stat_engine.power_analysis_t_test(
            effect_size=ctx.get("effect_size"),
            n_per_group=ctx.get("n_per_group"),
            alpha=ctx.get("alpha", 0.05),
            power=ctx.get("power", 0.80),
            alternative=ctx.get("alternative", "two-sided"),
        )
