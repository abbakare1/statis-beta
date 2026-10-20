"""Statistical Executor — deterministic execution engine with guardrails.

Wraps the raw statistical engine functions with:
1. Dataset verification (hash, schema, variables exist)
2. Input validation (types, ranges, completeness)
3. Numerical sanity checks (p-values in [0,1], correlations in [-1,1], etc.)
4. Standardized StatisticalResult output

This module NEVER uses the LLM. Every computation is deterministic.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.statistics.result import (
    StatisticalResult,
    ExecutionStatus,
    SanityStatus,
    NumericalSanity,
    DataVerification,
)
from app.statistics.registry import get_method


# ---------------------------------------------------------------------------
# Dataset Verification
# ---------------------------------------------------------------------------

def verify_dataset(
    file_path: Path,
    expected_content_hash: str | None = None,
    expected_schema_hash: str | None = None,
    expected_dataset_id: str | None = None,
    expected_version: str | None = None,
    variables_required: list[str] | None = None,
) -> DataVerification:
    """Verify dataset identity and compute hashes.

    Returns DataVerification with hashes, shape, and any warnings.

    If expected_content_hash or expected_schema_hash are provided and do not
    match, identity_verified is set to False and the mismatch is recorded.
    """
    verification = DataVerification()

    try:
        df = _load_dataframe(file_path)
    except Exception as e:
        verification.verification_warnings.append(f"Failed to load dataset: {e}")
        return verification

    verification.n_rows = len(df)
    verification.n_cols = len(df.columns)
    verification.columns = list(df.columns)

    # Compute content hash (all data)
    try:
        content_bytes = df.to_csv(index=False).encode()
        verification.content_hash = hashlib.sha256(content_bytes).hexdigest()
    except Exception:
        verification.verification_warnings.append("Failed to compute content hash")

    # Compute schema hash (column names + dtypes)
    try:
        schema = {col: str(df[col].dtype) for col in df.columns}
        schema_bytes = json.dumps(schema, sort_keys=True).encode()
        verification.schema_hash = hashlib.sha256(schema_bytes).hexdigest()
    except Exception:
        verification.verification_warnings.append("Failed to compute schema hash")

    # Verify identity — record mismatches
    mismatch_reasons = []

    if expected_content_hash and verification.content_hash:
        if verification.content_hash != expected_content_hash:
            mismatch_reasons.append(
                f"CONTENT_HASH_MISMATCH: expected={expected_content_hash[:16]}..., "
                f"observed={verification.content_hash[:16]}..."
            )
            verification.verification_warnings.append(mismatch_reasons[-1])

    if expected_schema_hash and verification.schema_hash:
        if verification.schema_hash != expected_schema_hash:
            mismatch_reasons.append(
                f"SCHEMA_HASH_MISMATCH: expected={expected_schema_hash[:16]}..., "
                f"observed={verification.schema_hash[:16]}..."
            )
            verification.verification_warnings.append(mismatch_reasons[-1])

    if expected_dataset_id:
        # Dataset ID is informational — record but don't block on it alone
        if not expected_dataset_id:
            mismatch_reasons.append("DATASET_ID_EMPTY: expected dataset_id was empty")

    if expected_version:
        if not expected_version:
            mismatch_reasons.append("VERSION_EMPTY: expected version was empty")

    # Set identity verification status
    if mismatch_reasons:
        verification.identity_verified = False
        verification.identity_mismatch_reason = "; ".join(mismatch_reasons)
    elif expected_content_hash or expected_schema_hash:
        # Expected hashes were provided and all matched
        verification.identity_verified = True
        verification.verified = True
    else:
        # No expected hashes provided — identity cannot be verified
        verification.identity_verified = False
        verification.verified = False

    # Check required variables
    if variables_required:
        verification.variables_used = variables_required
        missing = [v for v in variables_required if v not in df.columns]
        if missing:
            verification.verification_warnings.append(
                f"Required variables not found: {', '.join(missing)}"
            )

    # Complete cases
    if variables_required:
        cols_in_df = [v for v in variables_required if v in df.columns]
        if cols_in_df:
            verification.n_complete_cases = int(df[cols_in_df].dropna().shape[0])

    return verification


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

def validate_inputs(
    df: pd.DataFrame,
    method_id: str,
    outcome: str | None = None,
    predictors: list[str] | None = None,
    group: str | None = None,
    variables: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Validate inputs before execution.

    Returns (warnings, blockers).
    """
    warnings = []
    blockers = []

    all_vars = []
    if outcome:
        all_vars.append(outcome)
    if predictors:
        all_vars.extend(predictors)
    if group:
        all_vars.append(group)
    if variables:
        all_vars.extend(variables)

    # Remove duplicates
    all_vars = list(dict.fromkeys(all_vars))

    # Check variable existence
    for var in all_vars:
        if var not in df.columns:
            blockers.append(f"Variable '{var}' not found in dataset.")

    if blockers:
        return warnings, blockers

    # Check method requirements
    method = get_method(method_id)
    if method is None:
        blockers.append(f"Unknown method '{method_id}' not in registry.")
        return warnings, blockers

    # Check sample size
    n = len(df)
    if n < method.min_observations:
        blockers.append(
            f"Dataset has {n} observations, but {method.name} requires at least {method.min_observations}."
        )

    # Check group requirements
    if group and group in df.columns:
        n_groups = df[group].nunique()
        if method.min_groups > 1 and n_groups < method.min_groups:
            blockers.append(
                f"Variable '{group}' has {n_groups} groups, but {method.name} requires at least {method.min_groups}."
            )

    # Check for complete cases
    if all_vars:
        n_complete = df[all_vars].dropna().shape[0]
        if n_complete == 0:
            blockers.append("No complete cases available for analysis.")
        elif n_complete < method.min_observations:
            blockers.append(
                f"Only {n_complete} complete cases, but {method.name} requires at least {method.min_observations}."
            )
        elif n_complete < n * 0.5:
            warnings.append(
                f"Only {n_complete}/{n} observations ({n_complete/n*100:.1f}%) have complete data."
            )

    # Check for constant variables in predictors
    if predictors:
        for pred in predictors:
            if pred in df.columns and pd.api.types.is_numeric_dtype(df[pred]):
                if df[pred].nunique(dropna=True) <= 1:
                    blockers.append(f"Predictor '{var}' has no variation (constant).")

    return warnings, blockers


# ---------------------------------------------------------------------------
# Numerical Sanity Validation
# ---------------------------------------------------------------------------

def validate_numerical_sanity(
    method_id: str,
    result: dict[str, Any],
) -> NumericalSanity:
    """Run deterministic sanity checks on computed values.

    Checks:
    - p-values in [0, 1]
    - correlations in [-1, 1]
    - effect sizes are finite
    - sample sizes are positive
    - degrees of freedom are positive
    """
    sanity = NumericalSanity()
    checks = []

    # p-value check
    p_val = result.get("p_value")
    if p_val is not None:
        check = {"name": "p_value_range", "value": p_val}
        if not np.isfinite(p_val):
            sanity.blockers.append(f"p-value is not finite: {p_val}")
            check["status"] = "fail"
        elif p_val < 0 or p_val > 1:
            sanity.blockers.append(f"p-value out of [0,1] range: {p_val}")
            check["status"] = "fail"
        else:
            check["status"] = "pass"
        checks.append(check)

    # Correlation check
    r_val = result.get("r")
    if r_val is not None:
        check = {"name": "correlation_range", "value": r_val}
        if not np.isfinite(r_val):
            sanity.blockers.append(f"Correlation is not finite: {r_val}")
            check["status"] = "fail"
        elif abs(r_val) > 1.0:
            sanity.blockers.append(f"Correlation out of [-1,1] range: {r_val}")
            check["status"] = "fail"
        else:
            check["status"] = "pass"
        checks.append(check)

    # Effect size check (Cohen's d, odds ratio, etc.)
    for key in ("cohens_d", "effect_size", "odds_ratio"):
        val = result.get(key)
        if val is not None:
            check = {"name": f"{key}_finite", "value": val}
            if not np.isfinite(val):
                sanity.warnings.append(f"{key} is not finite: {val}")
                check["status"] = "warning"
            else:
                check["status"] = "pass"
            checks.append(check)

    # Sample size checks
    for key in ("n_group1", "n_group2", "n_pairs", "n", "n_obs"):
        val = result.get(key)
        if val is not None:
            check = {"name": f"{key}_positive", "value": val}
            if val <= 0:
                sanity.blockers.append(f"{key} is not positive: {val}")
                check["status"] = "fail"
            else:
                check["status"] = "pass"
            checks.append(check)

    # Degrees of freedom check
    df_val = result.get("df")
    if df_val is not None:
        check = {"name": "df_positive", "value": df_val}
        if not np.isfinite(df_val):
            sanity.blockers.append(f"Degrees of freedom is not finite: {df_val}")
            check["status"] = "fail"
        elif df_val <= 0:
            sanity.warnings.append(f"Degrees of freedom is not positive: {df_val}")
            check["status"] = "warning"
        else:
            check["status"] = "pass"
        checks.append(check)

    # F-statistic check (ANOVA, regression)
    f_val = result.get("f_statistic")
    if f_val is not None:
        check = {"name": "f_statistic_nonneg", "value": f_val}
        if not np.isfinite(f_val):
            sanity.blockers.append(f"F-statistic is not finite: {f_val}")
            check["status"] = "fail"
        elif f_val < 0:
            sanity.blockers.append(f"F-statistic is negative: {f_val}")
            check["status"] = "fail"
        else:
            check["status"] = "pass"
        checks.append(check)

    # Chi-square check
    chi2_val = result.get("chi2")
    if chi2_val is not None:
        check = {"name": "chi2_nonneg", "value": chi2_val}
        if not np.isfinite(chi2_val):
            sanity.blockers.append(f"Chi-square is not finite: {chi2_val}")
            check["status"] = "fail"
        elif chi2_val < 0:
            sanity.blockers.append(f"Chi-square is negative: {chi2_val}")
            check["status"] = "fail"
        else:
            check["status"] = "pass"
        checks.append(check)

    # U-statistic check (Mann-Whitney)
    u_val = result.get("u_statistic")
    if u_val is not None:
        check = {"name": "u_statistic_nonneg", "value": u_val}
        if not np.isfinite(u_val):
            sanity.blockers.append(f"U-statistic is not finite: {u_val}")
            check["status"] = "fail"
        elif u_val < 0:
            sanity.blockers.append(f"U-statistic is negative: {u_val}")
            check["status"] = "fail"
        else:
            check["status"] = "pass"
        checks.append(check)

    # Confidence interval check
    ci_lower = result.get("ci_95_lower")
    ci_upper = result.get("ci_95_upper")
    if ci_lower is not None and ci_upper is not None:
        check = {"name": "ci_order", "lower": ci_lower, "upper": ci_upper}
        if ci_lower > ci_upper:
            sanity.warnings.append(f"CI lower ({ci_lower}) > CI upper ({ci_upper})")
            check["status"] = "warning"
        else:
            check["status"] = "pass"
        checks.append(check)

    sanity.checks = checks
    sanity.status = SanityStatus.FAIL if sanity.blockers else (
        SanityStatus.WARNING if sanity.warnings else SanityStatus.PASS
    )

    return sanity


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------

class StatisticalExecutor:
    """Deterministic execution engine with guardrails.

    Every method:
    1. Verifies the dataset
    2. Validates inputs
    3. Executes the computation
    4. Runs numerical sanity checks
    5. Returns a standardized StatisticalResult
    """

    def execute(
        self,
        method_id: str,
        file_path: Path,
        *,
        outcome: str | None = None,
        predictors: list[str] | None = None,
        group: str | None = None,
        group1: Any = None,
        group2: Any = None,
        variable1: str | None = None,
        variable2: str | None = None,
        x: str | None = None,
        y: str | None = None,
        outcome1: str | None = None,
        outcome2: str | None = None,
        covariates: list[str] | None = None,
        method: str | None = None,
        equal_var: bool = False,
        alternative: str = "two-sided",
        event_col: str | None = None,
        ci_alpha: float = 0.05,
        expected_content_hash: str | None = None,
        expected_schema_hash: str | None = None,
        expected_dataset_id: str | None = None,
        expected_version: str | None = None,
    ) -> StatisticalResult:
        """Execute a statistical method with full guardrails.

        Returns a StatisticalResult regardless of success or failure.

        If expected_content_hash or expected_schema_hash are provided and
        do not match the actual dataset, execution is BLOCKED — no
        statistical computation occurs.
        """
        # Get method info from registry
        method_entry = get_method(method_id)
        method_name = method_entry.name if method_entry else method_id

        # Build list of variables for verification
        variables = []
        if outcome:
            variables.append(outcome)
        if predictors:
            variables.extend(predictors)
        if group:
            variables.append(group)
        if variable1:
            variables.append(variable1)
        if variable2:
            variables.append(variable2)
        if x:
            variables.append(x)
        if y:
            variables.append(y)
        if outcome1:
            variables.append(outcome1)
        if outcome2:
            variables.append(outcome2)
        if covariates:
            variables.extend(covariates)
        if event_col:
            variables.append(event_col)
        variables = list(dict.fromkeys(variables))  # dedupe

        # Step 1: Verify dataset identity
        data_verification = verify_dataset(
            file_path,
            expected_content_hash=expected_content_hash,
            expected_schema_hash=expected_schema_hash,
            expected_dataset_id=expected_dataset_id,
            expected_version=expected_version,
            variables_required=variables,
        )

        # Step 1a: BLOCK on identity mismatch — no computation occurs
        if expected_content_hash or expected_schema_hash:
            if not data_verification.identity_verified:
                block_reason = (
                    f"Dataset identity verification failed. "
                    f"{data_verification.identity_mismatch_reason or 'Unknown mismatch'}. "
                    f"Execution is BLOCKED. Create a new dataset identity and "
                    f"revalidate the analysis before execution can proceed."
                )
                return StatisticalResult(
                    method_id=method_id,
                    method_name=method_name,
                    execution_status=ExecutionStatus.BLOCKED,
                    data_verification=data_verification,
                    guardrails_passed=False,
                    guardrail_blockers=[block_reason],
                )

        # Step 2: Load data and validate inputs
        try:
            df = _load_dataframe(file_path)
        except Exception as e:
            return StatisticalResult(
                method_id=method_id,
                method_name=method_name,
                execution_status=ExecutionStatus.FAILED,
                data_verification=data_verification,
                guardrails_passed=False,
                guardrail_blockers=[f"Failed to load dataset: {e}"],
            )

        warnings, blockers = validate_inputs(
            df, method_id,
            outcome=outcome, predictors=predictors,
            group=group, variables=variables,
        )

        if blockers:
            return StatisticalResult(
                method_id=method_id,
                method_name=method_name,
                execution_status=ExecutionStatus.BLOCKED,
                data_verification=data_verification,
                guardrails_passed=False,
                guardrail_warnings=warnings,
                guardrail_blockers=blockers,
            )

        # Step 3: Execute computation
        result_dict = {}
        computation_warnings = []

        try:
            result_dict = self._dispatch(
                method_id, file_path,
                outcome=outcome, predictors=predictors,
                group=group, group1=group1, group2=group2,
                variable1=variable1, variable2=variable2,
                x=x, y=y, outcome1=outcome1, outcome2=outcome2,
                covariates=covariates, method=method,
                equal_var=equal_var, alternative=alternative,
                event_col=event_col, ci_alpha=ci_alpha,
            )
        except Exception as e:
            return StatisticalResult(
                method_id=method_id,
                method_name=method_name,
                execution_status=ExecutionStatus.FAILED,
                data_verification=data_verification,
                guardrails_passed=True,
                guardrail_warnings=warnings,
                computation_warnings=[f"Execution failed: {e}"],
            )

        # Check for engine-level errors
        if "error" in result_dict:
            return StatisticalResult(
                method_id=method_id,
                method_name=method_name,
                execution_status=ExecutionStatus.FAILED,
                data_verification=data_verification,
                guardrails_passed=True,
                guardrail_warnings=warnings,
                computation_warnings=[result_dict["error"]],
                engine_result=result_dict,
            )

        # Extract warnings from engine
        computation_warnings.extend(result_dict.get("warnings", []))

        # Step 4: Numerical sanity validation
        numerical_sanity = validate_numerical_sanity(method_id, result_dict)

        # Step 5: Statistical diagnostics
        from app.statistics.method_diagnostics import run_method_diagnostics
        from app.statistics.diagnostics import DiagnosticDiagnosticStatus, build_diagnostic_summary
        from app.statistics.defensibility import assess_defensibility

        diag_results = run_method_diagnostics(
            method_id, df, result_dict,
            outcome=outcome, group=group, predictors=predictors,
            variable1=variable1, variable2=variable2,
            outcome1=outcome1, outcome2=outcome2,
            x=x, y=y, group1=group1, group2=group2,
            event_col=event_col,
        )
        assumptions_checked = list({d.diagnostic_name for d in diag_results})
        diag_summary = build_diagnostic_summary(diag_results, assumptions_checked=assumptions_checked)

        # Step 6: Build StatisticalResult
        stat_result = self._build_result(
            method_id=method_name,
            result_dict=result_dict,
            data_verification=data_verification,
            numerical_sanity=numerical_sanity,
            guardrail_warnings=warnings,
            computation_warnings=computation_warnings,
        )

        # Step 7: Add diagnostics to result
        stat_result.diagnostic_status = diag_summary.overall_status.value
        stat_result.diagnostic_summary = {
            "overall_status": diag_summary.overall_status.value,
            "n_diagnostics": len(diag_summary.diagnostics),
            "n_warnings": len(diag_summary.warnings),
            "n_review_required": len(diag_summary.review_required),
            "n_errors": len(diag_summary.errors),
            "assumptions_checked": diag_summary.assumptions_checked,
            "assumptions_failed": diag_summary.assumptions_failed,
            "assumptions_uncertain": diag_summary.assumptions_uncertain,
            "diagnostics": [d.model_dump() for d in diag_summary.diagnostics],
        }
        stat_result.methodological_warnings = diag_summary.warnings + diag_summary.review_required
        stat_result.review_required = diag_summary.review_required

        # Step 8: Defensibility assessment
        survival_meta = None
        if method_name in ("kaplan_meier", "log_rank", "cox_proportional_hazards"):
            n_events = result_dict.get("n_events", 0)
            n_total = result_dict.get("n_total", 0) or result_dict.get("n_obs", 0)
            ph_test = result_dict.get("ph_test")
            survival_meta = {
                "method_id": method_name,
                "n_events": n_events,
                "n_censored": result_dict.get("n_censored", 0),
                "n_total": n_total,
                "ph_assumption_met": (ph_test == "passed") if ph_test is not None else None,
                "sparse_events": n_events > 0 and n_events < 10,
                "all_events": n_events == n_total and n_total > 0,
                "no_events": n_events == 0,
            }
        stat_result.defensibility_status = assess_defensibility(
            stat_result.execution_status,
            numerical_sanity.status,
            diag_summary.overall_status,
            execution_blocked=(stat_result.execution_status != ExecutionStatus.SUCCESS),
            survival_metadata=survival_meta,
        ).value

        # Compute fingerprint (incorporates diagnostics)
        stat_result.compute_fingerprint()

        return stat_result

    def _dispatch(
        self,
        method_id: str,
        file_path: Path,
        **kwargs,
    ) -> dict[str, Any]:
        """Dispatch to the appropriate engine function."""
        from app.statistics.engine import (
            independent_t_test,
            paired_t_test,
            chi_square_test,
            fisher_exact_test,
            correlation_test,
            one_way_anova,
            logistic_regression,
            linear_regression,
            mann_whitney_u,
            wilcoxon_signed_rank,
        )
        from app.statistics.survival import (
            kaplan_meier,
            log_rank,
            cox_proportional_hazards,
        )

        dispatch_map = {
            "independent_t_test": lambda: independent_t_test(
                file_path,
                outcome=kwargs["outcome"],
                group=kwargs["group"],
                group1=kwargs.get("group1"),
                group2=kwargs.get("group2"),
                equal_var=kwargs.get("equal_var", False),
                alternative=kwargs.get("alternative", "two-sided"),
            ),
            "paired_t_test": lambda: paired_t_test(
                file_path,
                outcome1=kwargs["outcome1"],
                outcome2=kwargs["outcome2"],
                alternative=kwargs.get("alternative", "two-sided"),
            ),
            "chi_square": lambda: chi_square_test(
                file_path,
                variable1=kwargs["variable1"],
                variable2=kwargs["variable2"],
            ),
            "fisher_exact": lambda: fisher_exact_test(
                file_path,
                variable1=kwargs["variable1"],
                variable2=kwargs["variable2"],
            ),
            "pearson_correlation": lambda: correlation_test(
                file_path,
                x=kwargs["x"],
                y=kwargs["y"],
                method="pearson",
            ),
            "spearman_correlation": lambda: correlation_test(
                file_path,
                x=kwargs["x"],
                y=kwargs["y"],
                method="spearman",
            ),
            "one_way_anova": lambda: one_way_anova(
                file_path,
                outcome=kwargs["outcome"],
                group=kwargs["group"],
            ),
            "linear_regression": lambda: linear_regression(
                file_path,
                outcome=kwargs["outcome"],
                predictors=kwargs["predictors"],
                covariates=kwargs.get("covariates"),
            ),
            "logistic_regression": lambda: logistic_regression(
                file_path,
                outcome=kwargs["outcome"],
                predictors=kwargs["predictors"],
                covariates=kwargs.get("covariates"),
            ),
            "mann_whitney_u": lambda: mann_whitney_u(
                file_path,
                outcome=kwargs["outcome"],
                group=kwargs["group"],
                group1=kwargs.get("group1"),
                group2=kwargs.get("group2"),
                alternative=kwargs.get("alternative", "two-sided"),
            ),
            "wilcoxon_signed_rank": lambda: wilcoxon_signed_rank(
                file_path,
                outcome1=kwargs["outcome1"],
                outcome2=kwargs["outcome2"],
                alternative=kwargs.get("alternative", "two-sided"),
            ),
            # Survival methods
            "kaplan_meier": lambda: kaplan_meier(
                file_path,
                time_col=kwargs["outcome"],
                event_col=kwargs["event_col"],
                group_col=kwargs.get("group"),
                ci_alpha=kwargs.get("ci_alpha", 0.05),
            ),
            "log_rank": lambda: log_rank(
                file_path,
                time_col=kwargs["outcome"],
                event_col=kwargs["event_col"],
                group_col=kwargs["group"],
                group1=kwargs.get("group1"),
                group2=kwargs.get("group2"),
            ),
            "cox_proportional_hazards": lambda: cox_proportional_hazards(
                file_path,
                time_col=kwargs["outcome"],
                event_col=kwargs["event_col"],
                covariates=kwargs.get("predictors", []),
                group_col=kwargs.get("group"),
            ),
        }

        if method_id not in dispatch_map:
            raise ValueError(f"Unknown method_id: {method_id}")

        return dispatch_map[method_id]()

    def _build_result(
        self,
        method_id: str,
        result_dict: dict[str, Any],
        data_verification: DataVerification,
        numerical_sanity: NumericalSanity,
        guardrail_warnings: list[str],
        computation_warnings: list[str],
    ) -> StatisticalResult:
        """Build a StatisticalResult from engine output."""
        # Extract common fields
        test_stat = None
        p_val = None
        effect = None
        effect_name = None
        ci = None
        df_val = None
        n_obs = None
        n_groups = None
        group_sizes = None
        descriptive = {}

        # Method-specific extraction — ordered from most specific to least

        # Cox proportional hazards
        if "hazard_ratios" in result_dict:
            n_obs = result_dict.get("n_obs")
            n_events = result_dict.get("n_events")
            descriptive = {
                "hazard_ratios": result_dict.get("hazard_ratios"),
                "coefficients": result_dict.get("coefficients"),
                "standard_errors": result_dict.get("standard_errors"),
                "n_events": n_events,
                "n_covariates": result_dict.get("n_covariates"),
                "events_per_covariate": result_dict.get("events_per_covariate"),
                "converged": result_dict.get("converged"),
                "concordance_index": result_dict.get("concordance_index"),
                "aic": result_dict.get("aic"),
                "log_likelihood": result_dict.get("log_likelihood"),
                "ph_test": result_dict.get("ph_test"),
                "ph_violations": result_dict.get("ph_violations"),
            }
            # Use the first covariate's HR as the primary effect for compatibility
            hrs = result_dict.get("hazard_ratios", {})
            if hrs:
                first_key = next(iter(hrs))
                effect = hrs[first_key]
                effect_name = f"Hazard ratio ({first_key})"
            p_val = None  # Cox overall p is not a single value

        # Log-rank test
        elif "degrees_of_freedom" in result_dict and "test_statistic" in result_dict and "group1" in result_dict:
            test_stat = result_dict.get("test_statistic")
            p_val = result_dict.get("p_value")
            df_val = result_dict.get("degrees_of_freedom")
            n_obs = result_dict.get("n_group1", 0) + result_dict.get("n_group2", 0)
            n_groups = 2
            group_sizes = {
                str(result_dict.get("group1", "g1")): result_dict.get("n_group1", 0),
                str(result_dict.get("group2", "g2")): result_dict.get("n_group2", 0),
            }
            descriptive = {
                "n_events_group1": result_dict.get("n_events_group1"),
                "n_events_group2": result_dict.get("n_events_group2"),
                "n_censored_group1": result_dict.get("n_censored_group1"),
                "n_censored_group2": result_dict.get("n_censored_group2"),
            }

        # Kaplan-Meier (has "groups" dict)
        elif "groups" in result_dict and "validation" in result_dict:
            n_obs = result_dict.get("n_complete") or result_dict.get("n_total")
            n_events = result_dict.get("n_events")
            n_censored = result_dict.get("n_censored")
            n_groups = result_dict.get("n_groups", 1)
            group_sizes = {}
            for g_name, g_data in result_dict.get("groups", {}).items():
                group_sizes[str(g_name)] = g_data.get("n", 0)
            descriptive = {
                "n_events": n_events,
                "n_censored": n_censored,
                "groups": {
                    g_name: {
                        "median_survival": g_data.get("median_survival"),
                        "n_events": g_data.get("n_events"),
                        "n_censored": g_data.get("n_censored"),
                    }
                    for g_name, g_data in result_dict.get("groups", {}).items()
                },
            }
            # KM doesn't have a single test statistic unless grouped
            # For ungrouped KM, there's no p-value

        elif "pseudo_r2" in result_dict:
            # Logistic regression
            test_stat = None
            p_val = result_dict.get("log_likelihood_ratio_p")
            effect = result_dict.get("pseudo_r2")
            effect_name = "Pseudo R-squared"
            n_obs = result_dict.get("n_obs")
            descriptive = {
                "pseudo_r2": result_dict.get("pseudo_r2"),
                "aic": result_dict.get("aic"),
                "bic": result_dict.get("bic"),
                "converged": result_dict.get("converged"),
            }

        elif "r_squared" in result_dict and "f_statistic" in result_dict:
            # Linear regression (has both r_squared and f_statistic)
            test_stat = result_dict["f_statistic"]
            p_val = result_dict.get("f_p_value")
            effect = result_dict.get("r_squared")
            effect_name = "R-squared"
            n_obs = result_dict.get("n_obs")
            descriptive = {
                "r_squared": result_dict.get("r_squared"),
                "adj_r_squared": result_dict.get("adj_r_squared"),
                "aic": result_dict.get("aic"),
                "bic": result_dict.get("bic"),
            }

        elif "t_statistic" in result_dict and "n_pairs" in result_dict:
            # Paired t-test (has t_statistic + n_pairs, not group1/group2)
            test_stat = result_dict["t_statistic"]
            p_val = result_dict.get("p_value")
            df_val = result_dict.get("df")
            n_obs = result_dict.get("n_pairs")
            n_groups = 2
            ci = (
                result_dict.get("ci_95_lower"),
                result_dict.get("ci_95_upper"),
            ) if result_dict.get("ci_95_lower") is not None else None
            descriptive = {
                "n_pairs": result_dict.get("n_pairs"),
                "mean_difference": result_dict.get("mean_difference"),
                "std_difference": result_dict.get("std_difference"),
                "se_difference": result_dict.get("se_difference"),
            }

        elif "t_statistic" in result_dict:
            # Independent t-test
            test_stat = result_dict["t_statistic"]
            p_val = result_dict.get("p_value")
            effect = result_dict.get("cohens_d")
            effect_name = "Cohen's d" if effect is not None else None
            df_val = result_dict.get("df")
            n_obs = result_dict.get("n_group1", 0) + result_dict.get("n_group2", 0)
            n_groups = 2
            group_sizes = {
                str(result_dict.get("group1", "g1")): result_dict.get("n_group1", 0),
                str(result_dict.get("group2", "g2")): result_dict.get("n_group2", 0),
            }
            ci = (
                result_dict.get("ci_95_lower"),
                result_dict.get("ci_95_upper"),
            ) if result_dict.get("ci_95_lower") is not None else None
            descriptive = {
                "mean_group1": result_dict.get("mean_group1"),
                "mean_group2": result_dict.get("mean_group2"),
                "std_group1": result_dict.get("std_group1"),
                "std_group2": result_dict.get("std_group2"),
                "difference": result_dict.get("difference"),
                "se_difference": result_dict.get("se_difference"),
            }

        elif "u_statistic" in result_dict:
            test_stat = result_dict["u_statistic"]
            p_val = result_dict.get("p_value")
            n_obs = result_dict.get("n_group1", 0) + result_dict.get("n_group2", 0)
            n_groups = 2
            group_sizes = {
                str(result_dict.get("group1", "g1")): result_dict.get("n_group1", 0),
                str(result_dict.get("group2", "g2")): result_dict.get("n_group2", 0),
            }
            descriptive = {
                "median_group1": result_dict.get("median_group1"),
                "median_group2": result_dict.get("median_group2"),
            }

        elif "r" in result_dict:
            test_stat = result_dict["r"]
            p_val = result_dict.get("p_value")
            effect = result_dict.get("r")
            effect_name = "r"
            n_obs = result_dict.get("n")
            ci = (
                result_dict.get("ci_95_lower"),
                result_dict.get("ci_95_upper"),
            ) if result_dict.get("ci_95_lower") is not None else None
            descriptive = {
                "r_squared": result_dict.get("r_squared"),
            }

        elif "chi2" in result_dict:
            test_stat = result_dict["chi2"]
            p_val = result_dict.get("p_value")
            df_val = result_dict.get("df")
            table = result_dict.get("table", {})
            if table:
                n_obs = sum(sum(row.values()) if isinstance(row, dict) else row for row in table.values())
            descriptive = {
                "table": table,
                "n_cells_low_expected": result_dict.get("n_cells_low_expected", 0),
            }

        elif "odds_ratio" in result_dict and "chi2" not in result_dict:
            test_stat = result_dict["odds_ratio"]
            p_val = result_dict.get("p_value")
            effect = result_dict.get("odds_ratio")
            effect_name = "Odds ratio"
            table = result_dict.get("table", {})
            if table:
                n_obs = sum(sum(row.values()) if isinstance(row, dict) else row for row in table.values())
            descriptive = {
                "table": table,
            }

        elif "f_statistic" in result_dict:
            # One-way ANOVA
            test_stat = result_dict["f_statistic"]
            p_val = result_dict.get("p_value")
            n_groups = result_dict.get("n_groups")
            descriptive = {
                "group_statistics": result_dict.get("group_stats", {}),
            }

        return StatisticalResult(
            method_id=method_id,
            method_name=method_id,
            execution_status=ExecutionStatus.SUCCESS,
            data_verification=data_verification,
            test_statistic=test_stat,
            p_value=p_val,
            effect_size=effect,
            effect_size_name=effect_name,
            confidence_interval=ci,
            degrees_of_freedom=df_val,
            n_observed=n_obs,
            n_groups=n_groups,
            group_sizes=group_sizes,
            descriptive=descriptive,
            engine_result=result_dict,
            numerical_sanity=numerical_sanity,
            guardrails_passed=True,
            guardrail_warnings=guardrail_warnings,
            computation_warnings=computation_warnings,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import json

def _load_dataframe(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    elif suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    elif suffix == ".parquet":
        return pd.read_parquet(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")
