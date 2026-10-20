"""Transformation Engine — deterministic, parameterized data transformations.

Supports: LOG, LOG1P, SQRT, STANDARDIZE, MIN_MAX, YEO_JOHNSON, BOX_COX, WINSORIZE.
All transformations are parameterized. Parameters are stored as fitted parameters.
Domain validation enforced. Original dataset never modified.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from app.statistics.data_quality import (
    TransformationType,
    TransformationSafety,
    TransformationCandidate,
    HandlingStrategy,
    HandlingAction,
)


# ---------------------------------------------------------------------------
# Domain Validation
# ---------------------------------------------------------------------------

def validate_transformation_domain(
    series: pd.Series,
    transformation: TransformationType,
) -> tuple[TransformationSafety, list[str], dict[str, Any]]:
    """Validate that a variable satisfies domain constraints for a transformation.

    Returns (safety, domain_violations, fitted_parameters).
    """
    valid = series.dropna()

    if len(valid) == 0:
        return TransformationSafety.BLOCKED, ["No valid values to transform."], {}

    if transformation == TransformationType.LOG:
        if (valid <= 0).any():
            n_violations = int((valid <= 0).sum())
            return (
                TransformationSafety.BLOCKED,
                [f"LOG requires x > 0. Found {n_violations} values <= 0."],
                {},
            )
        return TransformationSafety.SAFE, [], {}

    if transformation == TransformationType.LOG1P:
        if (valid < -1).any():
            n_violations = int((valid < -1).sum())
            return (
                TransformationSafety.BLOCKED,
                [f"LOG1P requires x >= -1. Found {n_violations} values < -1."],
                {},
            )
        return TransformationSafety.SAFE, [], {}

    if transformation == TransformationType.SQRT:
        if (valid < 0).any():
            n_violations = int((valid < 0).sum())
            return (
                TransformationSafety.BLOCKED,
                [f"SQRT requires x >= 0. Found {n_violations} negative values."],
                {},
            )
        return TransformationSafety.SAFE, [], {}

    if transformation == TransformationType.BOX_COX:
        if (valid <= 0).any():
            n_violations = int((valid <= 0).sum())
            return (
                TransformationSafety.BLOCKED,
                [f"Box-Cox requires x > 0. Found {n_violations} values <= 0."],
                {},
            )
        return TransformationSafety.SAFE, [], {}

    if transformation == TransformationType.YEO_JOHNSON:
        # Yeo-Johnson handles all real numbers
        return TransformationSafety.SAFE, [], {}

    if transformation == TransformationType.STANDARDIZE:
        return TransformationSafety.SAFE, [], {}

    if transformation == TransformationType.MIN_MAX:
        return TransformationSafety.SAFE, [], {}

    if transformation == TransformationType.WINSORIZE:
        return TransformationSafety.SAFE, [], {}

    return TransformationSafety.REQUIRES_REVIEW, [f"Unknown transformation: {transformation}"], {}


# ---------------------------------------------------------------------------
# Apply Transformations
# ---------------------------------------------------------------------------

def apply_transformation(
    df: pd.DataFrame,
    variable: str,
    transformation: TransformationType,
    parameters: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply a deterministic transformation to a variable.

    Returns (transformed_df, fitted_parameters). Original df is NOT modified.
    """
    if variable not in df.columns:
        raise ValueError(f"Variable '{variable}' not found in dataset.")

    result_df = df.copy()
    valid_mask = result_df[variable].notna()
    valid_values = result_df.loc[valid_mask, variable].copy()

    fitted_params: dict[str, Any] = {}

    if transformation == TransformationType.LOG:
        result_df.loc[valid_mask, variable] = np.log(valid_values.values)
        fitted_params = {"operation": "log", "base": "e"}

    elif transformation == TransformationType.LOG1P:
        result_df.loc[valid_mask, variable] = np.log1p(valid_values.values)
        fitted_params = {"operation": "log1p"}

    elif transformation == TransformationType.SQRT:
        result_df.loc[valid_mask, variable] = np.sqrt(valid_values.values)
        fitted_params = {"operation": "sqrt"}

    elif transformation == TransformationType.STANDARDIZE:
        mean = float(valid_values.mean())
        std = float(valid_values.std(ddof=1))
        if std > 0:
            result_df.loc[valid_mask, variable] = (valid_values.values - mean) / std
        else:
            result_df.loc[valid_mask, variable] = 0.0
        fitted_params = {"operation": "standardize", "mean": mean, "std": std}

    elif transformation == TransformationType.MIN_MAX:
        min_val = float(valid_values.min())
        max_val = float(valid_values.max())
        range_val = max_val - min_val
        if range_val > 0:
            result_df.loc[valid_mask, variable] = (valid_values.values - min_val) / range_val
        else:
            result_df.loc[valid_mask, variable] = 0.5
        fitted_params = {"operation": "min_max", "min": min_val, "max": max_val, "range": range_val}

    elif transformation == TransformationType.BOX_COX:
        try:
            transformed, lmbda = sp_stats.boxcox(valid_values.values)
            result_df.loc[valid_mask, variable] = transformed
            fitted_params = {"operation": "box_cox", "lambda": float(lmbda)}
        except Exception as e:
            raise ValueError(f"Box-Cox transformation failed: {e}")

    elif transformation == TransformationType.YEO_JOHNSON:
        try:
            transformed, lmbda = sp_stats.yeojohnson(valid_values.values)
            result_df.loc[valid_mask, variable] = transformed
            fitted_params = {"operation": "yeo_johnson", "lambda": float(lmbda)}
        except Exception as e:
            raise ValueError(f"Yeo-Johnson transformation failed: {e}")

    elif transformation == TransformationType.WINSORIZE:
        lower_pct = parameters.get("lower_pct", 0.05) if parameters else 0.05
        upper_pct = parameters.get("upper_pct", 0.95) if parameters else 0.95
        lower = float(valid_values.quantile(lower_pct))
        upper = float(valid_values.quantile(upper_pct))
        result_df.loc[valid_mask, variable] = valid_values.values.clip(lower, upper)
        fitted_params = {
            "operation": "winsorize",
            "lower_pct": lower_pct,
            "upper_pct": upper_pct,
            "lower_bound": lower,
            "upper_bound": upper,
        }

    else:
        raise ValueError(f"Unknown transformation: {transformation}")

    return result_df, fitted_params


# ---------------------------------------------------------------------------
# Impact Assessment
# ---------------------------------------------------------------------------

def compute_transformation_impact(
    original: pd.DataFrame,
    transformed: pd.DataFrame,
    variable: str,
) -> dict[str, Any]:
    """Compute deterministic impact summary for a transformation.

    Returns before/after comparison. Does NOT interpret significance.
    """
    orig_series = original[variable].dropna()
    trans_series = transformed[variable].dropna()

    impact: dict[str, Any] = {
        "variable": variable,
        "n_before": len(orig_series),
        "n_after": len(trans_series),
        "n_missing_before": int(original[variable].isna().sum()),
        "n_missing_after": int(transformed[variable].isna().sum()),
    }

    if pd.api.types.is_numeric_dtype(orig_series) and len(orig_series) > 0:
        impact["before"] = {
            "mean": round(float(orig_series.mean()), 6),
            "std": round(float(orig_series.std(ddof=1)), 6) if len(orig_series) > 1 else None,
            "min": round(float(orig_series.min()), 6),
            "q25": round(float(orig_series.quantile(0.25)), 6),
            "median": round(float(orig_series.median()), 6),
            "q75": round(float(orig_series.quantile(0.75)), 6),
            "max": round(float(orig_series.max()), 6),
            "skewness": round(float(orig_series.skew()), 6) if len(orig_series) > 2 else None,
        }

    if pd.api.types.is_numeric_dtype(trans_series) and len(trans_series) > 0:
        impact["after"] = {
            "mean": round(float(trans_series.mean()), 6),
            "std": round(float(trans_series.std(ddof=1)), 6) if len(trans_series) > 1 else None,
            "min": round(float(trans_series.min()), 6),
            "q25": round(float(trans_series.quantile(0.25)), 6),
            "median": round(float(trans_series.median()), 6),
            "q75": round(float(trans_series.quantile(0.75)), 6),
            "max": round(float(trans_series.max()), 6),
            "skewness": round(float(trans_series.skew()), 6) if len(trans_series) > 2 else None,
        }

    return impact


# ---------------------------------------------------------------------------
# Transformation Candidate Generation
# ---------------------------------------------------------------------------

def suggest_transformations(
    df: pd.DataFrame,
    variable: str,
    reason: str | None = None,
) -> list[TransformationCandidate]:
    """Suggest appropriate transformations for a variable based on its distribution.

    Uses evidence-based status levels:
    - AVAILABLE: transformation can be applied, but no evidence supports it
    - CANDIDATE: some evidence suggests it might help
    - RECOMMENDED: strong evidence supports this transformation

    Does NOT choose based on significance. Only suggests based on domain compatibility
    and distributional evidence.
    """
    if variable not in df.columns:
        return []

    series = df[variable].dropna()
    if len(series) < 10 or not pd.api.types.is_numeric_dtype(series):
        return []

    candidates: list[TransformationCandidate] = []

    # Check skewness
    skewness = float(series.skew()) if len(series) > 2 else 0.0

    # Determine if transformation is actually needed
    abs_skew = abs(skewness)
    # Thresholds: |skew| < 0.5 = approximately symmetric, 0.5-1.0 = moderate, > 1.0 = substantial
    needs_transformation = abs_skew > 0.5

    # Determine skewness direction for accurate description
    if skewness > 0.5:
        skew_direction = "right-skewed"
    elif skewness < -0.5:
        skew_direction = "left-skewed"
    else:
        skew_direction = "approximately symmetric"

    # LOG — only if all positive
    if (series > 0).all():
        safety, violations, _ = validate_transformation_domain(series, TransformationType.LOG)
        # LOG is recommended for right-skewed positive data
        if skewness > 1.0 and safety == TransformationSafety.SAFE:
            status = "recommended"
            log_reason = f"Right-skewed distribution (skewness={skewness:.2f}). Log may reduce skewness."
        elif skewness > 0.5 and safety == TransformationSafety.SAFE:
            status = "candidate"
            log_reason = f"Moderately right-skewed (skewness={skewness:.2f}). Log transformation may help."
        else:
            status = "available"
            log_reason = "No transformation currently indicated. Log available if needed."
        candidates.append(TransformationCandidate(
            transformation_type=TransformationType.LOG,
            variable=variable,
            reason=reason or log_reason,
            safety=safety,
            domain_violations=violations,
            interpretational_consequence="Interprets as multiplicative change. Coefficients represent log-ratios.",
        ))

    # LOG1P — if no values < -1
    if (series >= -1).all():
        safety, violations, _ = validate_transformation_domain(series, TransformationType.LOG1P)
        if skewness > 0.5 and safety == TransformationSafety.SAFE:
            status = "candidate"
            log1p_reason = f"Log1p transformation. Handles zero values. Skewness={skewness:.2f}."
        else:
            status = "available"
            log1p_reason = "No transformation currently indicated. Log1p available if needed."
        candidates.append(TransformationCandidate(
            transformation_type=TransformationType.LOG1P,
            variable=variable,
            reason=reason or log1p_reason,
            safety=safety,
            domain_violations=violations,
            interpretational_consequence="Similar to log but handles zeros. Coefficients represent log-ratios.",
        ))

    # SQRT — if all non-negative
    if (series >= 0).all():
        safety, violations, _ = validate_transformation_domain(series, TransformationType.SQRT)
        if skewness > 0.5 and safety == TransformationSafety.SAFE:
            status = "candidate"
            sqrt_reason = f"Square root transformation. Moderate reduction in skewness. Skewness={skewness:.2f}."
        else:
            status = "available"
            sqrt_reason = "No transformation currently indicated. Square root available if needed."
        candidates.append(TransformationCandidate(
            transformation_type=TransformationType.SQRT,
            variable=variable,
            reason=reason or sqrt_reason,
            safety=safety,
            domain_violations=violations,
            interpretational_consequence="Reduces right skewness less aggressively than log.",
        ))

    # BOX_COX — if all positive
    if (series > 0).all():
        safety, violations, _ = validate_transformation_domain(series, TransformationType.BOX_COX)
        if abs_skew > 1.0 and safety == TransformationSafety.SAFE:
            status = "candidate"
            bc_reason = f"Box-Cox finds optimal power transformation. Substantial skewness ({skewness:.2f})."
        else:
            status = "available"
            bc_reason = "No transformation currently indicated. Box-Cox available if needed."
        candidates.append(TransformationCandidate(
            transformation_type=TransformationType.BOX_COX,
            variable=variable,
            reason=reason or bc_reason,
            safety=safety,
            domain_violations=violations,
            interpretational_consequence="Power-transformed variable. Interpretation depends on fitted lambda.",
        ))

    # YEO_JOHNSON — always available
    safety, violations, _ = validate_transformation_domain(series, TransformationType.YEO_JOHNSON)
    if abs_skew > 1.0 and safety == TransformationSafety.SAFE:
        status = "candidate"
        yj_reason = f"Yeo-Johnson handles negative values. Substantial skewness ({skewness:.2f})."
    else:
        status = "available"
        yj_reason = "No transformation currently indicated. Yeo-Johnson available if needed."
    candidates.append(TransformationCandidate(
        transformation_type=TransformationType.YEO_JOHNSON,
        variable=variable,
        reason=reason or yj_reason,
        safety=safety,
        domain_violations=violations,
        interpretational_consequence="Power-transformed variable. Interpretation depends on fitted lambda.",
    ))

    # STANDARDIZE — always available, recommended if scale differs between variables
    candidates.append(TransformationCandidate(
        transformation_type=TransformationType.STANDARDIZE,
        variable=variable,
        reason=reason or "Standardize to mean=0, sd=1. Available for methods requiring standardized inputs.",
        safety=TransformationSafety.SAFE,
        domain_violations=[],
        interpretational_consequence="Coefficients represent standard deviation changes.",
    ))

    # MIN_MAX — always available
    candidates.append(TransformationCandidate(
        transformation_type=TransformationType.MIN_MAX,
        variable=variable,
        reason=reason or "Min-max scaling to [0,1] range. Available for algorithms requiring bounded inputs.",
        safety=TransformationSafety.SAFE,
        domain_violations=[],
        interpretational_consequence="Coefficients represent proportion of range.",
    ))

    return candidates


# ---------------------------------------------------------------------------
# Imputation (simple deterministic)
# ---------------------------------------------------------------------------

def apply_imputation(
    df: pd.DataFrame,
    variable: str,
    method: str = "median",
    fill_value: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply deterministic imputation to a variable.

    Methods: "mean", "median", "mode", "constant", "forward_fill", "backward_fill".
    Returns (imputed_df, imputation_params).
    """
    if variable not in df.columns:
        raise ValueError(f"Variable '{variable}' not found in dataset.")

    result_df = df.copy()
    n_missing = int(result_df[variable].isna().sum())
    imputation_params: dict[str, Any] = {
        "variable": variable,
        "method": method,
        "n_missing": n_missing,
    }

    if n_missing == 0:
        return result_df, imputation_params

    if method == "mean" and pd.api.types.is_numeric_dtype(result_df[variable]):
        fill_val = float(result_df[variable].mean())
        result_df[variable] = result_df[variable].fillna(fill_val)
        imputation_params["fill_value"] = fill_val

    elif method == "median" and pd.api.types.is_numeric_dtype(result_df[variable]):
        fill_val = float(result_df[variable].median())
        result_df[variable] = result_df[variable].fillna(fill_val)
        imputation_params["fill_value"] = fill_val

    elif method == "mode":
        mode_vals = result_df[variable].mode()
        if len(mode_vals) > 0:
            fill_val = mode_vals.iloc[0]
            result_df[variable] = result_df[variable].fillna(fill_val)
            imputation_params["fill_value"] = str(fill_val)

    elif method == "constant":
        result_df[variable] = result_df[variable].fillna(fill_value)
        imputation_params["fill_value"] = str(fill_value)

    elif method == "forward_fill":
        result_df[variable] = result_df[variable].ffill()

    elif method == "backward_fill":
        result_df[variable] = result_df[variable].bfill()

    else:
        raise ValueError(f"Unsupported imputation method: {method}")

    imputation_params["n_imputed"] = n_missing
    return result_df, imputation_params


# ---------------------------------------------------------------------------
# Exclusion
# ---------------------------------------------------------------------------

def apply_exclusion(
    df: pd.DataFrame,
    indices: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exclude specific observations by index.

    Returns (excluded_df, exclusion_params). Original df not modified.
    """
    result_df = df.copy()
    valid_indices = [i for i in indices if i in result_df.index]
    n_excluded = len(valid_indices)
    result_df = result_df.drop(valid_indices)

    return result_df, {
        "n_excluded": n_excluded,
        "indices": valid_indices,
        "n_remaining": len(result_df),
    }


# ---------------------------------------------------------------------------
# Winsorize
# ---------------------------------------------------------------------------

def apply_winsorize(
    df: pd.DataFrame,
    variable: str,
    lower_pct: float = 0.05,
    upper_pct: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Winsorize a variable to specified percentiles."""
    if variable not in df.columns:
        raise ValueError(f"Variable '{variable}' not found in dataset.")

    result_df = df.copy()
    valid = result_df[variable].dropna()
    lower = float(valid.quantile(lower_pct))
    upper = float(valid.quantile(upper_pct))

    result_df[variable] = result_df[variable].clip(lower, upper)

    return result_df, {
        "variable": variable,
        "method": "winsorize",
        "lower_pct": lower_pct,
        "upper_pct": upper_pct,
        "lower_bound": lower,
        "upper_bound": upper,
        "n_winsorized_lower": int((valid < lower).sum()),
        "n_winsorized_upper": int((valid > upper).sum()),
    }
