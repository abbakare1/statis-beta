"""Missing Data Engine — deterministic missingness analysis.

Provides variable-level, dataset-level, and pattern-level missing data characterization.
Does NOT claim MCAR/MAR/MNAR — classifies observed patterns only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.statistics.data_quality import (
    DatasetMissingness,
    VariableMissingness,
    VariableType,
    MissingnessStatus,
    MissingnessPattern,
    HandlingStrategy,
    HandlingAction,
    MissingDataStrategy,
)


def analyze_missingness(
    df: pd.DataFrame,
    outcome: str | None = None,
    predictors: list[str] | None = None,
) -> DatasetMissingness:
    """Deterministic missingness analysis for a DataFrame.

    Returns a DatasetMissingness with variable-level and dataset-level summaries.
    Does NOT infer mechanism — only characterizes observed patterns.
    """
    predictors = predictors or []
    total_rows = len(df)
    total_cells = total_rows * len(df.columns)

    # Variable-level
    variables: list[VariableMissingness] = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        n_observed = total_rows - n_missing
        pct = round(n_missing / total_rows * 100, 2) if total_rows > 0 else 0.0

        status = _classify_missingness_status(pct, n_missing, total_rows)
        var_type = _infer_variable_type(df[col])

        is_outcome = col == outcome
        is_predictor = col in predictors

        recommendation = _recommend_strategy(
            status, var_type, is_outcome, is_predictor, n_missing, total_rows,
        )

        variables.append(VariableMissingness(
            variable=col,
            variable_type=var_type,
            n_missing=n_missing,
            n_observed=n_observed,
            total=total_rows,
            pct_missing=pct,
            status=status,
            is_outcome=is_outcome,
            is_predictor=is_predictor,
            recommendation=recommendation,
        ))

    # Dataset-level
    complete_mask = ~df.isna().any(axis=1)
    complete_cases = int(complete_mask.sum())
    incomplete_cases = total_rows - complete_cases
    total_missing_cells = int(df.isna().sum().sum())
    pct_missing_overall = round(total_missing_cells / total_cells * 100, 2) if total_cells > 0 else 0.0

    # Pattern detection
    patterns = _detect_missingness_patterns(df)
    pattern_class, pattern_details = _classify_pattern(df, patterns)

    return DatasetMissingness(
        total_rows=total_rows,
        complete_cases=complete_cases,
        incomplete_cases=incomplete_cases,
        total_missing_cells=total_missing_cells,
        total_cells=total_cells,
        pct_missing_overall=pct_missing_overall,
        variables=variables,
        pattern_classification=pattern_class,
        pattern_details=pattern_details,
        missingness_patterns=patterns,
    )


def generate_missing_data_strategies(
    missingness: DatasetMissingness,
    outcome: str | None = None,
    predictors: list[str] | None = None,
) -> list[HandlingStrategy]:
    """Generate candidate handling strategies based on missingness profile.

    Does NOT automatically impute. Produces explicit candidates for researcher review.
    """
    predictors = predictors or []
    strategies: list[HandlingStrategy] = []
    strategy_id_counter = 0

    for var in missingness.variables:
        if var.status == MissingnessStatus.NONE:
            continue

        # Strategy 1: Complete-case analysis (always consider)
        if var.status in (MissingnessStatus.LOW, MissingnessStatus.MODERATE):
            strategy_id_counter += 1
            strategies.append(HandlingStrategy(
                strategy_id=f"mca_{strategy_id_counter}",
                action=HandlingAction.NO_ACTION,
                target_variable=var.variable,
                details=(
                    f"Complete-case analysis: exclude {var.n_missing} observations "
                    f"with missing '{var.variable}' ({var.pct_missing}% missing). "
                    f"Impact: {missingness.complete_cases} complete cases available."
                ),
                impact_summary={
                    "n_excluded": var.n_missing,
                    "n_remaining": missingness.total_rows - var.n_missing,
                    "pct_excluded": round(var.n_missing / missingness.total_rows * 100, 2) if missingness.total_rows > 0 else 0,
                },
                requires_approval=True,
            ))

        # Strategy 2: Simple imputation (mean/median/mode)
        if var.status in (MissingnessStatus.LOW, MissingnessStatus.MODERATE) and not var.is_outcome:
            strategy_id_counter += 1
            impute_method = "median" if var.variable_type == "numeric" else "mode"
            strategies.append(HandlingStrategy(
                strategy_id=f"si_{strategy_id_counter}",
                action=HandlingAction.IMPUTE,
                target_variable=var.variable,
                details=(
                    f"Simple imputation ({impute_method}) for '{var.variable}': "
                    f"impute {var.n_missing} missing values using {impute_method} "
                    f"of observed values. Caution: reduces variance."
                ),
                impact_summary={
                    "n_imputed": var.n_missing,
                    "method": impute_method,
                },
                requires_approval=True,
            ))

        # Strategy 3: Missing indicator
        if var.status in (MissingnessStatus.MODERATE, MissingnessStatus.HIGH):
            strategy_id_counter += 1
            strategies.append(HandlingStrategy(
                strategy_id=f"mi_{strategy_id_counter}",
                action=HandlingAction.IMPUTE,
                target_variable=var.variable,
                details=(
                    f"Missing indicator: create binary variable '{var.variable}_missing' "
                    f"flagging {var.n_missing} missing observations. "
                    f"Keep original variable with missing values."
                ),
                impact_summary={
                    "n_indicator": var.n_missing,
                    "new_variable": f"{var.variable}_missing",
                },
                requires_approval=True,
            ))

        # Strategy 4: Variable exclusion
        if var.status in (MissingnessStatus.HIGH, MissingnessStatus.EXTREME):
            strategy_id_counter += 1
            strategies.append(HandlingStrategy(
                strategy_id=f"ev_{strategy_id_counter}",
                action=HandlingAction.EXCLUDE_VARIABLE,
                target_variable=var.variable,
                details=(
                    f"Exclude variable '{var.variable}': {var.pct_missing}% missing. "
                    f"Consider dropping from analysis."
                ),
                impact_summary={
                    "n_variables_removed": 1,
                    "variables": [var.variable],
                },
                requires_approval=True,
            ))

        # Strategy 5: No action (explicit)
        strategy_id_counter += 1
        strategies.append(HandlingStrategy(
            strategy_id=f"na_{strategy_id_counter}",
            action=HandlingAction.NO_ACTION,
            target_variable=var.variable,
            details=(
                f"No action on '{var.variable}': retain {var.n_missing} missing values. "
                f"Analysis must handle missingness appropriately."
            ),
            impact_summary={
                "n_missing": var.n_missing,
            },
            requires_approval=False,
        ))

    # Outcome missing — flag specially
    if outcome:
        for var in missingness.variables:
            if var.variable == outcome and var.status != MissingnessStatus.NONE:
                strategy_id_counter += 1
                strategies.append(HandlingStrategy(
                    strategy_id=f"om_{strategy_id_counter}",
                    action=HandlingAction.RETAIN,
                    target_variable=outcome,
                    details=(
                        f"MISSING OUTCOME '{outcome}': {var.n_missing} observations "
                        f"({var.pct_missing}%) have missing outcome values. "
                        f"Cannot impute outcome without methodological justification. "
                        f"Flag for researcher review."
                    ),
                    impact_summary={
                        "n_missing_outcome": var.n_missing,
                        "severity": "critical",
                    },
                    requires_approval=True,
                ))

    return strategies


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_missingness_status(pct: float, n_missing: int, total: int) -> MissingnessStatus:
    """Classify missingness into a severity status."""
    if n_missing == 0:
        return MissingnessStatus.NONE
    if n_missing == total:
        return MissingnessStatus.ALL_MISSING
    if pct <= 5:
        return MissingnessStatus.LOW
    if pct <= 20:
        return MissingnessStatus.MODERATE
    if pct <= 50:
        return MissingnessStatus.HIGH
    return MissingnessStatus.EXTREME


def _infer_variable_type(series: pd.Series) -> str:
    """Infer variable type from a pandas Series."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return VariableType.DATETIME
    if pd.api.types.is_bool_dtype(series):
        return VariableType.BOOLEAN
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        if n_unique <= 2:
            return VariableType.BINARY
        return VariableType.NUMERIC
    if series.dtype == object:
        n_unique = series.nunique()
        if n_unique <= 20:
            return VariableType.CATEGORICAL
        return VariableType.TEXT
    return VariableType.TEXT


def _recommend_strategy(
    status: MissingnessStatus,
    var_type: str,
    is_outcome: bool,
    is_predictor: bool,
    n_missing: int,
    total: int,
) -> str:
    """Generate a recommendation string for a variable's missingness."""
    if status == MissingnessStatus.NONE:
        return "No missing data."

    if is_outcome and status != MissingnessStatus.NONE:
        return (
            f"Missing outcome data ({status.value}). "
            f"Cannot impute without methodological justification. "
            f"Flag for researcher review."
        )

    if status in (MissingnessStatus.EXTREME, MissingnessStatus.ALL_MISSING):
        return (
            f"Extreme missingness ({status.value}). "
            f"Consider excluding this variable or using domain-specific methods."
        )

    if status == MissingnessStatus.HIGH:
        return (
            f"High missingness ({status.value}). "
            f"Evaluate missingness mechanism and consider multiple imputation."
        )

    if status == MissingnessStatus.MODERATE:
        if var_type == "numeric":
            return (
                f"Moderate missingness ({status.value}). "
                f"Mean/median imputation or multiple imputation may be appropriate."
            )
        return (
            f"Moderate missingness ({status.value}). "
            f"Mode imputation or multiple imputation may be appropriate."
        )

    if status == MissingnessStatus.LOW:
        return (
            f"Low missingness ({status.value}). "
            f"Complete-case analysis may be acceptable depending on mechanism."
        )

    return "Review missingness pattern."


def _detect_missingness_patterns(df: pd.DataFrame) -> list[dict]:
    """Identify common missingness patterns in the dataset.

    Returns a list of patterns, each describing which variables are missing together.
    """
    patterns = []
    df_missing = df.isna()

    # Only consider rows with some missing data
    has_missing = df_missing.any(axis=1)
    if not has_missing.any():
        return patterns

    df_with_missing = df_missing[has_missing]

    # Group by missingness pattern
    pattern_groups = df_with_missing.groupby(list(df_with_missing.columns))
    for pattern_key, group in pattern_groups:
        if isinstance(pattern_key, tuple):
            missing_vars = [df.columns[i] for i, is_missing in enumerate(pattern_key) if is_missing]
        else:
            # Single column case
            missing_vars = [df.columns[0]] if pattern_key else []

        if missing_vars:
            patterns.append({
                "pattern": missing_vars,
                "n_observations": len(group),
                "pct_affected": round(len(group) / len(df) * 100, 2) if len(df) > 0 else 0,
            })

    # Sort by frequency
    patterns.sort(key=lambda x: x["n_observations"], reverse=True)

    return patterns[:20]  # Top 20 patterns


def _classify_pattern(
    df: pd.DataFrame,
    patterns: list[dict],
) -> tuple[MissingnessPattern, str | None]:
    """Classify the observed missingness pattern.

    Does NOT claim mechanism (MCAR/MAR/MNAR). Only flags concerns.
    """
    if not patterns:
        return MissingnessPattern.OBSERVED_PATTERN, "No missing data patterns detected."

    total_rows = len(df)
    top_pattern = patterns[0]
    top_pct = top_pattern["pct_affected"]

    # Single variable dominates missingness
    if len(patterns) == 1 and len(top_pattern["pattern"]) == 1:
        var = top_pattern["pattern"][0]
        if top_pct > 30:
            return (
                MissingnessPattern.POTENTIAL_CONCERN,
                f"Single variable '{var}' dominates missingness ({top_pct}% of rows). "
                f"Mechanism should be assessed by the researcher.",
            )
        return (
            MissingnessPattern.OBSERVED_PATTERN,
            f"Missingness concentrated in variable '{var}' ({top_pct}% of rows).",
        )

    # Multiple variables missing together
    if len(top_pattern["pattern"]) > 1:
        return (
            MissingnessPattern.REQUIRES_RESEARCHER_ASSESSMENT,
            f"Multiple variables missing together in {top_pct}% of rows: "
            f"{', '.join(top_pattern['pattern'])}. "
            f"Pattern suggests systematic missingness — researcher assessment needed.",
        )

    # Many different patterns
    if len(patterns) > 5:
        return (
            MissingnessPattern.POTENTIAL_CONCERN,
            f"{len(patterns)} distinct missingness patterns detected. "
            f"Missingness may not be simple — researcher assessment recommended.",
        )

    return MissingnessPattern.OBSERVED_PATTERN, None
