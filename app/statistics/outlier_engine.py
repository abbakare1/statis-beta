"""Outlier Engine — deterministic outlier detection and characterization.

Provides IQR, Z-score, and MAD methods. Does NOT automatically remove outliers.
Classifies each outlier conceptually and generates candidate handling strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.statistics.data_quality import (
    VariableOutliers,
    OutlierFlag,
    OutlierMethod,
    OutlierInterpretation,
    OutlierHandlingStrategy,
    HandlingStrategy,
    HandlingAction,
)


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    k: float = 1.5,
) -> list[VariableOutliers]:
    """Detect outliers using the IQR method.

    For each numeric column, flags observations beyond Q1 - k*IQR or Q3 + k*IQR.
    """
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    results = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) < 10:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower_fence = q1 - k * iqr
        upper_fence = q3 + k * iqr

        mask = (series < lower_fence) | (series > upper_fence)
        flagged_indices = series[mask].index.tolist()
        flagged_values = series[mask].values.tolist()

        flags = []
        for idx, val in zip(flagged_indices, flagged_values):
            interpretation = _interpret_outlier(float(val), q1, q3, iqr, series)
            strategies = _suggest_handling(interpretation, "iqr")
            flags.append(OutlierFlag(
                observation_index=int(idx),
                variable=col,
                value=float(val),
                method="iqr",
                threshold=k,
                interpretation=interpretation,
                handling_strategies=strategies,
            ))

        n_flagged = len(flags)
        pct = round(n_flagged / len(series) * 100, 2) if len(series) > 0 else 0.0

        overall_interpretation = _aggregate_interpretation(flags)
        overall_strategies = _aggregate_strategies(flags)

        results.append(VariableOutliers(
            variable=col,
            method="iqr",
            threshold=k,
            n_flagged=n_flagged,
            pct_flagged=pct,
            min_value=float(series.min()),
            max_value=float(series.max()),
            fence_lower=lower_fence,
            fence_upper=upper_fence,
            flags=flags,
            interpretation=overall_interpretation,
            handling_strategies=overall_strategies,
        ))

    return results


def detect_outliers_zscore(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 3.0,
) -> list[VariableOutliers]:
    """Detect outliers using Z-score method.

    Flags observations with |z| >= threshold.
    """
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    results = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) < 10:
            continue

        mean = series.mean()
        std = series.std(ddof=1)
        if std == 0:
            continue

        z_scores = (series - mean) / std
        mask = z_scores.abs() >= threshold
        flagged_indices = series[mask].index.tolist()
        flagged_values = series[mask].values.tolist()

        flags = []
        for idx, val, z in zip(flagged_indices, flagged_values, z_scores[mask]):
            interpretation = _interpret_outlier_zscore(float(z), float(val), mean, std)
            strategies = _suggest_handling(interpretation, "zscore")
            flags.append(OutlierFlag(
                observation_index=int(idx),
                variable=col,
                value=float(val),
                method="zscore",
                threshold=threshold,
                interpretation=interpretation,
                handling_strategies=strategies,
            ))

        n_flagged = len(flags)
        pct = round(n_flagged / len(series) * 100, 2) if len(series) > 0 else 0.0

        overall_interpretation = _aggregate_interpretation(flags)
        overall_strategies = _aggregate_strategies(flags)

        results.append(VariableOutliers(
            variable=col,
            method="zscore",
            threshold=threshold,
            n_flagged=n_flagged,
            pct_flagged=pct,
            min_value=float(series.min()),
            max_value=float(series.max()),
            flags=flags,
            interpretation=overall_interpretation,
            handling_strategies=overall_strategies,
        ))

    return results


def detect_outliers_mad(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 3.0,
) -> list[VariableOutliers]:
    """Detect outliers using Median Absolute Deviation (MAD).

    MAD = median(|Xi - median(X)|)
    Modified Z = 0.6745 * (Xi - median) / MAD
    """
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    results = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) < 10:
            continue

        median = float(series.median())
        mad = float((series - median).abs().median())
        if mad == 0:
            continue

        modified_z = 0.6745 * (series - median) / mad
        mask = modified_z.abs() >= threshold
        flagged_indices = series[mask].index.tolist()
        flagged_values = series[mask].values.tolist()

        flags = []
        for idx, val, z in zip(flagged_indices, flagged_values, modified_z[mask]):
            interpretation = _interpret_outlier_zscore(float(z), float(val), median, mad)
            strategies = _suggest_handling(interpretation, "mad")
            flags.append(OutlierFlag(
                observation_index=int(idx),
                variable=col,
                value=float(val),
                method="mad",
                threshold=threshold,
                interpretation=interpretation,
                handling_strategies=strategies,
            ))

        n_flagged = len(flags)
        pct = round(n_flagged / len(series) * 100, 2) if len(series) > 0 else 0.0

        overall_interpretation = _aggregate_interpretation(flags)
        overall_strategies = _aggregate_strategies(flags)

        results.append(VariableOutliers(
            variable=col,
            method="mad",
            threshold=threshold,
            n_flagged=n_flagged,
            pct_flagged=pct,
            min_value=float(series.min()),
            max_value=float(series.max()),
            flags=flags,
            interpretation=overall_interpretation,
            handling_strategies=overall_strategies,
        ))

    return results


def detect_outliers(
    df: pd.DataFrame,
    methods: list[str] | None = None,
    columns: list[str] | None = None,
) -> list[VariableOutliers]:
    """Run outlier detection using specified methods.

    Returns combined results. If multiple methods flag the same observation,
    it appears in both results (caller can deduplicate).
    """
    if methods is None:
        methods = ["iqr"]

    all_results = []
    for method in methods:
        if method == "iqr":
            all_results.extend(detect_outliers_iqr(df, columns))
        elif method == "zscore":
            all_results.extend(detect_outliers_zscore(df, columns))
        elif method == "mad":
            all_results.extend(detect_outliers_mad(df, columns))

    return all_results


def generate_outlier_strategies(
    outlier_results: list[VariableOutliers],
) -> list[HandlingStrategy]:
    """Generate handling strategies for detected outliers.

    RETAIN is always the default unless there is evidence for another action.
    """
    strategies = []
    strategy_id = 0

    for result in outlier_results:
        if result.n_flagged == 0:
            continue

        # RETAIN — always offered
        strategy_id += 1
        strategies.append(HandlingStrategy(
            strategy_id=f"or_{strategy_id}",
            action=HandlingAction.RETAIN,
            target_variable=result.variable,
            details=(
                f"Retain all {result.n_flagged} flagged observations in '{result.variable}'. "
                f"Values: {result.flags[0].value:.2f} to {result.flags[-1].value:.2f} "
                f"(method: {result.method}, threshold: {result.threshold}). "
                f"Flagged observations are {result.pct_flagged}% of the variable."
            ),
            impact_summary={
                "n_flagged": result.n_flagged,
                "pct_flagged": result.pct_flagged,
                "method": result.method,
            },
            requires_approval=False,
        ))

        # EXCLUDE — only if there is a plausible reason
        if result.interpretation in (
            OutlierInterpretation.POSSIBLE_DATA_ERROR,
            OutlierInterpretation.POSSIBLE_INFLUENTIAL_OBSERVATION,
        ):
            strategy_id += 1
            strategies.append(HandlingStrategy(
                strategy_id=f"oe_{strategy_id}",
                action=HandlingAction.EXCLUDE_OBSERVATIONS,
                target_variable=result.variable,
                target_observations=[f.observation_index for f in result.flags],
                details=(
                    f"Exclude {result.n_flagged} observations from '{result.variable}' "
                    f"flagged as {result.interpretation.value}. "
                    f"Requires researcher confirmation that values are errors or "
                    f"influential. Impact: n reduces by {result.n_flagged}."
                ),
                impact_summary={
                    "n_excluded": result.n_flagged,
                    "indices": [f.observation_index for f in result.flags],
                },
                requires_approval=True,
            ))

        # WINSORIZE
        if result.interpretation in (
            OutlierInterpretation.POSSIBLE_TRUE_EXTREME,
            OutlierInterpretation.POSSIBLE_INFLUENTIAL_OBSERVATION,
        ):
            strategy_id += 1
            strategies.append(HandlingStrategy(
                strategy_id=f"ow_{strategy_id}",
                action=HandlingAction.WINSORIZE,
                target_variable=result.variable,
                details=(
                    f"Winsorize {result.n_flagged} flagged observations in '{result.variable}' "
                    f"to the nearest non-outlier value. Preserves sample size but "
                    f"reduces extreme influence."
                ),
                impact_summary={
                    "n_winsorized": result.n_flagged,
                    "method": result.method,
                },
                requires_approval=True,
            ))

        # ROBUST METHOD
        strategy_id += 1
        strategies.append(HandlingStrategy(
            strategy_id=f"orm_{strategy_id}",
            action=HandlingAction.RETAIN,
            target_variable=result.variable,
            details=(
                f"Use robust statistical methods for '{result.variable}' "
                f"(e.g., Spearman instead of Pearson, Mann-Whitney instead of t-test). "
                f"Retains all observations while reducing outlier influence."
            ),
            impact_summary={
                "method": "robust_alternative",
                "n_affected": result.n_flagged,
            },
            requires_approval=False,
        ))

    return strategies


def compute_sensitivity_variants(
    outlier_results: list[VariableOutliers],
) -> list[dict]:
    """Compute sensitivity analysis variants based on outlier detection.

    Returns primary (retain all) and sensitivity variants for researcher comparison.
    """
    variants = [
        {
            "variant_label": "primary",
            "description": "Retain all observations (default analysis)",
            "handling_changes": {},
        }
    ]

    for result in outlier_results:
        if result.n_flagged == 0:
            continue

        # Sensitivity: exclude flagged observations
        variants.append({
            "variant_label": f"exclude_{result.variable}",
            "description": (
                f"Exclude {result.n_flagged} outlier observations in '{result.variable}' "
                f"({result.method} method, threshold: {result.threshold})"
            ),
            "handling_changes": {
                "exclude_observations": {
                    result.variable: [f.observation_index for f in result.flags],
                },
            },
        })

    return variants


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _interpret_outlier(
    value: float, q1: float, q3: float, iqr: float, series: pd.Series,
) -> OutlierInterpretation:
    """Interpret an IQR-flagged outlier.

    Does NOT claim it is a data error merely because it is extreme.
    """
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    extreme_lower = q1 - 3 * iqr
    extreme_upper = q3 + 3 * iqr

    # Very extreme (beyond 3*IQR) — more likely data error but still not certain
    if value < extreme_lower or value > extreme_upper:
        return OutlierInterpretation.POSSIBLE_DATA_ERROR

    # Moderately extreme — could be true extreme or influential
    if value < lower or value > upper:
        return OutlierInterpretation.POSSIBLE_TRUE_EXTREME

    return OutlierInterpretation.UNKNOWN


def _interpret_outlier_zscore(
    z: float, value: float, center: float, spread: float,
) -> OutlierInterpretation:
    """Interpret a Z-score flagged outlier."""
    abs_z = abs(z)

    if abs_z >= 5:
        return OutlierInterpretation.POSSIBLE_DATA_ERROR
    if abs_z >= 4:
        return OutlierInterpretation.POSSIBLE_INFLUENTIAL_OBSERVATION
    return OutlierInterpretation.POSSIBLE_TRUE_EXTREME


def _suggest_handling(
    interpretation: OutlierInterpretation, method: str,
) -> list[OutlierHandlingStrategy]:
    """Suggest handling strategies based on interpretation."""
    if interpretation == OutlierInterpretation.POSSIBLE_DATA_ERROR:
        return [
            OutlierHandlingStrategy.RETAIN,
            OutlierHandlingStrategy.EXCLUDE,
            OutlierHandlingStrategy.SENSITIVITY_ANALYSIS,
        ]
    if interpretation == OutlierInterpretation.POSSIBLE_TRUE_EXTREME:
        return [
            OutlierHandlingStrategy.RETAIN,
            OutlierHandlingStrategy.ROBUST_METHOD,
            OutlierHandlingStrategy.TRANSFORM,
            OutlierHandlingStrategy.SENSITIVITY_ANALYSIS,
        ]
    if interpretation == OutlierInterpretation.POSSIBLE_INFLUENTIAL_OBSERVATION:
        return [
            OutlierHandlingStrategy.RETAIN,
            OutlierHandlingStrategy.WINSORIZE,
            OutlierHandlingStrategy.ROBUST_METHOD,
            OutlierHandlingStrategy.SENSITIVITY_ANALYSIS,
        ]
    return [OutlierHandlingStrategy.RETAIN]


def _aggregate_interpretation(flags: list[OutlierFlag]) -> OutlierInterpretation:
    """Aggregate interpretations across multiple flagged outliers."""
    if not flags:
        return OutlierInterpretation.UNKNOWN

    # If any are POSSIBLE_DATA_ERROR, that dominates
    if any(f.interpretation == OutlierInterpretation.POSSIBLE_DATA_ERROR for f in flags):
        return OutlierInterpretation.POSSIBLE_DATA_ERROR
    if any(f.interpretation == OutlierInterpretation.POSSIBLE_INFLUENTIAL_OBSERVATION for f in flags):
        return OutlierInterpretation.POSSIBLE_INFLUENTIAL_OBSERVATION
    if any(f.interpretation == OutlierInterpretation.POSSIBLE_TRUE_EXTREME for f in flags):
        return OutlierInterpretation.POSSIBLE_TRUE_EXTREME
    return OutlierInterpretation.UNKNOWN


def _aggregate_strategies(flags: list[OutlierFlag]) -> list[OutlierHandlingStrategy]:
    """Aggregate unique strategies across all flags."""
    seen = set()
    result = []
    for f in flags:
        for s in f.handling_strategies:
            if s not in seen:
                seen.add(s)
                result.append(s)
    return result
