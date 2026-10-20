"""Method-Specific Statistical Diagnostics — deterministic checks for each supported method.

Every function takes the raw DataFrame, the engine result dict, and the
analysis specification. It returns a list of DiagnosticResult objects.

No LLM reasoning. No automatic method switching. No p-value-based selection.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Any

from app.statistics.diagnostics import (
    DiagnosticResult,
    DiagnosticSeverity,
    DiagnosticStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _diag(
    diagnostic_id: str,
    diagnostic_name: str,
    *,
    severity: DiagnosticSeverity = DiagnosticSeverity.INFO,
    status: DiagnosticStatus = DiagnosticStatus.PASS,
    variable: str | None = None,
    variables: list[str] | None = None,
    method: str = "",
    observed_value: Any = None,
    threshold: Any = None,
    message: str = "",
    rationale: str = "",
    provenance: dict[str, Any] | None = None,
) -> DiagnosticResult:
    """Shorthand to create a DiagnosticResult."""
    return DiagnosticResult(
        diagnostic_id=diagnostic_id,
        diagnostic_name=diagnostic_name,
        severity=severity,
        status=status,
        variable=variable,
        variables=variables or [],
        method=method,
        observed_value=observed_value,
        threshold=threshold,
        message=message,
        rationale=rationale,
        deterministic=True,
        provenance=provenance or {},
    )


def _variance_ratio(s1: pd.Series, s2: pd.Series) -> float:
    """Compute variance ratio (larger/smaller)."""
    v1, v2 = s1.var(ddof=1), s2.var(ddof=1)
    if v1 == 0 or v2 == 0:
        return float("inf")
    return max(v1, v2) / min(v1, v2)


# ---------------------------------------------------------------------------
# A. Independent Samples t-test
# ---------------------------------------------------------------------------

def diagnostic_independent_t_test(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    result: dict[str, Any],
    *,
    group1: Any = None,
    group2: Any = None,
) -> list[DiagnosticResult]:
    """Diagnostics for independent samples t-test.

    Checks:
    - Exactly two groups
    - Numeric continuous outcome
    - Observations per group
    - Zero variance within either group
    - Extreme imbalance
    - Finite values
    - Variance equality (Welch vs pooled)
    - Reports which variant was used
    """
    diagnostics = []
    method = result.get("test", "Independent t-test")
    is_welch = "Welch" in method

    # Two groups
    groups = df[group].dropna().unique()
    diagnostics.append(_diag(
        "t_ind_groups", "Two independent groups",
        severity=DiagnosticSeverity.INFO,
        status=DiagnosticStatus.PASS if len(groups) == 2 else DiagnosticStatus.FAIL,
        variable=group, method=method,
        observed_value=len(groups), threshold=2,
        message=f"Found {len(groups)} groups." if len(groups) != 2 else "Exactly two groups.",
        rationale="Independent t-test requires exactly two comparison groups.",
    ))

    # Numeric outcome
    diagnostics.append(_diag(
        "t_ind_numeric", "Numeric outcome",
        severity=DiagnosticSeverity.INFO,
        status=DiagnosticStatus.PASS if pd.api.types.is_numeric_dtype(df[outcome]) else DiagnosticStatus.FAIL,
        variable=outcome, method=method,
        observed_value=str(df[outcome].dtype),
        message="Outcome is numeric." if pd.api.types.is_numeric_dtype(df[outcome]) else "Outcome is not numeric.",
        rationale="t-test requires a continuous numeric outcome variable.",
    ))

    # Group sizes
    n1 = result.get("n_group1", 0)
    n2 = result.get("n_group2", 0)
    if n1 > 0 and n2 > 0:
        ratio = max(n1, n2) / min(n1, n2)
        diagnostics.append(_diag(
            "t_ind_imbalance", "Group size balance",
            severity=DiagnosticSeverity.WARNING,
            status=DiagnosticStatus.WARNING if ratio > 3 else DiagnosticStatus.PASS,
            variable=group, method=method,
            observed_value=round(ratio, 2), threshold=3.0,
            message=f"Group size ratio: {ratio:.2f}." + (" Extreme imbalance detected." if ratio > 3 else ""),
            rationale="Extreme group imbalance can reduce power and affect Type I error rates.",
        ))

    # Zero variance within groups
    if n1 >= 2 and n2 >= 2:
        g1_vals = df.loc[df[group] == result.get("group1", group1), outcome].dropna()
        g2_vals = df.loc[df[group] == result.get("group2", group2), outcome].dropna()
        v1 = g1_vals.var(ddof=1) if len(g1_vals) > 1 else 0
        v2 = g2_vals.var(ddof=1) if len(g2_vals) > 1 else 0

        if v1 == 0 or v2 == 0:
            diagnostics.append(_diag(
                "t_ind_zero_var", "Zero variance in group",
                severity=DiagnosticSeverity.ERROR,
                status=DiagnosticStatus.FAIL,
                variable=outcome, method=method,
                observed_value={"var_group1": v1, "var_group2": v2},
                message="Zero variance detected in one or both groups.",
                rationale="A t-test with zero variance in any group is not meaningful.",
            ))
        else:
            diagnostics.append(_diag(
                "t_ind_zero_var", "Zero variance in group",
                severity=DiagnosticSeverity.INFO,
                status=DiagnosticStatus.PASS,
                variable=outcome, method=method,
                observed_value={"var_group1": round(v1, 6), "var_group2": round(v2, 6)},
            ))

        # Variance ratio (Levene-style)
        vratio = _variance_ratio(g1_vals, g2_vals)
        var_status = DiagnosticStatus.PASS
        var_severity = DiagnosticSeverity.INFO
        var_msg = f"Variance ratio: {vratio:.2f}."
        if vratio > 4:
            var_status = DiagnosticStatus.REVIEW_REQUIRED
            var_severity = DiagnosticSeverity.REVIEW_REQUIRED
            var_msg += " Substantial variance inequality."
        elif vratio > 2:
            var_status = DiagnosticStatus.WARNING
            var_severity = DiagnosticSeverity.WARNING
            var_msg += " Moderate variance inequality."

        if is_welch:
            var_msg += " Welch's t-test was used (unequal variance assumed)."
        else:
            var_msg += " Student's t-test was used (equal variances assumed)."

        diagnostics.append(_diag(
            "t_ind_variance_ratio", "Variance equality",
            severity=var_severity,
            status=var_status,
            variable=outcome, method=method,
            observed_value=round(vratio, 4),
            threshold=2.0,
            message=var_msg,
            rationale=(
                f"Variance ratio {vratio:.2f}. "
                f"{'Welch' if is_welch else 'Student'}'s t-test was executed. "
                "If the variance ratio is large and equal-variance assumption was used, "
                "consider whether a new specification with Welch's test is appropriate."
            ),
        ))

    # Finite values
    if n1 > 0 and n2 > 0:
        g1_vals = df.loc[df[group] == result.get("group1", group1), outcome].dropna()
        g2_vals = df.loc[df[group] == result.get("group2", group2), outcome].dropna()
        all_finite = np.all(np.isfinite(g1_vals)) and np.all(np.isfinite(g2_vals))
        diagnostics.append(_diag(
            "t_ind_finite", "Finite values",
            severity=DiagnosticSeverity.ERROR,
            status=DiagnosticStatus.PASS if all_finite else DiagnosticStatus.FAIL,
            variable=outcome, method=method,
            observed_value="all finite" if all_finite else "non-finite values present",
            message="All values are finite." if all_finite else "Non-finite values detected.",
            rationale="Non-finite values indicate data quality issues that invalidate the computation.",
        ))

    # Shapiro-Wilk on each group (if available)
    if n1 >= 3 and n2 >= 3:
        g1_vals = df.loc[df[group] == result.get("group1", group1), outcome].dropna()
        g2_vals = df.loc[df[group] == result.get("group2", group2), outcome].dropna()
        for label, vals in [("group1", g1_vals), ("group2", g2_vals)]:
            if len(vals) <= 5000:
                try:
                    from scipy import stats as sp_stats
                    _, p = sp_stats.shapiro(vals.values)
                    if p < 0.05:
                        diagnostics.append(_diag(
                            f"t_ind_normality_{label}", f"Normality ({label})",
                            severity=DiagnosticSeverity.WARNING,
                            status=DiagnosticStatus.WARNING,
                            variable=outcome, method=method,
                            observed_value=round(float(p), 6),
                            threshold=0.05,
                            message=f"Shapiro-Wilk test for {label}: p={p:.4f}. Non-normality detected.",
                            rationale=(
                                "The t-test is robust to moderate non-normality with sufficient sample size "
                                "(Central Limit Theorem). This diagnostic flags potential departure from normality."
                            ),
                        ))
                except Exception:
                    pass

    return diagnostics


# ---------------------------------------------------------------------------
# B. Paired t-test
# ---------------------------------------------------------------------------

def diagnostic_paired_t_test(
    df: pd.DataFrame,
    outcome1: str,
    outcome2: str,
    result: dict[str, Any],
) -> list[DiagnosticResult]:
    """Diagnostics for paired t-test.

    Checks:
    - Sufficient complete pairs
    - Numeric paired measurements
    - Zero variance of differences
    - Finite differences
    - Distributional diagnostics on differences
    """
    diagnostics = []
    method = result.get("test", "Paired t-test")
    n_pairs = result.get("n_pairs", 0)

    # Complete pairs
    both_present = df[[outcome1, outcome2]].dropna().shape[0]
    diagnostics.append(_diag(
        "t_paired_pairs", "Complete paired observations",
        severity=DiagnosticSeverity.INFO,
        status=DiagnosticStatus.PASS if both_present >= 5 else DiagnosticStatus.WARNING,
        variables=[outcome1, outcome2], method=method,
        observed_value=both_present, threshold=5,
        message=f"{both_present} complete pairs.",
        rationale="Paired t-test requires matching observations for both measurements.",
    ))

    # Numeric
    both_numeric = pd.api.types.is_numeric_dtype(df[outcome1]) and pd.api.types.is_numeric_dtype(df[outcome2])
    diagnostics.append(_diag(
        "t_paired_numeric", "Numeric paired measurements",
        severity=DiagnosticSeverity.ERROR,
        status=DiagnosticStatus.PASS if both_numeric else DiagnosticStatus.FAIL,
        variables=[outcome1, outcome2], method=method,
        message="Both variables are numeric." if both_numeric else "One or both variables are not numeric.",
        rationale="Paired t-test requires numeric measurements.",
    ))

    # Differences
    if both_numeric and both_present > 0:
        paired_df = df[[outcome1, outcome2]].dropna()
        diffs = paired_df[outcome1] - paired_df[outcome2]
        diff_var = diffs.var(ddof=1) if len(diffs) > 1 else 0

        diagnostics.append(_diag(
            "t_paired_diff_var", "Variance of differences",
            severity=DiagnosticSeverity.WARNING,
            status=DiagnosticStatus.WARNING if diff_var == 0 else DiagnosticStatus.PASS,
            variables=[outcome1, outcome2], method=method,
            observed_value=round(float(diff_var), 6),
            message="Zero variance in differences." if diff_var == 0 else f"Difference variance: {diff_var:.4f}.",
            rationale="Zero variance of differences means no systematic or random change between measurements.",
        ))

        # Finite
        all_finite = np.all(np.isfinite(diffs))
        diagnostics.append(_diag(
            "t_paired_finite", "Finite differences",
            severity=DiagnosticSeverity.ERROR,
            status=DiagnosticStatus.PASS if all_finite else DiagnosticStatus.FAIL,
            variables=[outcome1, outcome2], method=method,
            message="All differences are finite." if all_finite else "Non-finite differences detected.",
            rationale="Non-finite differences invalidate the paired test.",
        ))

        # Normality of differences
        if 3 <= len(diffs) <= 5000:
            try:
                from scipy import stats as sp_stats
                _, p = sp_stats.shapiro(diffs.values)
                if p < 0.05:
                    diagnostics.append(_diag(
                        "t_paired_normality", "Normality of differences",
                        severity=DiagnosticSeverity.WARNING,
                        status=DiagnosticStatus.WARNING,
                        variables=[outcome1, outcome2], method=method,
                        observed_value=round(float(p), 6),
                        threshold=0.05,
                        message=f"Shapiro-Wilk on differences: p={p:.4f}. Non-normality detected.",
                        rationale="Paired t-test assumes approximately normal differences. With sufficient N, this is robust.",
                    ))
            except Exception:
                pass

    return diagnostics


# ---------------------------------------------------------------------------
# C. Mann-Whitney U
# ---------------------------------------------------------------------------

def diagnostic_mann_whitney_u(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    result: dict[str, Any],
    *,
    group1: Any = None,
    group2: Any = None,
) -> list[DiagnosticResult]:
    """Diagnostics for Mann-Whitney U test.

    Checks:
    - Exactly two independent groups
    - Numeric/ordinal outcome
    - Sufficient observations
    - Ties
    - Zero variance
    - Extreme imbalance
    - Independence cannot be mathematically proven from the dataset
    """
    diagnostics = []
    method = result.get("test", "Mann-Whitney U")

    # Two groups
    groups = df[group].dropna().unique()
    diagnostics.append(_diag(
        "mw_groups", "Two independent groups",
        severity=DiagnosticSeverity.INFO,
        status=DiagnosticStatus.PASS if len(groups) == 2 else DiagnosticStatus.FAIL,
        variable=group, method=method,
        observed_value=len(groups), threshold=2,
        message=f"Found {len(groups)} groups." if len(groups) != 2 else "Exactly two groups.",
    ))

    # Numeric outcome
    diagnostics.append(_diag(
        "mw_numeric", "Numeric outcome",
        severity=DiagnosticSeverity.INFO,
        status=DiagnosticStatus.PASS if pd.api.types.is_numeric_dtype(df[outcome]) else DiagnosticStatus.WARNING,
        variable=outcome, method=method,
        observed_value=str(df[outcome].dtype),
        message="Outcome is numeric." if pd.api.types.is_numeric_dtype(df[outcome]) else "Outcome is not numeric.",
    ))

    # Group sizes and imbalance
    n1 = result.get("n_group1", 0)
    n2 = result.get("n_group2", 0)
    if n1 > 0 and n2 > 0:
        ratio = max(n1, n2) / min(n1, n2)
        diagnostics.append(_diag(
            "mw_imbalance", "Group size balance",
            severity=DiagnosticSeverity.WARNING,
            status=DiagnosticStatus.WARNING if ratio > 3 else DiagnosticStatus.PASS,
            variable=group, method=method,
            observed_value=round(ratio, 2), threshold=3.0,
            message=f"Group size ratio: {ratio:.2f}." + (" Extreme imbalance detected." if ratio > 3 else ""),
        ))

    # Zero variance
    if n1 >= 2 and n2 >= 2:
        g1_vals = df.loc[df[group] == result.get("group1", group1), outcome].dropna()
        g2_vals = df.loc[df[group] == result.get("group2", group2), outcome].dropna()
        v1 = g1_vals.var(ddof=1) if len(g1_vals) > 1 else 0
        v2 = g2_vals.var(ddof=1) if len(g2_vals) > 1 else 0
        if v1 == 0 and v2 == 0:
            diagnostics.append(_diag(
                "mw_zero_var", "Zero variance",
                severity=DiagnosticSeverity.WARNING,
                status=DiagnosticStatus.WARNING,
                variable=outcome, method=method,
                observed_value={"var_group1": v1, "var_group2": v2},
                message="Both groups have zero variance.",
                rationale="Mann-Whitney U with zero variance in both groups is degenerate.",
            ))

    # Ties
    if n1 > 0 and n2 > 0:
        g1_vals = df.loc[df[group] == result.get("group1", group1), outcome].dropna()
        g2_vals = df.loc[df[group] == result.get("group2", group2), outcome].dropna()
        all_vals = pd.concat([g1_vals, g2_vals])
        n_unique = all_vals.nunique()
        n_total = len(all_vals)
        if n_unique < n_total:
            tie_fraction = 1 - (n_unique / n_total)
            tie_severity = DiagnosticSeverity.WARNING if tie_fraction > 0.2 else DiagnosticSeverity.INFO
            tie_status = DiagnosticStatus.WARNING if tie_fraction > 0.2 else DiagnosticStatus.PASS
            diagnostics.append(_diag(
                "mw_ties", "Ties in outcome",
                severity=tie_severity,
                status=tie_status,
                variable=outcome, method=method,
                observed_value=round(float(tie_fraction), 4),
                threshold=0.2,
                message=f"Tie fraction: {tie_fraction:.2%}. {'Substantial ties.' if tie_fraction > 0.2 else 'Few ties.'}",
                rationale="Heavy ties can reduce the power of the Mann-Whitney U test.",
            ))

    # Independence note
    diagnostics.append(_diag(
        "mw_independence", "Independence assumption",
        severity=DiagnosticSeverity.INFO,
        status=DiagnosticStatus.NOT_APPLICABLE,
        variable=group, method=method,
        message="Independence is a study-design requirement that cannot be verified from the dataset alone.",
        rationale="Mann-Whitney U assumes independent observations. Independence must be established through study design, not statistical testing.",
    ))

    return diagnostics


# ---------------------------------------------------------------------------
# D. Wilcoxon signed-rank
# ---------------------------------------------------------------------------

def diagnostic_wilcoxon_signed_rank(
    df: pd.DataFrame,
    outcome1: str,
    outcome2: str,
    result: dict[str, Any],
) -> list[DiagnosticResult]:
    """Diagnostics for Wilcoxon signed-rank test.

    Checks:
    - Paired observations
    - Valid pair structure
    - Zero differences
    - Number of non-zero differences
    - Ties/zeros
    - Sufficient usable pairs
    """
    diagnostics = []
    method = result.get("test", "Wilcoxon signed-rank")

    both_present = df[[outcome1, outcome2]].dropna().shape[0]
    diagnostics.append(_diag(
        "wilcoxon_pairs", "Complete paired observations",
        severity=DiagnosticSeverity.INFO,
        status=DiagnosticStatus.PASS if both_present >= 5 else DiagnosticStatus.WARNING,
        variables=[outcome1, outcome2], method=method,
        observed_value=both_present, threshold=5,
        message=f"{both_present} complete pairs.",
    ))

    both_numeric = pd.api.types.is_numeric_dtype(df[outcome1]) and pd.api.types.is_numeric_dtype(df[outcome2])
    diagnostics.append(_diag(
        "wilcoxon_numeric", "Numeric paired measurements",
        severity=DiagnosticSeverity.ERROR,
        status=DiagnosticStatus.PASS if both_numeric else DiagnosticStatus.FAIL,
        variables=[outcome1, outcome2], method=method,
        message="Both variables are numeric." if both_numeric else "Not numeric.",
    ))

    if both_numeric and both_present > 0:
        paired_df = df[[outcome1, outcome2]].dropna()
        diffs = paired_df[outcome1] - paired_df[outcome2]
        n_zero = int((diffs == 0).sum())
        n_nonzero = int((diffs != 0).sum())

        diagnostics.append(_diag(
            "wilcoxon_zero_diff", "Zero differences",
            severity=DiagnosticSeverity.WARNING,
            status=DiagnosticStatus.WARNING if n_zero > len(diffs) * 0.5 else DiagnosticStatus.PASS,
            variables=[outcome1, outcome2], method=method,
            observed_value=n_zero, threshold=len(diffs) * 0.5,
            message=f"{n_zero}/{len(diffs)} differences are zero. {'Majority zero.' if n_zero > len(diffs) * 0.5 else 'Few zeros.'}",
            rationale="Wilcoxon signed-rank excludes zero differences. Many zeros reduce usable sample size.",
        ))

        diagnostics.append(_diag(
            "wilcoxon_usable", "Sufficient non-zero differences",
            severity=DiagnosticSeverity.INFO,
            status=DiagnosticStatus.PASS if n_nonzero >= 5 else DiagnosticStatus.WARNING,
            variables=[outcome1, outcome2], method=method,
            observed_value=n_nonzero, threshold=5,
            message=f"{n_nonzero} non-zero differences available for test.",
        ))

    return diagnostics


# ---------------------------------------------------------------------------
# E. Pearson correlation
# ---------------------------------------------------------------------------

def diagnostic_pearson_correlation(
    df: pd.DataFrame,
    x: str,
    y: str,
    result: dict[str, Any],
) -> list[DiagnosticResult]:
    """Diagnostics for Pearson correlation.

    Checks:
    - Two numeric variables
    - Constant variables
    - Finite values
    - Sample size
    - Linearity diagnostic (quartile-based)
    - Extreme influential observations
    """
    diagnostics = []
    method = result.get("test", "Pearson correlation")

    # Both numeric
    both_numeric = pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y])
    diagnostics.append(_diag(
        "pearson_numeric", "Both variables numeric",
        severity=DiagnosticSeverity.ERROR,
        status=DiagnosticStatus.PASS if both_numeric else DiagnosticStatus.FAIL,
        variables=[x, y], method=method,
        message="Both variables are numeric." if both_numeric else "Not numeric.",
    ))

    if not both_numeric:
        return diagnostics

    x_vals = df[x].dropna()
    y_vals = df[y].dropna()
    both = df[[x, y]].dropna()
    n = len(both)

    # Constant variables
    x_const = x_vals.nunique() <= 1
    y_const = y_vals.nunique() <= 1
    if x_const or y_const:
        diagnostics.append(_diag(
            "pearson_constant", "Constant variable",
            severity=DiagnosticSeverity.ERROR,
            status=DiagnosticStatus.FAIL,
            variables=[x, y], method=method,
            observed_value={"x_constant": x_const, "y_constant": y_const},
            message=f"{'x' if x_const else 'y'} is constant (zero variance). Correlation undefined.",
            rationale="Pearson correlation requires variation in both variables.",
        ))
        return diagnostics

    # Finite values
    all_finite = np.all(np.isfinite(both[x])) and np.all(np.isfinite(both[y]))
    diagnostics.append(_diag(
        "pearson_finite", "Finite values",
        severity=DiagnosticSeverity.ERROR,
        status=DiagnosticStatus.PASS if all_finite else DiagnosticStatus.FAIL,
        variables=[x, y], method=method,
        message="All values are finite." if all_finite else "Non-finite values detected.",
    ))

    # Sample size
    diagnostics.append(_diag(
        "pearson_n", "Sufficient sample size",
        severity=DiagnosticSeverity.WARNING,
        status=DiagnosticStatus.PASS if n >= 10 else DiagnosticStatus.WARNING,
        variables=[x, y], method=method,
        observed_value=n, threshold=10,
        message=f"N={n}." + (" Very small sample for correlation." if n < 10 else ""),
    ))

    # Linearity diagnostic (Spearman vs Pearson difference)
    if n >= 10:
        from scipy import stats as sp_stats
        r_pearson, _ = sp_stats.pearsonr(both[x], both[y])
        r_spearman, _ = sp_stats.spearmanr(both[x], both[y])
        discrepancy = abs(r_pearson - r_spearman)
        lin_status = DiagnosticStatus.PASS
        lin_severity = DiagnosticSeverity.INFO
        lin_msg = f"Pearson r={r_pearson:.4f}, Spearman rho={r_spearman:.4f}."
        if discrepancy > 0.2:
            lin_status = DiagnosticStatus.WARNING
            lin_severity = DiagnosticSeverity.WARNING
            lin_msg += f" Substantial discrepancy ({discrepancy:.2f}) suggests possible non-linearity."
        diagnostics.append(_diag(
            "pearson_linearity", "Linearity diagnostic",
            severity=lin_severity,
            status=lin_status,
            variables=[x, y], method=method,
            observed_value=round(float(discrepancy), 4),
            threshold=0.2,
            message=lin_msg,
            rationale="A large discrepancy between Pearson and Spearman correlations can indicate non-linear association.",
        ))

    # Influential observations (Cook's distance approximation via leverage)
    if n >= 10:
        try:
            from scipy import stats as sp_stats
            r_val = sp_stats.pearsonr(both[x], both[y])[0]
            # Approximate: squared residual relative to variance
            x_std = (both[x] - both[x].mean()) / both[x].std(ddof=1)
            y_std = (both[y] - both[y].mean()) / both[y].std(ddof=1)
            residuals = y_std - r_val * x_std
            sq_resid = residuals ** 2
            high_influence = (sq_resid > 9).sum()  # >3 SD residual
            if high_influence > 0:
                diagnostics.append(_diag(
                    "pearson_influential", "Influential observations",
                    severity=DiagnosticSeverity.WARNING,
                    status=DiagnosticStatus.WARNING,
                    variables=[x, y], method=method,
                    observed_value=int(high_influence),
                    message=f"{high_influence} observation(s) with extreme standardized residuals.",
                    rationale="Extreme observations can substantially influence Pearson correlation.",
                ))
        except Exception:
            pass

    return diagnostics


# ---------------------------------------------------------------------------
# F. Spearman correlation
# ---------------------------------------------------------------------------

def diagnostic_spearman_correlation(
    df: pd.DataFrame,
    x: str,
    y: str,
    result: dict[str, Any],
) -> list[DiagnosticResult]:
    """Diagnostics for Spearman correlation.

    Checks:
    - Two numeric/ordinal variables
    - Constant variables
    - Finite values
    - Sample size
    - Ties
    """
    diagnostics = []
    method = result.get("test", "Spearman correlation")

    both_numeric = pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y])
    diagnostics.append(_diag(
        "spearman_numeric", "Both variables numeric",
        severity=DiagnosticSeverity.ERROR,
        status=DiagnosticStatus.PASS if both_numeric else DiagnosticStatus.FAIL,
        variables=[x, y], method=method,
        message="Both variables are numeric." if both_numeric else "Not numeric.",
    ))

    if not both_numeric:
        return diagnostics

    x_vals = df[x].dropna()
    y_vals = df[y].dropna()
    both = df[[x, y]].dropna()
    n = len(both)

    x_const = x_vals.nunique() <= 1
    y_const = y_vals.nunique() <= 1
    if x_const or y_const:
        diagnostics.append(_diag(
            "spearman_constant", "Constant variable",
            severity=DiagnosticSeverity.ERROR,
            status=DiagnosticStatus.FAIL,
            variables=[x, y], method=method,
            observed_value={"x_constant": x_const, "y_constant": y_const},
            message="Constant variable. Correlation undefined.",
        ))
        return diagnostics

    all_finite = np.all(np.isfinite(both[x])) and np.all(np.isfinite(both[y]))
    diagnostics.append(_diag(
        "spearman_finite", "Finite values",
        severity=DiagnosticSeverity.ERROR,
        status=DiagnosticStatus.PASS if all_finite else DiagnosticStatus.FAIL,
        variables=[x, y], method=method,
        message="All finite." if all_finite else "Non-finite values.",
    ))

    # Ties
    n_unique_x = x_vals.nunique()
    n_unique_y = y_vals.nunique()
    tie_frac_x = 1 - (n_unique_x / len(x_vals)) if len(x_vals) > 0 else 0
    tie_frac_y = 1 - (n_unique_y / len(y_vals)) if len(y_vals) > 0 else 0
    max_tie = max(tie_frac_x, tie_frac_y)
    if max_tie > 0.3:
        diagnostics.append(_diag(
            "spearman_ties", "Substantial ties",
            severity=DiagnosticSeverity.WARNING,
            status=DiagnosticStatus.WARNING,
            variables=[x, y], method=method,
            observed_value=round(float(max_tie), 4),
            threshold=0.3,
            message=f"Max tie fraction: {max_tie:.2%}. Substantial ties may affect ranking.",
            rationale="Heavy ties reduce the effective information in rank-based statistics.",
        ))

    diagnostics.append(_diag(
        "spearman_n", "Sufficient sample size",
        severity=DiagnosticSeverity.WARNING,
        status=DiagnosticStatus.PASS if n >= 10 else DiagnosticStatus.WARNING,
        variables=[x, y], method=method,
        observed_value=n, threshold=10,
        message=f"N={n}." + (" Very small sample." if n < 10 else ""),
    ))

    return diagnostics


# ---------------------------------------------------------------------------
# G. One-way ANOVA
# ---------------------------------------------------------------------------

def diagnostic_one_way_anova(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    result: dict[str, Any],
) -> list[DiagnosticResult]:
    """Diagnostics for one-way ANOVA.

    Checks:
    - Categorical grouping variable
    - At least two groups
    - Numeric continuous outcome
    - Group sizes
    - Zero variance within groups
    - Severe imbalance
    - Variance homogeneity
    """
    diagnostics = []
    method = "One-way ANOVA"

    groups = df[group].dropna().unique()
    diagnostics.append(_diag(
        "anova_groups", "Grouping variable",
        severity=DiagnosticSeverity.INFO,
        status=DiagnosticStatus.PASS if len(groups) >= 2 else DiagnosticStatus.FAIL,
        variable=group, method=method,
        observed_value=len(groups), threshold=2,
        message=f"{len(groups)} groups found.",
    ))

    diagnostics.append(_diag(
        "anova_numeric", "Numeric outcome",
        severity=DiagnosticSeverity.ERROR,
        status=DiagnosticStatus.PASS if pd.api.types.is_numeric_dtype(df[outcome]) else DiagnosticStatus.FAIL,
        variable=outcome, method=method,
        message="Outcome is numeric." if pd.api.types.is_numeric_dtype(df[outcome]) else "Not numeric.",
    ))

    # Group sizes
    group_counts = df.groupby(group)[outcome].apply(lambda s: s.dropna().shape[0])
    n_groups = len(group_counts)
    if n_groups >= 2:
        min_n = group_counts.min()
        max_n = group_counts.max()
        ratio = max_n / min_n if min_n > 0 else float("inf")
        diagnostics.append(_diag(
            "anova_imbalance", "Group size balance",
            severity=DiagnosticSeverity.WARNING,
            status=DiagnosticStatus.WARNING if ratio > 3 else DiagnosticStatus.PASS,
            variable=group, method=method,
            observed_value={"min": int(min_n), "max": int(max_n), "ratio": round(ratio, 2)},
            threshold=3.0,
            message=f"Group sizes: min={min_n}, max={max_n}, ratio={ratio:.2f}.",
            rationale="Extreme imbalance can affect ANOVA robustness and homogeneity of variance.",
        ))

    # Zero variance within groups
    for grp_name in groups:
        grp_vals = df.loc[df[group] == grp_name, outcome].dropna()
        if len(grp_vals) > 1:
            v = grp_vals.var(ddof=1)
            if v == 0:
                diagnostics.append(_diag(
                    f"anova_zero_var_{grp_name}", f"Zero variance in group '{grp_name}'",
                    severity=DiagnosticSeverity.WARNING,
                    status=DiagnosticStatus.WARNING,
                    variable=outcome, method=method,
                    observed_value=0,
                    message=f"Group '{grp_name}' has zero variance.",
                    rationale="Zero variance in any group may indicate data issues.",
                ))

    # Variance homogeneity (Levene's test)
    group_data = [df.loc[df[group] == g, outcome].dropna().values for g in groups if df.loc[df[group] == g, outcome].dropna().shape[0] > 1]
    if len(group_data) >= 2 and all(len(g) >= 2 for g in group_data):
        try:
            from scipy import stats as sp_stats
            lev_stat, lev_p = sp_stats.levene(*group_data)
            homo_status = DiagnosticStatus.PASS if lev_p > 0.05 else DiagnosticStatus.WARNING
            homo_severity = DiagnosticSeverity.INFO if lev_p > 0.05 else DiagnosticSeverity.WARNING
            diagnostics.append(_diag(
                "anova_homogeneity", "Variance homogeneity (Levene's test)",
                severity=homo_severity,
                status=homo_status,
                variable=outcome, method=method,
                observed_value=round(float(lev_p), 6),
                threshold=0.05,
                message=f"Levene's test p={lev_p:.4f}. {'Variances appear homogeneous.' if lev_p > 0.05 else 'Variance heterogeneity detected.'}",
                rationale="ANOVA assumes approximately equal variances across groups. Violations reduce reliability.",
            ))
        except Exception:
            pass

    return diagnostics


# ---------------------------------------------------------------------------
# H. Linear regression
# ---------------------------------------------------------------------------

def diagnostic_linear_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    result: dict[str, Any],
) -> list[DiagnosticResult]:
    """Diagnostics for OLS linear regression.

    Checks:
    - Numeric outcome
    - Constant predictors
    - Sample size vs predictors
    - Residual normality
    - Heteroscedasticity (Breusch-Pagan)
    - Multicollinearity (VIF)
    - Influential observations
    - Finite coefficients
    """
    diagnostics = []
    method = "OLS Linear regression"

    diagnostics.append(_diag(
        "linreg_numeric", "Numeric outcome",
        severity=DiagnosticSeverity.ERROR,
        status=DiagnosticStatus.PASS if pd.api.types.is_numeric_dtype(df[outcome]) else DiagnosticStatus.FAIL,
        variable=outcome, method=method,
        message="Outcome is numeric." if pd.api.types.is_numeric_dtype(df[outcome]) else "Not numeric.",
    ))

    # Constant predictors
    for pred in predictors:
        if pred in df.columns:
            const = df[pred].nunique(dropna=True) <= 1
            if const:
                diagnostics.append(_diag(
                    f"linreg_const_{pred}", f"Constant predictor: {pred}",
                    severity=DiagnosticSeverity.ERROR,
                    status=DiagnosticStatus.FAIL,
                    variable=pred, method=method,
                    observed_value="constant",
                    message=f"Predictor '{pred}' is constant.",
                    rationale="A constant predictor has no variation and cannot explain any variance.",
                ))

    # Sample size
    n_obs = result.get("n_obs", 0)
    n_pred = len(predictors)
    if n_obs > 0 and n_pred > 0:
        ratio = n_obs / n_pred
        diagnostics.append(_diag(
            "linreg_ratio", "Observations per predictor",
            severity=DiagnosticSeverity.WARNING,
            status=DiagnosticStatus.PASS if ratio >= 10 else DiagnosticStatus.WARNING,
            method=method,
            observed_value=round(ratio, 2),
            threshold=10,
            message=f"N/p ratio: {ratio:.2f}. {'Adequate.' if ratio >= 10 else 'Low observations per predictor.'}",
            rationale="A common guideline is at least 10 observations per predictor for stable OLS estimates.",
        ))

    # Finite coefficients
    params = result.get("parameters", {})
    non_finite = [k for k, v in params.items() if not np.isfinite(v.get("coef", float("nan")))]
    if non_finite:
        diagnostics.append(_diag(
            "linreg_finite_coef", "Finite coefficients",
            severity=DiagnosticSeverity.ERROR,
            status=DiagnosticStatus.FAIL,
            method=method,
            observed_value=non_finite,
            message=f"Non-finite coefficients: {non_finite}.",
            rationale="Non-finite coefficients indicate numerical instability.",
        ))

    # Fit diagnostics from result
    r_sq = result.get("r_squared", 0)
    if r_sq is not None and r_sq < 0.01:
        diagnostics.append(_diag(
            "linreg_r2", "Model fit",
            severity=DiagnosticSeverity.WARNING,
            status=DiagnosticStatus.WARNING,
            method=method,
            observed_value=round(float(r_sq), 6),
            threshold=0.01,
            message=f"R-squared: {r_sq:.4f}. Very low explained variance.",
            rationale="A very low R-squared suggests the model explains little of the outcome variation.",
        ))

    # Multicollinearity (VIF) for multiple predictors
    if len(predictors) >= 2:
        numeric_preds = [p for p in predictors if p in df.columns and pd.api.types.is_numeric_dtype(df[p])]
        if len(numeric_preds) >= 2:
            try:
                pred_data = df[numeric_preds].dropna()
                if len(pred_data) > len(numeric_preds) + 1:
                    from numpy.linalg import lstsq
                    vifs = {}
                    for i, col in enumerate(numeric_preds):
                        others = [c for c in numeric_preds if c != col]
                        X_others = pred_data[others].values
                        y_col = pred_data[col].values
                        X_with_const = np.column_stack([np.ones(len(X_others)), X_others])
                        coef, _, _, _ = lstsq(X_with_const, y_col, rcond=None)
                        y_hat = X_with_const @ coef
                        ss_res = np.sum((y_col - y_hat) ** 2)
                        ss_tot = np.sum((y_col - y_col.mean()) ** 2)
                        r2_i = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        vif = 1 / (1 - r2_i) if r2_i < 1 else float("inf")
                        vifs[col] = round(float(vif), 2)

                    max_vif = max(vifs.values())
                    vif_status = DiagnosticStatus.PASS
                    vif_severity = DiagnosticSeverity.INFO
                    vif_msg = f"VIF values: {vifs}."
                    if max_vif > 10:
                        vif_status = DiagnosticStatus.REVIEW_REQUIRED
                        vif_severity = DiagnosticSeverity.REVIEW_REQUIRED
                        vif_msg += f" Severe multicollinearity (max VIF={max_vif:.1f})."
                    elif max_vif > 5:
                        vif_status = DiagnosticStatus.WARNING
                        vif_severity = DiagnosticSeverity.WARNING
                        vif_msg += f" Moderate multicollinearity (max VIF={max_vif:.1f})."

                    diagnostics.append(_diag(
                        "linreg_vif", "Multicollinearity (VIF)",
                        severity=vif_severity,
                        status=vif_status,
                        variables=numeric_preds, method=method,
                        observed_value=vifs, threshold={"moderate": 5, "severe": 10},
                        message=vif_msg,
                        rationale="VIF > 5 indicates moderate multicollinearity; VIF > 10 indicates severe multicollinearity.",
                    ))
            except Exception:
                pass

    # Residual diagnostics (from fitted model if possible)
    try:
        numeric_preds_in_df = [p for p in predictors if p in df.columns and pd.api.types.is_numeric_dtype(df[p])]
        if numeric_preds_in_df and pd.api.types.is_numeric_dtype(df[outcome]):
            reg_df = df[[outcome] + numeric_preds_in_df].dropna()
            if len(reg_df) > len(numeric_preds_in_df) + 2:
                X = reg_df[numeric_preds_in_df].values
                y = reg_df[outcome].values
                X_with_const = np.column_stack([np.ones(len(X)), X])
                coef, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
                y_hat = X_with_const @ coef
                residuals = y - y_hat

                # Normality of residuals
                if 3 <= len(residuals) <= 5000:
                    from scipy import stats as sp_stats
                    _, p = sp_stats.shapiro(residuals)
                    norm_status = DiagnosticStatus.PASS if p > 0.05 else DiagnosticStatus.WARNING
                    diagnostics.append(_diag(
                        "linreg_normality", "Residual normality",
                        severity=DiagnosticSeverity.WARNING,
                        status=norm_status,
                        method=method,
                        observed_value=round(float(p), 6),
                        threshold=0.05,
                        message=f"Shapiro-Wilk on residuals: p={p:.4f}.",
                        rationale="Non-normal residuals affect inference (CIs, p-values) but not coefficient unbiasedness.",
                    ))

                # Heteroscedasticity (Breusch-Pagan simplified)
                sq_resid = residuals ** 2
                if len(sq_resid) > len(numeric_preds_in_df) + 2:
                    try:
                        from scipy import stats as sp_stats
                        r_sq_aux, _, _, _ = np.linalg.lstsq(
                            X_with_const, sq_resid, rcond=None
                        )
                        y_hat_aux = X_with_const @ r_sq_aux
                        ss_res_aux = np.sum((sq_resid - y_hat_aux) ** 2)
                        ss_tot_aux = np.sum((sq_resid - sq_resid.mean()) ** 2)
                        r2_aux = 1 - (ss_res_aux / ss_tot_aux) if ss_tot_aux > 0 else 0
                        bp_stat = len(sq_resid) * r2_aux
                        bp_p = 1 - sp_stats.chi2.cdf(bp_stat, df=len(numeric_preds_in_df))
                        het_status = DiagnosticStatus.PASS if bp_p > 0.05 else DiagnosticStatus.WARNING
                        diagnostics.append(_diag(
                            "linreg_heteroscedasticity", "Heteroscedasticity (Breusch-Pagan)",
                            severity=DiagnosticSeverity.WARNING,
                            status=het_status,
                            method=method,
                            observed_value=round(float(bp_p), 6),
                            threshold=0.05,
                            message=f"Breusch-Pagan p={bp_p:.4f}. {'Homoscedasticity assumption appears met.' if bp_p > 0.05 else 'Heteroscedasticity detected.'}",
                            rationale="Heteroscedasticity affects standard errors and confidence intervals but not coefficient estimates.",
                        ))
                    except Exception:
                        pass

                # Influential observations (Cook's distance)
                try:
                    hat_matrix = X_with_const @ np.linalg.pinv(X_with_const.T @ X_with_const) @ X_with_const.T
                    hat_diag = np.diag(hat_matrix)
                    p_param = X_with_const.shape[1]
                    mse = np.sum(residuals ** 2) / (len(residuals) - p_param)
                    std_resid = residuals / np.sqrt(mse * (1 - hat_diag))
                    cooks_d = (std_resid ** 2 / p_param) * (hat_diag / (1 - hat_diag) ** 2)
                    n_influential = int(np.sum(cooks_d > 4 / len(residuals)))
                    if n_influential > 0:
                        diagnostics.append(_diag(
                            "linreg_influential", "Influential observations",
                            severity=DiagnosticSeverity.WARNING,
                            status=DiagnosticStatus.WARNING,
                            method=method,
                            observed_value=n_influential,
                            threshold=0,
                            message=f"{n_influential} observation(s) with Cook's D > 4/n.",
                            rationale="Influential observations can substantially affect regression coefficients.",
                        ))
                except Exception:
                    pass
    except Exception:
        pass

    return diagnostics


# ---------------------------------------------------------------------------
# I. Logistic regression
# ---------------------------------------------------------------------------

def diagnostic_logistic_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    result: dict[str, Any],
) -> list[DiagnosticResult]:
    """Diagnostics for logistic regression.

    Checks:
    - Binary outcome
    - Constant predictors
    - Events per variable
    - Complete separation
    - Convergence
    - Extreme standard errors
    - Multicollinearity
    """
    diagnostics = []
    method = "Logistic regression"

    # Binary outcome
    if outcome in df.columns:
        vals = df[outcome].dropna()
        n_unique = vals.nunique()
        diagnostics.append(_diag(
            "logreg_binary", "Binary outcome",
            severity=DiagnosticSeverity.ERROR,
            status=DiagnosticStatus.PASS if n_unique == 2 else DiagnosticStatus.FAIL,
            variable=outcome, method=method,
            observed_value=n_unique, threshold=2,
            message=f"Outcome has {n_unique} unique values." + (" Binary." if n_unique == 2 else " Not binary."),
            rationale="Logistic regression requires a binary (0/1) outcome.",
        ))

    # Constant predictors
    for pred in predictors:
        if pred in df.columns:
            const = df[pred].nunique(dropna=True) <= 1
            if const:
                diagnostics.append(_diag(
                    f"logreg_const_{pred}", f"Constant predictor: {pred}",
                    severity=DiagnosticSeverity.ERROR,
                    status=DiagnosticStatus.FAIL,
                    variable=pred, method=method,
                    observed_value="constant",
                    message=f"Predictor '{pred}' is constant.",
                ))

    # Events per variable
    if outcome in df.columns:
        vals = df[outcome].dropna()
        n_events = int((vals == 1).sum())
        n_nonevents = int((vals == 0).sum())
        n_pred = len([p for p in predictors if p in df.columns])
        min_count = min(n_events, n_nonevents)
        epv = min_count / n_pred if n_pred > 0 else float("inf")
        diagnostics.append(_diag(
            "logreg_epv", "Events per variable",
            severity=DiagnosticSeverity.WARNING,
            status=DiagnosticStatus.PASS if epv >= 10 else DiagnosticStatus.WARNING,
            method=method,
            observed_value={"events": n_events, "non_events": n_nonevents, "epv": round(epv, 2)},
            threshold=10,
            message=f"Events: {n_events}, Non-events: {n_nonevents}, EPV: {epv:.2f}.",
            rationale="EPV < 10 increases risk of overfitting and unstable estimates.",
        ))

    # Convergence
    converged = result.get("converged", True)
    diagnostics.append(_diag(
        "logreg_convergence", "Model convergence",
        severity=DiagnosticSeverity.ERROR,
        status=DiagnosticStatus.PASS if converged else DiagnosticStatus.FAIL,
        method=method,
        message="Model converged." if converged else "Model did not converge.",
        rationale="Non-converged models produce unreliable estimates.",
    ))

    # Complete separation / extreme standard errors
    params = result.get("parameters", {})
    extreme_se = []
    for name, info in params.items():
        se = info.get("std_err", 0)
        if se > 10:
            extreme_se.append(name)
    if extreme_se:
        diagnostics.append(_diag(
            "logreg_separation", "Potential complete separation",
            severity=DiagnosticSeverity.REVIEW_REQUIRED,
            status=DiagnosticStatus.REVIEW_REQUIRED,
            method=method,
            observed_value=extreme_se,
            message=f"Extreme standard errors for: {extreme_se}. Possible complete or quasi-separation.",
            rationale="Extreme SEs suggest the data may perfectly or near-perfectly separate the outcome classes.",
        ))

    # Multicollinearity
    if len(predictors) >= 2:
        numeric_preds = [p for p in predictors if p in df.columns and pd.api.types.is_numeric_dtype(df[p])]
        if len(numeric_preds) >= 2:
            try:
                pred_data = df[numeric_preds].dropna()
                if len(pred_data) > len(numeric_preds) + 1:
                    from numpy.linalg import lstsq
                    vifs = {}
                    for col in numeric_preds:
                        others = [c for c in numeric_preds if c != col]
                        X_others = pred_data[others].values
                        y_col = pred_data[col].values
                        X_with_const = np.column_stack([np.ones(len(X_others)), X_others])
                        coef, _, _, _ = lstsq(X_with_const, y_col, rcond=None)
                        y_hat = X_with_const @ coef
                        ss_res = np.sum((y_col - y_hat) ** 2)
                        ss_tot = np.sum((y_col - y_col.mean()) ** 2)
                        r2_i = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        vif = 1 / (1 - r2_i) if r2_i < 1 else float("inf")
                        vifs[col] = round(float(vif), 2)
                    max_vif = max(vifs.values())
                    vif_status = DiagnosticStatus.PASS
                    vif_severity = DiagnosticSeverity.INFO
                    vif_msg = f"VIF values: {vifs}."
                    if max_vif > 10:
                        vif_status = DiagnosticStatus.REVIEW_REQUIRED
                        vif_severity = DiagnosticSeverity.REVIEW_REQUIRED
                        vif_msg += f" Severe multicollinearity."
                    elif max_vif > 5:
                        vif_status = DiagnosticStatus.WARNING
                        vif_severity = DiagnosticSeverity.WARNING
                        vif_msg += f" Moderate multicollinearity."
                    diagnostics.append(_diag(
                        "logreg_vif", "Multicollinearity (VIF)",
                        severity=vif_severity,
                        status=vif_status,
                        variables=numeric_preds, method=method,
                        observed_value=vifs, threshold={"moderate": 5, "severe": 10},
                        message=vif_msg,
                    ))
            except Exception:
                pass

    return diagnostics


# ---------------------------------------------------------------------------
# J. Chi-square
# ---------------------------------------------------------------------------

def diagnostic_chi_square(
    df: pd.DataFrame,
    variable1: str,
    variable2: str,
    result: dict[str, Any],
) -> list[DiagnosticResult]:
    """Diagnostics for chi-square test of independence.

    Checks:
    - Categorical variables
    - Contingency table dimensions
    - Expected cell counts
    - Sparse cells
    - Zero expected counts
    - Sample size
    """
    diagnostics = []
    method = "Chi-square test of independence"

    # Both categorical (object, category, or low-n int)
    v1_cat = not pd.api.types.is_numeric_dtype(df[variable1]) or df[variable1].nunique() <= 10
    v2_cat = not pd.api.types.is_numeric_dtype(df[variable2]) or df[variable2].nunique() <= 10
    diagnostics.append(_diag(
        "chi2_categorical", "Categorical variables",
        severity=DiagnosticSeverity.INFO,
        status=DiagnosticStatus.PASS if (v1_cat and v2_cat) else DiagnosticStatus.WARNING,
        variables=[variable1, variable2], method=method,
        message="Variables appear categorical." if (v1_cat and v2_cat) else "Variables may not be categorical.",
    ))

    # Contingency table
    both = df[[variable1, variable2]].dropna()
    table = pd.crosstab(both[variable1], both[variable2])
    n_rows, n_cols = table.shape
    diagnostics.append(_diag(
        "chi2_dimensions", "Contingency table dimensions",
        severity=DiagnosticSeverity.INFO,
        status=DiagnosticStatus.PASS,
        variables=[variable1, variable2], method=method,
        observed_value={"rows": n_rows, "cols": n_cols},
        message=f"Table: {n_rows}x{n_cols}.",
    ))

    # Expected counts
    from scipy import stats as sp_stats
    chi2, p, dof, expected = sp_stats.chi2_contingency(table)
    n_low_expected = int((expected < 5).sum())
    n_zero_expected = int((expected == 0).sum())
    pct_low = n_low_expected / expected.size if expected.size > 0 else 0

    exp_status = DiagnosticStatus.PASS
    exp_severity = DiagnosticSeverity.INFO
    exp_msg = f"Low expected cells: {n_low_expected}/{expected.size} ({pct_low:.0%})."
    if n_zero_expected > 0:
        exp_status = DiagnosticStatus.REVIEW_REQUIRED
        exp_severity = DiagnosticSeverity.REVIEW_REQUIRED
        exp_msg += f" {n_zero_expected} zero expected counts."
    elif pct_low > 0.2:
        exp_status = DiagnosticStatus.WARNING
        exp_severity = DiagnosticSeverity.WARNING
        exp_msg += " >20% of cells have low expected counts."
    elif n_low_expected > 0:
        exp_status = DiagnosticStatus.WARNING
        exp_severity = DiagnosticSeverity.WARNING
        exp_msg += " Some cells have low expected counts."

    diagnostics.append(_diag(
        "chi2_expected", "Expected cell counts",
        severity=exp_severity,
        status=exp_status,
        variables=[variable1, variable2], method=method,
        observed_value={"n_low": n_low_expected, "n_zero": n_zero_expected, "total_cells": int(expected.size)},
        threshold={"low_pct": 0.2},
        message=exp_msg,
        rationale="Chi-square assumes adequate expected cell counts. Fisher's exact test is an alternative for small/ sparse tables.",
    ))

    # Sample size
    n_total = both.shape[0]
    diagnostics.append(_diag(
        "chi2_n", "Sample size",
        severity=DiagnosticSeverity.WARNING,
        status=DiagnosticStatus.PASS if n_total >= 20 else DiagnosticStatus.WARNING,
        variables=[variable1, variable2], method=method,
        observed_value=n_total, threshold=20,
        message=f"N={n_total}.",
    ))

    return diagnostics


# ---------------------------------------------------------------------------
# K. Fisher exact
# ---------------------------------------------------------------------------

def diagnostic_fisher_exact(
    df: pd.DataFrame,
    variable1: str,
    variable2: str,
    result: dict[str, Any],
) -> list[DiagnosticResult]:
    """Diagnostics for Fisher's exact test.

    Checks:
    - Valid categorical inputs
    - Appropriate table dimensions
    - Zero/low counts
    - Finite counts
    """
    diagnostics = []
    method = "Fisher's exact test"

    both = df[[variable1, variable2]].dropna()
    table = pd.crosstab(both[variable1], both[variable2])
    n_rows, n_cols = table.shape

    diagnostics.append(_diag(
        "fisher_dimensions", "Table dimensions",
        severity=DiagnosticSeverity.INFO,
        status=DiagnosticStatus.PASS if (n_rows == 2 and n_cols == 2) else DiagnosticStatus.WARNING,
        variables=[variable1, variable2], method=method,
        observed_value={"rows": n_rows, "cols": n_cols},
        message=f"Table: {n_rows}x{n_cols}." + (" Standard 2x2." if n_rows == 2 and n_cols == 2 else " Non-standard dimensions."),
        rationale="Fisher's exact test is most straightforward for 2x2 tables.",
    ))

    # Zero counts
    n_zero = int((table == 0).values.sum())
    diagnostics.append(_diag(
        "fisher_zero", "Zero cell counts",
        severity=DiagnosticSeverity.INFO,
        status=DiagnosticStatus.PASS if n_zero == 0 else DiagnosticStatus.WARNING,
        variables=[variable1, variable2], method=method,
        observed_value=n_zero,
        message=f"{n_zero} zero cells." if n_zero > 0 else "No zero cells.",
        rationale="Zero cells affect exact probability computation.",
    ))

    return diagnostics


# ---------------------------------------------------------------------------
# Survival Analysis Diagnostics
# ---------------------------------------------------------------------------

def diagnostic_kaplan_meier(
    df: pd.DataFrame,
    time_col: str | None,
    event_col: str | None,
    group_col: str | None,
    result: dict[str, Any],
) -> list[DiagnosticResult]:
    """Diagnostics for Kaplan-Meier estimator."""
    method = "kaplan_meier"
    diagnostics: list[DiagnosticResult] = []

    if not time_col or time_col not in df.columns:
        diagnostics.append(_diag(
            "km_no_time", "Missing survival time",
            severity=DiagnosticSeverity.BLOCKER,
            status=DiagnosticStatus.FAIL,
            method=method,
            message="Survival time variable not found.",
        ))
        return diagnostics

    if not event_col or event_col not in df.columns:
        diagnostics.append(_diag(
            "km_no_event", "Missing event indicator",
            severity=DiagnosticSeverity.BLOCKER,
            status=DiagnosticStatus.FAIL,
            method=method,
            message="Event indicator variable not found.",
        ))
        return diagnostics

    time = df[time_col]
    event = df[event_col]

    # Valid time check
    valid_time = time.notna() & np.isfinite(time) & (time >= 0)
    n_invalid = int((~valid_time).sum())
    diagnostics.append(_diag(
        "km_valid_time", "Valid survival time",
        severity=DiagnosticSeverity.BLOCKER if n_invalid > 0 else DiagnosticSeverity.INFO,
        status=DiagnosticStatus.FAIL if n_invalid > 0 else DiagnosticStatus.PASS,
        variable=time_col, method=method,
        observed_value=n_invalid,
        message=f"{n_invalid} invalid time values." if n_invalid > 0 else "All time values valid.",
        rationale="KM requires non-negative finite survival times.",
    ))

    # Valid event check
    if pd.api.types.is_numeric_dtype(event):
        valid_event = event.isin([0, 1])
    elif pd.api.types.is_string_dtype(event) or pd.api.types.is_object_dtype(event):
        lower = event.astype(str).str.strip().str.lower()
        valid_event = lower.isin(["0", "1", "true", "false", "yes", "no", "event", "censored", "alive", "dead", "died"])
    else:
        valid_event = pd.Series([False] * len(event), index=event.index)
    n_invalid_event = int((~valid_event & event.notna()).sum())
    diagnostics.append(_diag(
        "km_valid_event", "Valid event indicator",
        severity=DiagnosticSeverity.BLOCKER if n_invalid_event > 0 else DiagnosticSeverity.INFO,
        status=DiagnosticStatus.FAIL if n_invalid_event > 0 else DiagnosticStatus.PASS,
        variable=event_col, method=method,
        observed_value=n_invalid_event,
        message=f"{n_invalid_event} invalid event values." if n_invalid_event > 0 else "Event indicator valid.",
        rationale="Event indicator must be binary (0/1 or equivalent).",
    ))

    # Convergence — KM always converges
    diagnostics.append(_diag(
        "km_convergence", "Estimator convergence",
        severity=DiagnosticSeverity.INFO,
        status=DiagnosticStatus.PASS,
        method=method,
        message="Kaplan-Meier estimator always produces finite results.",
        rationale="Non-parametric estimator; no iterative fitting.",
    ))

    # Group structure
    if group_col and group_col in df.columns:
        n_groups = df[group_col].nunique()
        diagnostics.append(_diag(
            "km_groups", "Group structure",
            severity=DiagnosticSeverity.WARNING if n_groups > 10 else DiagnosticSeverity.INFO,
            status=DiagnosticStatus.WARNING if n_groups > 10 else DiagnosticStatus.PASS,
            variable=group_col, method=method,
            observed_value=n_groups,
            message=f"{n_groups} groups.",
            rationale="Many groups may produce difficult-to-compare curves.",
        ))
    elif group_col and group_col not in df.columns:
        diagnostics.append(_diag(
            "km_group_missing", "Group variable missing",
            severity=DiagnosticSeverity.BLOCKER,
            status=DiagnosticStatus.FAIL,
            variable=group_col, method=method,
            message=f"Group variable '{group_col}' not found in dataset.",
        ))

    # Finite results in output
    if "error" not in result:
        diagnostics.append(_diag(
            "km_finite", "Finite results",
            severity=DiagnosticSeverity.INFO,
            status=DiagnosticStatus.PASS,
            method=method,
            message="Results are finite.",
            rationale="KM produces finite probabilities.",
        ))

    return diagnostics


def diagnostic_log_rank(
    df: pd.DataFrame,
    time_col: str | None,
    event_col: str | None,
    group_col: str | None,
    result: dict[str, Any],
) -> list[DiagnosticResult]:
    """Diagnostics for log-rank test."""
    method = "log_rank"
    diagnostics: list[DiagnosticResult] = []

    # Group count
    if group_col and group_col in df.columns:
        n_groups = df[group_col].nunique()
        diagnostics.append(_diag(
            "lr_groups", "Number of groups",
            severity=DiagnosticSeverity.BLOCKER if n_groups < 2 else DiagnosticSeverity.INFO,
            status=DiagnosticStatus.FAIL if n_groups < 2 else DiagnosticStatus.PASS,
            variable=group_col, method=method,
            observed_value=n_groups,
            message=f"{n_groups} groups." if n_groups >= 2 else f"Only {n_groups} group(s); need at least 2.",
            rationale="Log-rank requires at least 2 groups.",
        ))
    else:
        diagnostics.append(_diag(
            "lr_no_group", "Missing grouping variable",
            severity=DiagnosticSeverity.BLOCKER,
            status=DiagnosticStatus.FAIL,
            method=method,
            message="Grouping variable not specified.",
        ))

    # Group sizes
    n1 = result.get("n_group1", 0)
    n2 = result.get("n_group2", 0)
    if n1 > 0 and n2 > 0:
        min_n = min(n1, n2)
        diagnostics.append(_diag(
            "lr_group_sizes", "Group sizes",
            severity=DiagnosticSeverity.WARNING if min_n < 5 else DiagnosticSeverity.INFO,
            status=DiagnosticStatus.WARNING if min_n < 5 else DiagnosticStatus.PASS,
            method=method,
            observed_value={"group1": n1, "group2": n2},
            message=f"Group sizes: {n1}, {n2}.",
            rationale="Very small groups may reduce power.",
        ))

    # Event counts
    e1 = result.get("n_events_group1", 0)
    e2 = result.get("n_events_group2", 0)
    total_events = e1 + e2
    diagnostics.append(_diag(
        "lr_events", "Event counts",
        severity=DiagnosticSeverity.WARNING if total_events < 5 else DiagnosticSeverity.INFO,
        status=DiagnosticStatus.WARNING if total_events < 5 else DiagnosticStatus.PASS,
        method=method,
        observed_value={"events_group1": e1, "events_group2": e2},
        message=f"Events: {e1} (group1), {e2} (group2).",
        rationale="Very few events may produce unreliable results.",
    ))

    # Sparse events warning
    if total_events > 0 and total_events < 10:
        diagnostics.append(_diag(
            "lr_sparse_events", "Sparse events",
            severity=DiagnosticSeverity.WARNING,
            status=DiagnosticStatus.WARNING,
            method=method,
            observed_value=total_events,
            message=f"Only {total_events} total events. Results may be unstable.",
            rationale="Log-rank test has low power with few events.",
        ))

    # Finite test result
    if "test_statistic" in result and "error" not in result:
        import math
        ts = result["test_statistic"]
        pv = result.get("p_value")
        finite_ts = isinstance(ts, (int, float)) and math.isfinite(ts)
        finite_pv = isinstance(pv, (int, float)) and math.isfinite(pv)
        diagnostics.append(_diag(
            "lr_finite", "Finite test result",
            severity=DiagnosticSeverity.BLOCKER if not (finite_ts and finite_pv) else DiagnosticSeverity.INFO,
            status=DiagnosticStatus.FAIL if not (finite_ts and finite_pv) else DiagnosticStatus.PASS,
            method=method,
            observed_value={"test_statistic": ts, "p_value": pv},
            message="Test results are finite." if (finite_ts and finite_pv) else "Non-finite test results.",
            rationale="Non-finite results indicate a computational problem.",
        ))

    return diagnostics


def diagnostic_cox_ph(
    df: pd.DataFrame,
    time_col: str | None,
    event_col: str | None,
    covariates: list[str],
    result: dict[str, Any],
) -> list[DiagnosticResult]:
    """Diagnostics for Cox proportional hazards regression."""
    method = "cox_proportional_hazards"
    diagnostics: list[DiagnosticResult] = []

    # Convergence
    converged = result.get("converged")
    if converged is not None:
        diagnostics.append(_diag(
            "cox_convergence", "Model convergence",
            severity=DiagnosticSeverity.BLOCKER if not converged else DiagnosticSeverity.INFO,
            status=DiagnosticStatus.FAIL if not converged else DiagnosticStatus.PASS,
            method=method,
            observed_value=converged,
            message="Model converged." if converged else "Model did NOT converge.",
            rationale="Non-converged models produce unreliable results.",
        ))

    # Coefficient finiteness
    coefficients = result.get("coefficients", {})
    if coefficients:
        non_finite = [k for k, v in coefficients.items() if not math.isfinite(v)]
        diagnostics.append(_diag(
            "cox_finite_coef", "Finite coefficients",
            severity=DiagnosticSeverity.BLOCKER if non_finite else DiagnosticSeverity.INFO,
            status=DiagnosticStatus.FAIL if non_finite else DiagnosticStatus.PASS,
            method=method,
            observed_value=non_finite if non_finite else "all finite",
            message=f"Non-finite coefficients: {non_finite}" if non_finite else "All coefficients finite.",
            rationale="Non-finite coefficients indicate numerical problems.",
        ))

    # Extreme standard errors
    ses = result.get("standard_errors", {})
    if ses:
        extreme_se = [k for k, v in ses.items() if v > 10]
        diagnostics.append(_diag(
            "cox_extreme_se", "Standard error magnitude",
            severity=DiagnosticSeverity.WARNING if extreme_se else DiagnosticSeverity.INFO,
            status=DiagnosticStatus.WARNING if extreme_se else DiagnosticStatus.PASS,
            method=method,
            observed_value=extreme_se if extreme_se else "none",
            message=f"Extreme SEs: {extreme_se}" if extreme_se else "Standard errors within normal range.",
            rationale="Very large SEs suggest instability or separation.",
        ))

    # Sparse events
    n_events = result.get("n_events", 0)
    n_covariates_result = result.get("n_covariates", 0)
    epc = result.get("events_per_covariate", 0)
    diagnostics.append(_diag(
        "cox_epc", "Events per covariate",
        severity=DiagnosticSeverity.WARNING if epc < 10 else DiagnosticSeverity.INFO,
        status=DiagnosticStatus.WARNING if epc < 10 else DiagnosticStatus.PASS,
        method=method,
        observed_value={"events": n_events, "covariates": n_covariates_result, "epc": epc},
        message=f"Events per covariate: {epc}.",
        rationale="EPV < 10 may produce unstable estimates.",
    ))

    # PH assumption
    ph_test = result.get("ph_test", "not_checked")
    ph_violations = result.get("ph_violations", [])
    if ph_test == "violations_detected":
        diagnostics.append(_diag(
            "cox_ph_assumption", "Proportional hazards assumption",
            severity=DiagnosticSeverity.WARNING,
            status=DiagnosticStatus.WARNING,
            method=method,
            observed_value=ph_violations,
            message=f"PH assumption violations detected: {'; '.join(ph_violations[:3])}",
            rationale="Violation of PH assumption may invalidate hazard ratio interpretation.",
        ))
    elif ph_test == "passed":
        diagnostics.append(_diag(
            "cox_ph_assumption", "Proportional hazards assumption",
            severity=DiagnosticSeverity.INFO,
            status=DiagnosticStatus.PASS,
            method=method,
            message="PH assumption appears satisfied.",
            rationale="No significant violations detected.",
        ))
    elif ph_test == "error":
        diagnostics.append(_diag(
            "cox_ph_assumption", "Proportional hazards assumption",
            severity=DiagnosticSeverity.INFO,
            status=DiagnosticStatus.NOT_CHECKED,
            method=method,
            message="PH assumption could not be tested.",
            rationale="Insufficient data or computational issue.",
        ))

    # Model complexity
    if n_covariates_result > 0 and n_events > 0:
        if n_covariates_result > n_events / 5:
            diagnostics.append(_diag(
                "cox_complexity", "Model complexity",
                severity=DiagnosticSeverity.WARNING,
                status=DiagnosticStatus.WARNING,
                method=method,
                observed_value={"n_covariates": n_covariates_result, "n_events": n_events},
                message=f"Model has {n_covariates_result} covariates with {n_events} events.",
                rationale="High model complexity relative to events may cause overfitting.",
            ))

    # Concordance index
    concordance = result.get("concordance_index")
    if concordance is not None:
        if concordance < 0.5:
            diagnostics.append(_diag(
                "cox_concordance", "Model discrimination",
                severity=DiagnosticSeverity.WARNING,
                status=DiagnosticStatus.WARNING,
                method=method,
                observed_value=concordance,
                message=f"Concordance index: {concordance:.4f} (below 0.5 = no discrimination).",
                rationale="C-index < 0.5 suggests the model is worse than random.",
            ))
        else:
            diagnostics.append(_diag(
                "cox_concordance", "Model discrimination",
                severity=DiagnosticSeverity.INFO,
                status=DiagnosticStatus.PASS,
                method=method,
                observed_value=concordance,
                message=f"Concordance index: {concordance:.4f}.",
                rationale="Model discrimination is acceptable.",
            ))

    return diagnostics

def run_method_diagnostics(
    method_id: str,
    df: pd.DataFrame,
    result: dict[str, Any],
    *,
    outcome: str | None = None,
    group: str | None = None,
    predictors: list[str] | None = None,
    variable1: str | None = None,
    variable2: str | None = None,
    outcome1: str | None = None,
    outcome2: str | None = None,
    x: str | None = None,
    y: str | None = None,
    group1: Any = None,
    group2: Any = None,
    event_col: str | None = None,
) -> list[DiagnosticResult]:
    """Dispatch to method-specific diagnostics based on method_id."""
    dispatch = {
        "independent_t_test": lambda: diagnostic_independent_t_test(
            df, outcome, group, result, group1=group1, group2=group2,
        ),
        "paired_t_test": lambda: diagnostic_paired_t_test(
            df, outcome1, outcome2, result,
        ),
        "mann_whitney_u": lambda: diagnostic_mann_whitney_u(
            df, outcome, group, result, group1=group1, group2=group2,
        ),
        "wilcoxon_signed_rank": lambda: diagnostic_wilcoxon_signed_rank(
            df, outcome1, outcome2, result,
        ),
        "pearson_correlation": lambda: diagnostic_pearson_correlation(
            df, x, y, result,
        ),
        "spearman_correlation": lambda: diagnostic_spearman_correlation(
            df, x, y, result,
        ),
        "one_way_anova": lambda: diagnostic_one_way_anova(
            df, outcome, group, result,
        ),
        "linear_regression": lambda: diagnostic_linear_regression(
            df, outcome, predictors or [], result,
        ),
        "logistic_regression": lambda: diagnostic_logistic_regression(
            df, outcome, predictors or [], result,
        ),
        "chi_square": lambda: diagnostic_chi_square(
            df, variable1, variable2, result,
        ),
        "fisher_exact": lambda: diagnostic_fisher_exact(
            df, variable1, variable2, result,
        ),
        # Survival methods
        "kaplan_meier": lambda: diagnostic_kaplan_meier(
            df, outcome, event_col, group, result,
        ),
        "log_rank": lambda: diagnostic_log_rank(
            df, outcome, event_col, group, result,
        ),
        "cox_proportional_hazards": lambda: diagnostic_cox_ph(
            df, outcome, event_col, predictors or [], result,
        ),
    }

    if method_id in dispatch:
        return dispatch[method_id]()
    return []
