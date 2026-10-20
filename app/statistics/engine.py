"""Statistical Engine — deterministic computation layer.

All numerical results MUST come from this module, never from the LLM.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _ci_95_mean(series: pd.Series) -> tuple[float, float]:
    """95% CI for the mean using t-distribution."""
    n = series.count()
    if n < 2:
        return (float(series.mean()), float(series.mean()))
    se = series.std(ddof=1) / np.sqrt(n)
    t_crit = sp_stats.t.ppf(0.975, df=n - 1)
    mean = float(series.mean())
    return (mean - t_crit * se, mean + t_crit * se)


# ---------------------------------------------------------------------------
# Descriptive Statistics
# ---------------------------------------------------------------------------

def descriptive_statistics(
    file_path: Path,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Compute descriptive statistics for specified or all numeric columns."""
    df = _load_dataframe(file_path)
    if columns:
        df = df[[c for c in columns if c in df.columns]]

    results = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            n = len(s)
            if n == 0:
                results[col] = {"n": 0, "warning": "All values missing"}
                continue

            ci = _ci_95_mean(s)
            results[col] = {
                "n": int(n),
                "n_missing": int(df[col].isna().sum()),
                "mean": round(float(s.mean()), 6),
                "std": round(float(s.std(ddof=1), ), 6) if n > 1 else None,
                "se": round(float(s.std(ddof=1) / np.sqrt(n)), 6) if n > 1 else None,
                "ci_95_lower": round(ci[0], 6),
                "ci_95_upper": round(ci[1], 6),
                "min": round(float(s.min()), 6),
                "q25": round(float(s.quantile(0.25)), 6),
                "median": round(float(s.median()), 6),
                "q75": round(float(s.quantile(0.75)), 6),
                "max": round(float(s.max()), 6),
                "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 6),
                "skewness": round(float(s.skew()), 6) if n > 2 else None,
                "kurtosis": round(float(s.kurtosis()), 6) if n > 3 else None,
                " Shapiro_W": None,
                "shapiro_p": None,
            }
            if 3 <= n <= 5000:
                try:
                    stat, p = sp_stats.shapiro(s.values)
                    results[col]["shapiro_W"] = round(float(stat), 6)
                    results[col]["shapiro_p"] = round(float(p), 6)
                except Exception:
                    pass
        else:
            # Categorical
            vc = df[col].value_counts()
            n = int(df[col].count())
            results[col] = {
                "n": n,
                "n_missing": int(df[col].isna().sum()),
                "n_unique": int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in vc.head(10).items()},
            }

    return {
        "columns": list(results.keys()),
        "statistics": results,
        "n_rows": len(df),
        "engine": "scipy+numpy+pandas",
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# t-Tests
# ---------------------------------------------------------------------------

def independent_t_test(
    file_path: Path,
    outcome: str,
    group: str,
    group1: str | None = None,
    group2: str | None = None,
    equal_var: bool = False,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Welch's (or Student's) independent-samples t-test."""
    df = _load_dataframe(file_path)

    if group1 is not None and group2 is not None:
        df = df[df[group].isin([group1, group2])]

    groups = df[group].dropna().unique()
    if len(groups) < 2:
        return {"error": "Need at least two groups."}

    g1_name = group1 if group1 is not None else groups[0]
    g2_name = group2 if group2 is not None else groups[1]

    g1 = df.loc[df[group] == g1_name, outcome].dropna()
    g2 = df.loc[df[group] == g2_name, outcome].dropna()

    if len(g1) < 2 or len(g2) < 2:
        return {"error": "Each group must have at least 2 observations."}

    stat, p = sp_stats.ttest_ind(g1, g2, equal_var=equal_var, alternative=alternative)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(g1) - 1) * g1.std(ddof=1) ** 2 + (len(g2) - 1) * g2.std(ddof=1) ** 2)
        / (len(g1) + len(g2) - 2)
    )
    cohens_d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0

    # 95% CI for difference in means
    diff = g1.mean() - g2.mean()
    se_diff = np.sqrt(g1.var(ddof=1) / len(g1) + g2.var(ddof=1) / len(g2))
    df_welch = (
        (g1.var(ddof=1) / len(g1) + g2.var(ddof=1) / len(g2)) ** 2
        / (
            (g1.var(ddof=1) / len(g1)) ** 2 / (len(g1) - 1)
            + (g2.var(ddof=1) / len(g2)) ** 2 / (len(g2) - 1)
        )
    )
    t_crit = sp_stats.t.ppf(0.975, df=df_welch)

    return {
        "test": "Independent t-test" if equal_var else "Welch's t-test",
        "group1": g1_name,
        "group2": g2_name,
        "n_group1": len(g1),
        "n_group2": len(g2),
        "mean_group1": round(float(g1.mean()), 6),
        "mean_group2": round(float(g2.mean()), 6),
        "std_group1": round(float(g1.std(ddof=1)), 6),
        "std_group2": round(float(g2.std(ddof=1)), 6),
        "difference": round(float(diff), 6),
        "se_difference": round(float(se_diff), 6),
        "ci_95_lower": round(float(diff - t_crit * se_diff), 6),
        "ci_95_upper": round(float(diff + t_crit * se_diff), 6),
        "t_statistic": round(float(stat), 6),
        "df": round(float(df_welch), 4),
        "p_value": round(float(p), 8),
        "cohens_d": round(float(cohens_d), 6),
        "alternative": alternative,
        "equal_variances_assumed": equal_var,
        "engine": "scipy.stats",
    }


def paired_t_test(
    file_path: Path,
    outcome1: str,
    outcome2: str,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Paired-samples t-test."""
    df = _load_dataframe(file_path)
    d = df[outcome1].dropna() - df[outcome2].dropna()
    n = len(d)

    if n < 2:
        return {"error": "Need at least 2 paired observations."}

    stat, p = sp_stats.ttest_1samp(d, 0, alternative=alternative)
    se = d.std(ddof=1) / np.sqrt(n)
    t_crit = sp_stats.t.ppf(0.975, df=n - 1)
    mean_d = float(d.mean())

    return {
        "test": "Paired t-test",
        "outcome1": outcome1,
        "outcome2": outcome2,
        "n_pairs": n,
        "mean_difference": round(mean_d, 6),
        "std_difference": round(float(d.std(ddof=1)), 6),
        "se_difference": round(float(se), 6),
        "ci_95_lower": round(mean_d - t_crit * se, 6),
        "ci_95_upper": round(mean_d + t_crit * se, 6),
        "t_statistic": round(float(stat), 6),
        "df": n - 1,
        "p_value": round(float(p), 8),
        "alternative": alternative,
        "engine": "scipy.stats",
    }


# ---------------------------------------------------------------------------
# Chi-Square / Fisher Exact
# ---------------------------------------------------------------------------

def chi_square_test(
    file_path: Path,
    variable1: str,
    variable2: str,
    use_fisher: bool = False,
) -> dict[str, Any]:
    """Chi-square test of independence or Fisher's exact test."""
    df = _load_dataframe(file_path)
    ct = pd.crosstab(df[variable1], df[variable2])

    if use_fisher and ct.shape == (2, 2):
        odds_ratio, p = sp_stats.fisher_exact(ct.values)
        return {
            "test": "Fisher's exact test",
            "variable1": variable1,
            "variable2": variable2,
            "table": ct.to_dict(),
            "odds_ratio": round(float(odds_ratio), 6),
            "p_value": round(float(p), 8),
            "engine": "scipy.stats",
        }

    chi2, p, dof, expected = sp_stats.chi2_contingency(ct.values, correction=False)

    # Check expected cell counts
    n_low_expected = int((expected < 5).sum())
    total_cells = expected.size

    warnings = []
    if n_low_expected > 0:
        pct = round(n_low_expected / total_cells * 100, 1)
        warnings.append(
            f"{n_low_expected} cells ({pct}%) have expected count < 5. "
            "Chi-square may not be reliable. Consider Fisher's exact test."
        )

    return {
        "test": "Chi-square test of independence",
        "variable1": variable1,
        "variable2": variable2,
        "table": ct.to_dict(),
        "chi2": round(float(chi2), 6),
        "df": int(dof),
        "p_value": round(float(p), 8),
        "n_cells_low_expected": n_low_expected,
        "warnings": warnings,
        "engine": "scipy.stats",
    }


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def correlation_test(
    file_path: Path,
    x: str,
    y: str,
    method: str = "pearson",
) -> dict[str, Any]:
    """Pearson or Spearman correlation with confidence interval."""
    df = _load_dataframe(file_path)
    mask = df[x].notna() & df[y].notna()
    x_vals = df.loc[mask, x].values
    y_vals = df.loc[mask, y].values

    n = len(x_vals)
    if n < 3:
        return {"error": "Need at least 3 paired observations."}

    if method == "pearson":
        r, p = sp_stats.pearsonr(x_vals, y_vals)
        # Fisher z-transformation for CI
        z = np.arctanh(r)
        se_z = 1 / np.sqrt(n - 3)
        ci_lower = np.tanh(z - 1.96 * se_z)
        ci_upper = np.tanh(z + 1.96 * se_z)
    elif method == "spearman":
        r, p = sp_stats.spearmanr(x_vals, y_vals)
        # Approximate CI using Fisher z
        z = np.arctanh(r)
        se_z = 1 / np.sqrt(n - 3)
        ci_lower = np.tanh(z - 1.96 * se_z)
        ci_upper = np.tanh(z + 1.96 * se_z)
    else:
        return {"error": f"Unknown method: {method}. Use 'pearson' or 'spearman'."}

    # R-squared
    r_squared = float(r) ** 2

    return {
        "test": f"{method.title()} correlation",
        "x": x,
        "y": y,
        "n": n,
        "r": round(float(r), 6),
        "r_squared": round(r_squared, 6),
        "p_value": round(float(p), 8),
        "ci_95_lower": round(float(ci_lower), 6),
        "ci_95_upper": round(float(ci_upper), 6),
        "engine": "scipy.stats",
    }


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------

def one_way_anova(
    file_path: Path,
    outcome: str,
    group: str,
) -> dict[str, Any]:
    """One-way ANOVA with Tukey HSD post-hoc."""
    df = _load_dataframe(file_path)
    groups = df.groupby(group)[outcome].apply(lambda x: x.dropna().values, include_groups=False)
    group_names = list(groups.index)

    if len(group_names) < 2:
        return {"error": "Need at least 2 groups."}

    arrays = [v for v in groups.values if len(v) >= 1]
    names = [n for n, v in zip(group_names, groups.values) if len(v) >= 1]

    if len(arrays) < 2:
        return {"error": "Need at least 2 groups with data."}

    f_stat, p = sp_stats.f_oneway(*arrays)

    # Descriptive stats per group
    group_stats = {}
    for name, arr in zip(names, arrays):
        s = pd.Series(arr)
        group_stats[str(name)] = {
            "n": int(len(s)),
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std(ddof=1)), 6) if len(s) > 1 else 0,
        }

    # Tukey HSD post-hoc (if more than 2 groups and overall p < 0.05)
    tukey_results = None
    if len(arrays) > 2 and p < 0.05:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        tukey = pairwise_tukeyhsd(df[outcome].dropna(), df[group].dropna(), alpha=0.05)
        tukey_results = [
            {
                "group1": str(row[0]),
                "group2": str(row[1]),
                "meandiff": round(float(row[2]), 6),
                "p_adj": round(float(row[3]), 8),
                "lower": round(float(row[4]), 6),
                "upper": round(float(row[5]), 6),
                "reject": bool(row[6]),
            }
            for row in tukey._results_table.data[1:]
        ]

    return {
        "test": "One-way ANOVA",
        "outcome": outcome,
        "grouping_variable": group,
        "n_groups": len(arrays),
        "f_statistic": round(float(f_stat), 6),
        "p_value": round(float(p), 8),
        "group_statistics": group_stats,
        "tukey_hsd": tukey_results,
        "engine": "scipy.stats + statsmodels",
    }


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def logistic_regression(
    file_path: Path,
    outcome: str,
    predictors: list[str],
    covariates: list[str] | None = None,
) -> dict[str, Any]:
    """Logistic regression using statsmodels."""
    import statsmodels.api as sm

    df = _load_dataframe(file_path)
    all_vars = predictors + (covariates or [])
    df_model = df[[outcome] + all_vars].dropna()

    if len(df_model) < len(all_vars) + 5:
        return {"error": "Insufficient observations for logistic regression."}

    y = df_model[outcome].values
    X = df_model[all_vars].values

    # Convert categorical to dummies
    X_df = pd.get_dummies(df_model[all_vars], drop_first=True)
    X = X_df.values.astype(float)
    feature_names = list(X_df.columns)

    X_with_const = sm.add_constant(X)

    try:
        model = sm.Logit(y, X_with_const).fit(disp=False, maxiter=100)
    except Exception as e:
        return {"error": f"Model did not converge: {e}"}

    coef_names = ["const"] + feature_names
    params = {}
    for i, name in enumerate(coef_names):
        params[name] = {
            "coef": round(float(model.params[i]), 6),
            "std_err": round(float(model.bse[i]), 6),
            "z": round(float(model.tvalues[i]), 6),
            "p_value": round(float(model.pvalues[i]), 8),
            "ci_95_lower": round(float(model.conf_int()[i][0]), 6),
            "ci_95_upper": round(float(model.conf_int()[i][1]), 6),
            "odds_ratio": round(float(np.exp(model.params[i])), 6),
            "or_ci_lower": round(float(np.exp(model.conf_int()[i][0])), 6),
            "or_ci_upper": round(float(np.exp(model.conf_int()[i][1])), 6),
        }

    return {
        "test": "Logistic regression",
        "outcome": outcome,
        "predictors": feature_names,
        "n_obs": int(model.nobs),
        "pseudo_r2": round(float(model.prsquared), 6),
        "llf": round(float(model.llf), 4),
        "llnull": round(float(model.llnull), 4),
        "aic": round(float(model.aic), 4),
        "bic": round(float(model.bic), 4),
        "log_likelihood_ratio_p": round(float(model.llr_pvalue), 8),
        "parameters": params,
        "converged": bool(model.mle_retvals.get("converged", True)),
        "engine": "statsmodels",
    }


# ---------------------------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------------------------

def linear_regression(
    file_path: Path,
    outcome: str,
    predictors: list[str],
    covariates: list[str] | None = None,
) -> dict[str, Any]:
    """OLS linear regression."""
    import statsmodels.api as sm

    df = _load_dataframe(file_path)
    all_vars = predictors + (covariates or [])
    df_model = df[[outcome] + all_vars].dropna()

    if len(df_model) < len(all_vars) + 2:
        return {"error": "Insufficient observations for linear regression."}

    y = df_model[outcome].values.astype(float)
    X_df = pd.get_dummies(df_model[all_vars], drop_first=True)
    X = X_df.values.astype(float)
    feature_names = list(X_df.columns)

    X_with_const = sm.add_constant(X)

    try:
        model = sm.OLS(y, X_with_const).fit()
    except Exception as e:
        return {"error": f"Model fitting failed: {e}"}

    coef_names = ["const"] + feature_names
    params = {}
    for i, name in enumerate(coef_names):
        params[name] = {
            "coef": round(float(model.params[i]), 6),
            "std_err": round(float(model.bse[i]), 6),
            "t": round(float(model.tvalues[i]), 6),
            "p_value": round(float(model.pvalues[i]), 8),
            "ci_95_lower": round(float(model.conf_int()[i][0]), 6),
            "ci_95_upper": round(float(model.conf_int()[i][1]), 6),
        }

    return {
        "test": "OLS Linear regression",
        "outcome": outcome,
        "predictors": feature_names,
        "n_obs": int(model.nobs),
        "r_squared": round(float(model.rsquared), 6),
        "adj_r_squared": round(float(model.rsquared_adj), 6),
        "f_statistic": round(float(model.fvalue), 6),
        "f_p_value": round(float(model.f_pvalue), 8),
        "aic": round(float(model.aic), 4),
        "bic": round(float(model.bic), 4),
        "parameters": params,
        "engine": "statsmodels",
    }


# ---------------------------------------------------------------------------
# Mann-Whitney U
# ---------------------------------------------------------------------------

def mann_whitney_u(
    file_path: Path,
    outcome: str,
    group: str,
    group1: str | None = None,
    group2: str | None = None,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Mann-Whitney U test (nonparametric independent groups)."""
    df = _load_dataframe(file_path)

    if group1 is not None and group2 is not None:
        df = df[df[group].isin([group1, group2])]

    groups = df[group].dropna().unique()
    if len(groups) < 2:
        return {"error": "Need at least two groups."}

    g1_name = group1 if group1 is not None else groups[0]
    g2_name = group2 if group2 is not None else groups[1]

    g1 = df.loc[df[group] == g1_name, outcome].dropna()
    g2 = df.loc[df[group] == g2_name, outcome].dropna()

    if len(g1) < 1 or len(g2) < 1:
        return {"error": "Each group must have at least 1 observation."}

    stat, p = sp_stats.mannwhitneyu(g1, g2, alternative=alternative)

    return {
        "test": "Mann-Whitney U",
        "group1": g1_name,
        "group2": g2_name,
        "n_group1": len(g1),
        "n_group2": len(g2),
        "median_group1": round(float(g1.median()), 6),
        "median_group2": round(float(g2.median()), 6),
        "u_statistic": round(float(stat), 6),
        "p_value": round(float(p), 8),
        "alternative": alternative,
        "engine": "scipy.stats",
    }


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank(
    file_path: Path,
    outcome1: str,
    outcome2: str,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Wilcoxon signed-rank test for paired nonparametric data."""
    df = _load_dataframe(file_path)
    d = df[outcome1].dropna() - df[outcome2].dropna()
    d = d[d != 0]  # Wilcoxon excludes zero differences

    n = len(d)
    if n < 5:
        return {"error": "Need at least 5 non-zero paired differences."}

    stat, p = sp_stats.wilcoxon(d, alternative=alternative)

    return {
        "test": "Wilcoxon signed-rank",
        "outcome1": outcome1,
        "outcome2": outcome2,
        "n_pairs": n,
        "median_difference": round(float(d.median()), 6),
        "w_statistic": round(float(stat), 6),
        "p_value": round(float(p), 8),
        "alternative": alternative,
        "engine": "scipy.stats",
    }


# ---------------------------------------------------------------------------
# Fisher's exact
# ---------------------------------------------------------------------------

def fisher_exact_test(
    file_path: Path,
    variable1: str,
    variable2: str,
) -> dict[str, Any]:
    """Fisher's exact test for 2x2 tables."""
    return chi_square_test(file_path, variable1, variable2, use_fisher=True)


# ---------------------------------------------------------------------------
# Power / Sample Size
# ---------------------------------------------------------------------------

def power_analysis_t_test(
    effect_size: float | None = None,
    n_per_group: int | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Compute power, sample size, or minimum detectable effect for a two-sample t-test."""
    from statsmodels.stats.power import TTestIndPower

    analysis = TTestIndPower()

    if effect_size is not None and n_per_group is not None:
        # Compute power
        pwr = analysis.solve_power(
            effect_size=effect_size,
            nobs1=n_per_group,
            alpha=alpha,
            power=None,
            alternative=alternative,
        )
        return {
            "analysis": "Two-sample t-test power",
            "effect_size": effect_size,
            "n_per_group": n_per_group,
            "alpha": alpha,
            "power": round(float(pwr), 6),
            "engine": "statsmodels",
        }
    elif effect_size is not None and power is not None:
        # Compute sample size
        n = analysis.solve_power(
            effect_size=effect_size,
            nobs1=None,
            alpha=alpha,
            power=power,
            alternative=alternative,
        )
        n_ceil = int(np.ceil(n))
        return {
            "analysis": "Two-sample t-test sample size",
            "effect_size": effect_size,
            "required_n_per_group": n_ceil,
            "total_required_n": n_ceil * 2,
            "alpha": alpha,
            "power": power,
            "engine": "statsmodels",
        }
    elif n_per_group is not None and power is not None:
        # Compute minimum detectable effect
        es = analysis.solve_power(
            effect_size=None,
            nobs1=n_per_group,
            alpha=alpha,
            power=power,
            alternative=alternative,
        )
        return {
            "analysis": "Two-sample t-test minimum detectable effect",
            "n_per_group": n_per_group,
            "alpha": alpha,
            "power": power,
            "min_detectable_effect_size": round(float(es), 6),
            "engine": "statsmodels",
        }
    else:
        return {"error": "Provide at least two of: effect_size, n_per_group, power."}
