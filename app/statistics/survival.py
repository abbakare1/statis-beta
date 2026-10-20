"""Survival Analysis Engine — deterministic computation for time-to-event data.

All numerical results come from established deterministic libraries (lifelines, scipy).
No LLM is involved in any computation.

Methods implemented:
- Kaplan-Meier estimator
- Log-rank test (two-group comparison)
- Cox proportional hazards regression
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy import stats as sp_stats


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


# ---------------------------------------------------------------------------
# Survival Data Validation
# ---------------------------------------------------------------------------

def validate_survival_data(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str | None = None,
    covariates: list[str] | None = None,
) -> dict[str, Any]:
    """Validate survival dataset structure and content.

    Returns dict with:
    - valid: bool
    - n_total: int
    - n_complete: int
    - n_events: int
    - n_censored: int
    - event_proportion: float
    - missing_time: int
    - missing_event: int
    - invalid_time: int (negative or non-finite)
    - invalid_event: int (not 0/1)
    - blockers: list[str]
    - warnings: list[str]
    """
    result: dict[str, Any] = {
        "valid": True,
        "n_total": len(df),
        "n_complete": 0,
        "n_events": 0,
        "n_censored": 0,
        "event_proportion": 0.0,
        "missing_time": 0,
        "missing_event": 0,
        "invalid_time": 0,
        "invalid_event": 0,
        "blockers": [],
        "warnings": [],
    }

    # Check columns exist
    if time_col not in df.columns:
        result["blockers"].append(f"Time variable '{time_col}' not found in dataset.")
        result["valid"] = False
        return result

    if event_col not in df.columns:
        result["blockers"].append(f"Event variable '{event_col}' not found in dataset.")
        result["valid"] = False
        return result

    time = df[time_col]
    event = df[event_col]

    # Missing time
    result["missing_time"] = int(time.isna().sum())

    # Missing event
    result["missing_event"] = int(event.isna().sum())

    # Invalid time (negative or non-finite, excluding NaN which is "missing")
    if pd.api.types.is_numeric_dtype(time):
        valid_mask = time.notna()
        finite_mask = np.isfinite(time.values.astype(float))
        negative_mask = (time.values.astype(float) < 0)
        # Invalid = not missing AND (non-finite OR negative)
        invalid = valid_mask & (~finite_mask | negative_mask)
        result["invalid_time"] = int(invalid.sum())
    else:
        # Non-numeric time is invalid (except NaN which is missing)
        result["invalid_time"] = int(time.notna().sum())

    # Invalid event values (must be 0 or 1 after dropping NaN)
    event_valid = event.dropna()
    if pd.api.types.is_numeric_dtype(event):
        is_binary = event_valid.isin([0, 1])
    elif pd.api.types.is_string_dtype(event) or pd.api.types.is_object_dtype(event):
        lower = event_valid.astype(str).str.strip().str.lower()
        mapping = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0,
                    "event": 1, "censored": 0, "alive": 0, "dead": 1, "died": 1}
        mapped = lower.map(mapping)
        is_binary = mapped.notna()
    else:
        is_binary = pd.Series([False] * len(event_valid), index=event_valid.index)

    result["invalid_event"] = int((~is_binary).sum())

    # Complete cases (valid time + valid event)
    complete_time = time.notna() & np.isfinite(time.values.astype(float)) if pd.api.types.is_numeric_dtype(time) else time.notna()
    complete_event = event.notna() & is_binary if 'is_binary' in dir() else event.notna()
    complete_mask = complete_time & complete_event
    result["n_complete"] = int(complete_mask.sum())

    # Events and censored among complete cases
    if result["n_complete"] > 0:
        event_vals = event[complete_mask]
        if pd.api.types.is_numeric_dtype(event_vals):
            result["n_events"] = int((event_vals == 1).sum())
            result["n_censored"] = int((event_vals == 0).sum())
        elif pd.api.types.is_string_dtype(event_vals) or pd.api.types.is_object_dtype(event_vals):
            lower = event_vals.astype(str).str.strip().str.lower()
            mapping = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0,
                        "event": 1, "censored": 0, "alive": 0, "dead": 1, "died": 1}
            mapped = lower.map(mapping)
            result["n_events"] = int((mapped == 1).sum())
            result["n_censored"] = int((mapped == 0).sum())

        if result["n_complete"] > 0:
            result["event_proportion"] = result["n_events"] / result["n_complete"]

    # Validate group if provided
    if group_col and group_col in df.columns:
        n_groups = df[group_col].nunique()
        if n_groups < 2:
            result["warnings"].append(
                f"Grouping variable '{group_col}' has only {n_groups} unique value(s)."
            )
    elif group_col and group_col not in df.columns:
        result["blockers"].append(f"Group variable '{group_col}' not found in dataset.")
        result["valid"] = False

    # Validate covariates if provided
    if covariates:
        for cov in covariates:
            if cov not in df.columns:
                result["blockers"].append(f"Covariate '{cov}' not found in dataset.")
                result["valid"] = False

    # Check for zero events
    if result["n_events"] == 0 and result["n_complete"] > 0:
        result["warnings"].append("No events observed. Survival analysis may have limited value.")

    # Check for all events
    if result["n_censored"] == 0 and result["n_complete"] > 0 and result["n_events"] > 0:
        result["warnings"].append("All observations are events. No censoring observed.")

    # Check for sparse events (less than 5)
    if 0 < result["n_events"] < 5:
        result["warnings"].append(
            f"Very few events ({result['n_events']}). Results may be unreliable."
        )

    # Check for minimum observations
    if result["n_complete"] < 10:
        result["warnings"].append(
            f"Small complete-case sample size ({result['n_complete']}). "
            "Results should be interpreted with caution."
        )

    # Block if no complete cases
    if result["n_complete"] == 0:
        result["blockers"].append("No complete survival records found.")
        result["valid"] = False

    # Block if all events and no censoring for KM (not a blocker per se, but a concern)
    # Actually, KM works fine with all events.

    # Block if negative times found
    if result["invalid_time"] > 0:
        result["blockers"].append(
            f"{result['invalid_time']} observation(s) have invalid (negative or non-finite) survival times."
        )
        result["valid"] = False

    # Block if invalid event indicators
    if result["invalid_event"] > 0:
        result["blockers"].append(
            f"{result['invalid_event']} observation(s) have invalid event indicators "
            "(must be 0/1 or equivalent)."
        )
        result["valid"] = False

    return result


# ---------------------------------------------------------------------------
# Kaplan-Meier Estimator
# ---------------------------------------------------------------------------

def kaplan_meier(
    file_path: Path,
    time_col: str,
    event_col: str,
    group_col: str | None = None,
    ci_alpha: float = 0.05,
) -> dict[str, Any]:
    """Kaplan-Meier survival estimation.

    Returns a dict with survival estimates, confidence intervals,
    risk table data, and summary statistics.
    """
    df = _load_dataframe(file_path)

    # Validate
    validation = validate_survival_data(df, time_col, event_col, group_col)
    if not validation["valid"]:
        return {"error": "; ".join(validation["blockers"]), "validation": validation}

    # Drop invalid/missing rows
    time = df[time_col].copy()
    event_raw = df[event_col].copy()

    # Convert event to numeric
    if pd.api.types.is_string_dtype(event_raw) or pd.api.types.is_object_dtype(event_raw):
        lower = event_raw.astype(str).str.strip().str.lower()
        mapping = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0,
                    "event": 1, "censored": 0, "alive": 0, "dead": 1, "died": 1}
        event_num = lower.map(mapping).astype(float)
    else:
        event_num = event_raw.astype(float)

    # Filter to complete cases
    mask = time.notna() & np.isfinite(time) & event_num.notna() & (time >= 0)
    time = time[mask]
    event_num = event_num[mask]
    group = df.loc[mask, group_col] if group_col and group_col in df.columns else None

    result: dict[str, Any] = {
        "validation": validation,
        "n_total": len(df),
        "n_complete": len(time),
        "n_events": int(event_num.sum()),
        "n_censored": int((event_num == 0).sum()),
        "grouped": group_col is not None,
        "groups": {},
    }

    if group_col and group is not None:
        groups = sorted(group.unique())
        result["n_groups"] = len(groups)

        for g in groups:
            g_mask = group == g
            t_g = time[g_mask]
            e_g = event_num[g_mask]

            kmf = KaplanMeierFitter()
            kmf.fit(t_g, event_observed=e_g, label=str(g))

            # Extract survival function
            sf = kmf.survival_function_
            times = sf.index.tolist()
            surv_probs = sf.iloc[:, 0].tolist()

            # Confidence intervals
            ci = kmf.confidence_interval_survival_function_
            ci_lower = ci.iloc[:, 0].tolist() if ci is not None else None
            ci_upper = ci.iloc[:, 1].tolist() if ci is not None else None

            # Median survival
            median_surv = kmf.median_survival_time_
            if np.isinf(median_surv) or np.isnan(median_surv):
                median_surv = None
                median_surv_ci = None
            else:
                try:
                    median_surv_ci = kmf.confidence_interval_median_survival_time_
                    if median_surv_ci is not None:
                        median_surv_ci = (
                            float(median_surv_ci.iloc[0, 0]) if not np.isinf(median_surv_ci.iloc[0, 0]) else None,
                            float(median_surv_ci.iloc[0, 1]) if not np.isinf(median_surv_ci.iloc[0, 1]) else None,
                        )
                except Exception:
                    median_surv_ci = None

            # Risk table
            n_at_risk = []
            n_events_g = []
            n_censored_g = []
            for t_val in times:
                at_risk = int((t_g >= t_val).sum())
                n_at_risk.append(at_risk)
                events_at_t = int(((t_g == t_val) & (e_g == 1)).sum())
                n_events_g.append(events_at_t)
                censored_at_t = int(((t_g == t_val) & (e_g == 0)).sum())
                n_censored_g.append(censored_at_t)

            result["groups"][str(g)] = {
                "n": int(g_mask.sum()),
                "n_events": int(e_g.sum()),
                "n_censored": int((e_g == 0).sum()),
                "times": [round(t, 6) for t in times],
                "survival_probabilities": [round(p, 6) for p in surv_probs],
                "ci_lower": [round(p, 6) for p in ci_lower] if ci_lower else None,
                "ci_upper": [round(p, 6) for p in ci_upper] if ci_upper else None,
                "median_survival": round(median_surv, 6) if median_surv is not None else None,
                "median_survival_ci": median_surv_ci,
                "n_at_risk": n_at_risk,
                "n_events_at_time": n_events_g,
                "n_censored_at_time": n_censored_g,
            }

        result["test_statistic"] = None
        result["p_value"] = None
    else:
        # Ungrouped KM
        kmf = KaplanMeierFitter()
        kmf.fit(time, event_observed=event_num)

        sf = kmf.survival_function_
        times = sf.index.tolist()
        surv_probs = sf.iloc[:, 0].tolist()

        ci = kmf.confidence_interval_survival_function_
        ci_lower = ci.iloc[:, 0].tolist() if ci is not None else None
        ci_upper = ci.iloc[:, 1].tolist() if ci is not None else None

        median_surv = kmf.median_survival_time_
        if np.isinf(median_surv) or np.isnan(median_surv):
            median_surv = None
            median_surv_ci = None
        else:
            try:
                median_surv_ci = kmf.confidence_interval_median_survival_time_
                if median_surv_ci is not None:
                    median_surv_ci = (
                        float(median_surv_ci.iloc[0, 0]) if not np.isinf(median_surv_ci.iloc[0, 0]) else None,
                        float(median_surv_ci.iloc[0, 1]) if not np.isinf(median_surv_ci.iloc[0, 1]) else None,
                    )
            except Exception:
                median_surv_ci = None

        n_at_risk = []
        n_events_g = []
        n_censored_g = []
        for t_val in times:
            at_risk = int((time >= t_val).sum())
            n_at_risk.append(at_risk)
            events_at_t = int(((time == t_val) & (event_num == 1)).sum())
            n_events_g.append(events_at_t)
            censored_at_t = int(((time == t_val) & (event_num == 0)).sum())
            n_censored_g.append(censored_at_t)

        result["n_groups"] = 1
        result["groups"]["overall"] = {
            "n": int(len(time)),
            "n_events": int(event_num.sum()),
            "n_censored": int((event_num == 0).sum()),
            "times": [round(t, 6) for t in times],
            "survival_probabilities": [round(p, 6) for p in surv_probs],
            "ci_lower": [round(p, 6) for p in ci_lower] if ci_lower else None,
            "ci_upper": [round(p, 6) for p in ci_upper] if ci_upper else None,
            "median_survival": round(median_surv, 6) if median_surv is not None else None,
            "median_survival_ci": median_surv_ci,
            "n_at_risk": n_at_risk,
            "n_events_at_time": n_events_g,
            "n_censored_at_time": n_censored_g,
        }

    return result


# ---------------------------------------------------------------------------
# Log-Rank Test
# ---------------------------------------------------------------------------

def log_rank(
    file_path: Path,
    time_col: str,
    event_col: str,
    group_col: str,
    group1: Any = None,
    group2: Any = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Log-rank test for comparing two survival distributions.

    Returns test statistic, p-value, and group-level summaries.
    """
    df = _load_dataframe(file_path)

    # Validate
    validation = validate_survival_data(df, time_col, event_col, group_col)
    if not validation["valid"]:
        return {"error": "; ".join(validation["blockers"]), "validation": validation}

    if group_col not in df.columns:
        return {"error": f"Group variable '{group_col}' not found.", "validation": validation}

    # Prepare data
    time = df[time_col].copy()
    event_raw = df[event_col].copy()

    if pd.api.types.is_string_dtype(event_raw) or pd.api.types.is_object_dtype(event_raw):
        lower = event_raw.astype(str).str.strip().str.lower()
        mapping = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0,
                    "event": 1, "censored": 0, "alive": 0, "dead": 1, "died": 1}
        event_num = lower.map(mapping).astype(float)
    else:
        event_num = event_raw.astype(float)

    mask = time.notna() & np.isfinite(time) & event_num.notna() & (time >= 0)
    work_df = pd.DataFrame({
        "time": time[mask],
        "event": event_num[mask],
        "group": df.loc[mask, group_col],
    })

    # Determine groups
    unique_groups = sorted(work_df["group"].unique())
    if len(unique_groups) < 2:
        return {
            "error": f"Need at least 2 groups, found {len(unique_groups)}.",
            "validation": validation,
        }

    if group1 is not None and group2 is not None:
        # Filter to specified groups
        g1_mask = work_df["group"] == group1
        g2_mask = work_df["group"] == group2
        if g1_mask.sum() == 0 or g2_mask.sum() == 0:
            return {
                "error": f"Specified groups not found: group1={group1}, group2={group2}.",
                "validation": validation,
            }
        t1 = work_df.loc[g1_mask, "time"].values
        e1 = work_df.loc[g1_mask, "event"].values
        t2 = work_df.loc[g2_mask, "time"].values
        e2 = work_df.loc[g2_mask, "event"].values
        actual_group1 = group1
        actual_group2 = group2
    else:
        # Use first two groups
        g1 = unique_groups[0]
        g2 = unique_groups[1]
        t1 = work_df.loc[work_df["group"] == g1, "time"].values
        e1 = work_df.loc[work_df["group"] == g1, "event"].values
        t2 = work_df.loc[work_df["group"] == g2, "time"].values
        e2 = work_df.loc[work_df["group"] == g2, "event"].values
        actual_group1 = g1
        actual_group2 = g2

    # Perform log-rank test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = logrank_test(t1, t2, event_observed_A=e1, event_observed_B=e2)

    result: dict[str, Any] = {
        "test_statistic": round(float(lr.test_statistic), 6),
        "p_value": round(float(lr.p_value), 8),
        "degrees_of_freedom": 1,
        "group1": actual_group1,
        "group2": actual_group2,
        "n_group1": len(t1),
        "n_group2": len(t2),
        "n_events_group1": int(e1.sum()),
        "n_events_group2": int(e2.sum()),
        "n_censored_group1": int((e1 == 0).sum()),
        "n_censored_group2": int((e2 == 0).sum()),
        "validation": validation,
    }

    return result


# ---------------------------------------------------------------------------
# Cox Proportional Hazards Regression
# ---------------------------------------------------------------------------

def cox_proportional_hazards(
    file_path: Path,
    time_col: str,
    event_col: str,
    covariates: list[str],
    group_col: str | None = None,
    alpha: float = 0.05,
    penalizer: float = 0.0,
) -> dict[str, Any]:
    """Cox proportional hazards regression.

    Returns hazard ratios, confidence intervals, p-values, and model diagnostics.
    """
    df = _load_dataframe(file_path)

    # Validate
    validation = validate_survival_data(df, time_col, event_col, group_col, covariates)
    if not validation["valid"]:
        return {"error": "; ".join(validation["blockers"]), "validation": validation}

    # Prepare data
    time = df[time_col].copy()
    event_raw = df[event_col].copy()

    if pd.api.types.is_string_dtype(event_raw) or pd.api.types.is_object_dtype(event_raw):
        lower = event_raw.astype(str).str.strip().str.lower()
        mapping = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0,
                    "event": 1, "censored": 0, "alive": 0, "dead": 1, "died": 1}
        event_num = lower.map(mapping).astype(float)
    else:
        event_num = event_raw.astype(float)

    mask = time.notna() & np.isfinite(time) & event_num.notna() & (time >= 0)

    # Include covariates in mask
    cov_cols = []
    for cov in covariates:
        if cov in df.columns:
            cov_cols.append(cov)
            mask = mask & df[cov].notna()

    work_df = pd.DataFrame({
        "T": time[mask].values.astype(float),
        "E": event_num[mask].values.astype(float),
    })

    for cov in cov_cols:
        cov_data = df.loc[mask, cov]
        if pd.api.types.is_numeric_dtype(cov_data):
            work_df[cov] = cov_data.values.astype(float)
        else:
            # Encode categorical covariates as dummies
            dummies = pd.get_dummies(cov_data, prefix=cov, drop_first=True)
            for col in dummies.columns:
                work_df[col] = dummies[col].values.astype(float)

    all_covariates = [c for c in work_df.columns if c not in ("T", "E")]

    if len(all_covariates) == 0:
        return {
            "error": "No valid covariates found after data preparation.",
            "validation": validation,
        }

    # Check for constant covariates
    constant_covs = [c for c in all_covariates if work_df[c].nunique() <= 1]
    if constant_covs:
        return {
            "error": f"Constant covariates detected: {', '.join(constant_covs)}. "
                     "Cox model requires variation in predictors.",
            "validation": validation,
        }

    # Check events per covariate
    n_events = int(work_df["E"].sum())
    n_covariates = len(all_covariates)
    events_per_covariate = n_events / n_covariates if n_covariates > 0 else 0

    # Fit Cox model
    try:
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(work_df, duration_col="T", event_col="E")

        # lifelines CoxPHFitter does not expose a converged_ flag.
        # If fit() completes without exception, we treat it as converged.
        converged = True

        # Extract results
        summary = cph.summary
        coefficients = {}
        hazard_ratios = {}
        standard_errors = {}
        p_values = {}
        confidence_intervals = {}

        for cov_name in summary.index:
            coefficients[cov_name] = round(float(summary.loc[cov_name, "coef"]), 6)
            hazard_ratios[cov_name] = round(float(summary.loc[cov_name, "exp(coef)"]), 6)
            standard_errors[cov_name] = round(float(summary.loc[cov_name, "se(coef)"]), 6)
            p_values[cov_name] = round(float(summary.loc[cov_name, "p"]), 8)

            ci_lower = float(summary.loc[cov_name, "exp(coef) lower 95%"])
            ci_upper = float(summary.loc[cov_name, "exp(coef) upper 95%"])
            confidence_intervals[cov_name] = (round(ci_lower, 6), round(ci_upper, 6))

        # Model fit statistics
        try:
            log_likelihood = float(cph.log_likelihood_)
        except Exception:
            log_likelihood = None

        try:
            aic = float(cph.AIC_partial_)
        except Exception:
            aic = None

        try:
            concordance = float(cph.concordance_index_)
        except Exception:
            concordance = None

        result: dict[str, Any] = {
            "coefficients": coefficients,
            "hazard_ratios": hazard_ratios,
            "standard_errors": standard_errors,
            "p_values": p_values,
            "confidence_intervals": confidence_intervals,
            "n_obs": len(work_df),
            "n_events": n_events,
            "n_covariates": n_covariates,
            "events_per_covariate": round(events_per_covariate, 2),
            "converged": converged,
            "log_likelihood": log_likelihood,
            "aic": aic,
            "concordance_index": concordance,
            "covariates": all_covariates,
            "penalizer": penalizer,
            "validation": validation,
        }

        # Proportional hazards test
        try:
            ph_test = cph.check_assumptions(work_df, p_value_threshold=0.05, show_plots=False)
            result["ph_test"] = "passed"
            result["ph_violations"] = []
        except Exception as e:
            ph_str = str(e)
            if "Proportional" in ph_str or "assumption" in ph_str.lower():
                result["ph_test"] = "violations_detected"
                # Parse violations
                violations = []
                for line in ph_str.split("\n"):
                    if "had p" in line.lower() or "proportional" in line.lower():
                        violations.append(line.strip())
                result["ph_violations"] = violations if violations else [ph_str]
            else:
                result["ph_test"] = "error"
                result["ph_violations"] = [ph_str]

        return result

    except Exception as e:
        error_msg = str(e)
        if "convergence" in error_msg.lower():
            return {
                "error": f"Cox model failed to converge: {error_msg}",
                "validation": validation,
                "converged": False,
            }
        return {
            "error": f"Cox model execution failed: {error_msg}",
            "validation": validation,
        }
