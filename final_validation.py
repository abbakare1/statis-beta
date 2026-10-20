"""Final targeted validation before private beta.

This script independently validates STATIS numerical results using
independently written implementations that do not call STATIS's
statistical calculation functions.
"""

import pandas as pd
import numpy as np
import scipy.stats as sp_stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

CSV_PATH = Path("data/uploads/e825c062-891f-469e-a57c-c0bf7cb01758_rigorous_clinical_trial_data.csv")

# STATIS results from previous execution
STATIS_PRIMARY = {
    "treatment_coef": 5.263031,
    "treatment_se": 1.001064,
    "treatment_t": 5.257436,
    "treatment_p": 8.1e-07,
    "treatment_ci_lower": 3.277425,
    "treatment_ci_upper": 7.248637,
    "r_squared": 0.495143,
    "adj_r_squared": 0.470395,
    "f_statistic": 20.00749,
    "n_obs": 108,
}


def load_data():
    """Load the clinical trial dataset."""
    df = pd.read_csv(CSV_PATH)
    return df


def verify_missing_data(df):
    """Verify missing data accounting."""
    print("=" * 80)
    print("MISSING DATA ACCOUNTING")
    print("=" * 80)
    
    # Check overlap of missing BMI and missing Followup_PSS10
    missing_bmi = df[df["BMI"].isnull()]
    missing_pss = df[df["Followup_PSS10"].isnull()]
    
    overlap = df[df["BMI"].isnull() & df["Followup_PSS10"].isnull()]
    
    print(f"\nMissing BMI: {len(missing_bmi)}")
    print(f"Missing Followup_PSS10: {len(missing_pss)}")
    print(f"Overlap (both missing): {len(overlap)}")
    print(f"Unique participants excluded: {len(missing_bmi) + len(missing_pss) - len(overlap)}")
    
    # Complete-case analysis
    vars_needed = ["Followup_PSS10", "Baseline_PSS10", "Treatment_Arm", "Age", "BMI", "Sex"]
    df_complete = df.dropna(subset=vars_needed)
    print(f"\nComplete-case observations: {len(df_complete)}")
    print(f"Excluded: {len(df) - len(df_complete)}")
    
    return df_complete


def primary_regression_with_sex(df):
    """Primary regression with Sex (matching STATIS)."""
    print("\n" + "=" * 80)
    print("PRIMARY REGRESSION WITH SEX (matching STATIS)")
    print("=" * 80)
    
    df_complete = df.dropna(subset=["Followup_PSS10", "Baseline_PSS10", "Treatment_Arm", "Age", "BMI", "Sex"])
    
    # Encode treatment arm (Digital Intervention = 1, Standard Care = 0)
    # STATIS uses Standard Care as reference, so we need to match that
    df_complete = df_complete.copy()
    df_complete["treatment"] = (df_complete["Treatment_Arm"] == "Digital Intervention").astype(int)
    
    # Fit the model matching STATIS: Standard Care as reference
    formula = "Followup_PSS10 ~ C(Treatment_Arm, Treatment('Digital Intervention')) + Baseline_PSS10 + Age + BMI + C(Sex, Treatment('Female'))"
    model = smf.ols(formula, data=df_complete).fit()
    
    print(f"\nModel: {formula}")
    print(f"R-squared: {model.rsquared:.6f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.6f}")
    print(f"F-statistic: {model.fvalue:.6f}")
    print(f"N observations: {len(df_complete)}")
    
    # Treatment effect (Standard Care vs Digital Intervention)
    treatment_param = "C(Treatment_Arm, Treatment('Digital Intervention'))[T.Standard Care]"
    treatment_coef = model.params[treatment_param]
    treatment_se = model.bse[treatment_param]
    treatment_t = model.tvalues[treatment_param]
    treatment_p = model.pvalues[treatment_param]
    treatment_ci = model.conf_int().loc[treatment_param]
    
    print(f"\nTreatment Effect (Standard Care - Digital Intervention):")
    print(f"  Coefficient: {treatment_coef:.6f}")
    print(f"  Standard Error: {treatment_se:.6f}")
    print(f"  95% CI: [{treatment_ci[0]:.6f}, {treatment_ci[1]:.6f}]")
    print(f"  t-statistic: {treatment_t:.6f}")
    print(f"  p-value: {treatment_p:.10f}")
    
    return {
        "treatment_coef": treatment_coef,
        "treatment_se": treatment_se,
        "treatment_t": treatment_t,
        "treatment_p": treatment_p,
        "treatment_ci_lower": treatment_ci[0],
        "treatment_ci_upper": treatment_ci[1],
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_statistic": model.fvalue,
        "n_obs": len(df_complete),
    }


def primary_regression_without_sex(df):
    """Primary regression without Sex (alternative model)."""
    print("\n" + "=" * 80)
    print("PRIMARY REGRESSION WITHOUT SEX (alternative model)")
    print("=" * 80)
    
    df_complete = df.dropna(subset=["Followup_PSS10", "Baseline_PSS10", "Treatment_Arm", "Age", "BMI"])
    
    # Fit the model
    formula = "Followup_PSS10 ~ C(Treatment_Arm, Treatment('Digital Intervention')) + Baseline_PSS10 + Age + BMI"
    model = smf.ols(formula, data=df_complete).fit()
    
    print(f"\nModel: {formula}")
    print(f"R-squared: {model.rsquared:.6f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.6f}")
    print(f"F-statistic: {model.fvalue:.6f}")
    print(f"N observations: {len(df_complete)}")
    
    # Treatment effect (Standard Care vs Digital Intervention)
    treatment_param = "C(Treatment_Arm, Treatment('Digital Intervention'))[T.Standard Care]"
    treatment_coef = model.params[treatment_param]
    treatment_se = model.bse[treatment_param]
    treatment_t = model.tvalues[treatment_param]
    treatment_p = model.pvalues[treatment_param]
    treatment_ci = model.conf_int().loc[treatment_param]
    
    print(f"\nTreatment Effect (Standard Care - Digital Intervention):")
    print(f"  Coefficient: {treatment_coef:.6f}")
    print(f"  Standard Error: {treatment_se:.6f}")
    print(f"  95% CI: [{treatment_ci[0]:.6f}, {treatment_ci[1]:.6f}]")
    print(f"  t-statistic: {treatment_t:.6f}")
    print(f"  p-value: {treatment_p:.10f}")
    
    return {
        "treatment_coef": treatment_coef,
        "treatment_se": treatment_se,
        "treatment_t": treatment_t,
        "treatment_p": treatment_p,
        "treatment_ci_lower": treatment_ci[0],
        "treatment_ci_upper": treatment_ci[1],
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_statistic": model.fvalue,
        "n_obs": len(df_complete),
    }


def independent_ttest(df):
    """Independent t-test (Welch's)."""
    print("\n" + "=" * 80)
    print("INDEPENDENT T-TEST (Welch's)")
    print("=" * 80)
    
    digital = df[df["Treatment_Arm"] == "Digital Intervention"]["Followup_PSS10"].dropna()
    standard = df[df["Treatment_Arm"] == "Standard Care"]["Followup_PSS10"].dropna()
    
    print(f"\nDigital Intervention n={len(digital)}, mean={digital.mean():.4f}, sd={digital.std():.4f}")
    print(f"Standard Care n={len(standard)}, mean={standard.mean():.4f}, sd={standard.std():.4f}")
    
    t_stat, p_value = sp_stats.ttest_ind(digital, standard, equal_var=False)
    
    # Pooled variance t-test for comparison
    t_pooled, p_pooled = sp_stats.ttest_ind(digital, standard, equal_var=True)
    
    print(f"\nWelch's t-test (unequal variances):")
    print(f"  t-statistic: {t_stat:.6f}")
    print(f"  p-value: {p_value:.10f}")
    
    print(f"\nPooled variance t-test (equal variances):")
    print(f"  t-statistic: {t_pooled:.6f}")
    print(f"  p-value: {p_pooled:.10f}")
    
    return {
        "t_welch": t_stat,
        "p_welch": p_value,
        "t_pooled": t_pooled,
        "p_pooled": p_pooled,
        "n_digital": len(digital),
        "n_standard": len(standard),
    }


def mann_whitney(df):
    """Mann-Whitney U test."""
    print("\n" + "=" * 80)
    print("MANN-WHITNEY U TEST")
    print("=" * 80)
    
    digital = df[df["Treatment_Arm"] == "Digital Intervention"]["Followup_PSS10"].dropna()
    standard = df[df["Treatment_Arm"] == "Standard Care"]["Followup_PSS10"].dropna()
    
    u_stat, p_value = sp_stats.mannwhitneyu(digital, standard, alternative="two-sided")
    
    print(f"\nMann-Whitney U test:")
    print(f"  U statistic: {u_stat:.0f}")
    print(f"  p-value: {p_value:.10f}")
    
    return {
        "u_statistic": u_stat,
        "p_value": p_value,
        "n_digital": len(digital),
        "n_standard": len(standard),
    }


def anova(df):
    """One-way ANOVA."""
    print("\n" + "=" * 80)
    print("ONE-WAY ANOVA")
    print("=" * 80)
    
    df_anova = df.dropna(subset=["Followup_PSS10", "Treatment_Arm"])
    formula = "Followup_PSS10 ~ C(Treatment_Arm)"
    model = smf.ols(formula, data=df_anova).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    print(f"\nANOVA Table:")
    print(anova_table)
    
    f_stat = anova_table.loc["C(Treatment_Arm)", "F"]
    p_value = anova_table.loc["C(Treatment_Arm)", "PR(>F)"]
    
    print(f"\nF-statistic: {f_stat:.6f}")
    print(f"p-value: {p_value:.10f}")
    
    return {
        "f_statistic": f_stat,
        "p_value": p_value,
        "n_obs": len(df_anova),
    }


def pearson_correlation(df):
    """Pearson correlation between baseline and follow-up PSS-10."""
    print("\n" + "=" * 80)
    print("PEARSON CORRELATION (Baseline vs Follow-up PSS-10)")
    print("=" * 80)
    
    df_corr = df.dropna(subset=["Baseline_PSS10", "Followup_PSS10"])
    r, p_value = sp_stats.pearsonr(df_corr["Baseline_PSS10"], df_corr["Followup_PSS10"])
    
    print(f"\nPearson correlation:")
    print(f"  r = {r:.6f}")
    print(f"  p-value = {p_value:.10f}")
    
    return {
        "r": r,
        "p_value": p_value,
        "n_obs": len(df_corr),
    }


def spearman_correlation(df):
    """Spearman correlation between baseline and follow-up PSS-10."""
    print("\n" + "=" * 80)
    print("SPEARMAN CORRELATION (Baseline vs Follow-up PSS-10)")
    print("=" * 80)
    
    df_corr = df.dropna(subset=["Baseline_PSS10", "Followup_PSS10"])
    rho, p_value = sp_stats.spearmanr(df_corr["Baseline_PSS10"], df_corr["Followup_PSS10"])
    
    print(f"\nSpearman correlation:")
    print(f"  rho = {rho:.6f}")
    print(f"  p-value = {p_value:.10f}")
    
    return {
        "rho": rho,
        "p_value": p_value,
        "n_obs": len(df_corr),
    }


def compare_results(label, statis_value, independent_value, tolerance=0.001):
    """Compare two values and report PASS/FAIL."""
    diff = abs(statis_value - independent_value)
    if diff <= tolerance:
        print(f"  {label}: STATIS={statis_value:.6f}, Independent={independent_value:.6f}, Diff={diff:.6f} [PASS]")
        return True
    else:
        print(f"  {label}: STATIS={statis_value:.6f}, Independent={independent_value:.6f}, Diff={diff:.6f} [FAIL]")
        return False


def main():
    """Run all validations."""
    df = load_data()
    
    # Verify missing data accounting
    df_complete = verify_missing_data(df)
    
    # Primary regression with Sex (matching STATIS)
    with_sex = primary_regression_with_sex(df)
    
    # Compare with STATIS
    print("\n" + "=" * 80)
    print("COMPARISON WITH STATIS (Primary Regression with Sex)")
    print("=" * 80)
    
    all_pass = True
    all_pass &= compare_results("Treatment coefficient", STATIS_PRIMARY["treatment_coef"], with_sex["treatment_coef"])
    all_pass &= compare_results("Treatment SE", STATIS_PRIMARY["treatment_se"], with_sex["treatment_se"])
    all_pass &= compare_results("Treatment t-statistic", STATIS_PRIMARY["treatment_t"], with_sex["treatment_t"])
    all_pass &= compare_results("Treatment p-value", STATIS_PRIMARY["treatment_p"], with_sex["treatment_p"], tolerance=1e-10)
    all_pass &= compare_results("Treatment CI lower", STATIS_PRIMARY["treatment_ci_lower"], with_sex["treatment_ci_lower"])
    all_pass &= compare_results("Treatment CI upper", STATIS_PRIMARY["treatment_ci_upper"], with_sex["treatment_ci_upper"])
    all_pass &= compare_results("R-squared", STATIS_PRIMARY["r_squared"], with_sex["r_squared"])
    all_pass &= compare_results("Adjusted R-squared", STATIS_PRIMARY["adj_r_squared"], with_sex["adj_r_squared"])
    all_pass &= compare_results("F-statistic", STATIS_PRIMARY["f_statistic"], with_sex["f_statistic"])
    
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    
    # Primary regression without Sex (alternative model)
    without_sex = primary_regression_without_sex(df)
    
    # Secondary analyses
    ttest = independent_ttest(df)
    mw = mann_whitney(df)
    anova_results = anova(df)
    pearson = pearson_correlation(df)
    spearman = spearman_correlation(df)
    
    print("\n" + "=" * 80)
    print("SUMMARY OF INDEPENDENT VALIDATION")
    print("=" * 80)
    print(f"\nPrimary Regression (with Sex):")
    print(f"  Treatment effect: {with_sex['treatment_coef']:.4f} (SE={with_sex['treatment_se']:.4f})")
    print(f"  95% CI: [{with_sex['treatment_ci_lower']:.4f}, {with_sex['treatment_ci_upper']:.4f}]")
    print(f"  p-value: {with_sex['treatment_p']:.6f}")
    print(f"  R-squared: {with_sex['r_squared']:.4f}")
    
    print(f"\nPrimary Regression (without Sex):")
    print(f"  Treatment effect: {without_sex['treatment_coef']:.4f} (SE={without_sex['treatment_se']:.4f})")
    print(f"  95% CI: [{without_sex['treatment_ci_lower']:.4f}, {without_sex['treatment_ci_upper']:.4f}]")
    print(f"  p-value: {without_sex['treatment_p']:.6f}")
    print(f"  R-squared: {without_sex['r_squared']:.4f}")
    
    print(f"\nUnadjusted T-Test (Welch's):")
    print(f"  t-statistic: {ttest['t_welch']:.4f}")
    print(f"  p-value: {ttest['p_welch']:.6f}")
    
    print(f"\nMann-Whitney U Test:")
    print(f"  U statistic: {mw['u_statistic']:.0f}")
    print(f"  p-value: {mw['p_value']:.6f}")
    
    print(f"\nOne-Way ANOVA:")
    print(f"  F-statistic: {anova_results['f_statistic']:.4f}")
    print(f"  p-value: {anova_results['p_value']:.6f}")
    
    print(f"\nPearson Correlation (Baseline vs Follow-up PSS-10):")
    print(f"  r = {pearson['r']:.4f}")
    print(f"  p-value = {pearson['p_value']:.6f}")
    
    print(f"\nSpearman Correlation (Baseline vs Follow-up PSS-10):")
    print(f"  rho = {spearman['rho']:.4f}")
    print(f"  p-value = {spearman['p_value']:.6f}")


if __name__ == "__main__":
    main()
