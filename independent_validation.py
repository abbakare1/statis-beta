"""Independent numerical validation of STATIS clinical trial analysis.

This script independently computes the statistical results using SciPy and
statsmodels, then compares them with STATIS's output.
"""

import pandas as pd
import numpy as np
import scipy.stats as sp_stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

CSV_PATH = Path("data/uploads/e825c062-891f-469e-a57c-c0bf7cb01758_rigorous_clinical_trial_data.csv")


def load_data():
    """Load the clinical trial dataset."""
    df = pd.read_csv(CSV_PATH)
    return df


def primary_analysis(df):
    """Primary analysis: linear regression with baseline adjustment.
    
    Model: Followup_PSS10 ~ Treatment_Arm + Baseline_PSS10 + Age + BMI
    """
    print("=" * 80)
    print("PRIMARY ANALYSIS: Linear Regression with Baseline Adjustment")
    print("=" * 80)
    
    # Complete-case analysis
    vars_needed = ["Followup_PSS10", "Baseline_PSS10", "Treatment_Arm", "Age", "BMI"]
    df_complete = df.dropna(subset=vars_needed)
    
    print(f"\nTotal observations: {len(df)}")
    print(f"Complete-case observations: {len(df_complete)}")
    print(f"Excluded due to missing data: {len(df) - len(df_complete)}")
    
    # Encode treatment arm (Digital Intervention = 1, Standard Care = 0)
    df_complete = df_complete.copy()
    df_complete["treatment"] = (df_complete["Treatment_Arm"] == "Digital Intervention").astype(int)
    
    # Fit the model
    formula = "Followup_PSS10 ~ treatment + Baseline_PSS10 + Age + BMI"
    model = smf.ols(formula, data=df_complete).fit()
    
    print(f"\nModel: {formula}")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic: {model.fvalue:.4f}")
    print(f"p-value (F-test): {model.f_pvalue:.6f}")
    
    # Treatment effect
    treatment_coef = model.params["treatment"]
    treatment_se = model.bse["treatment"]
    treatment_t = model.tvalues["treatment"]
    treatment_p = model.pvalues["treatment"]
    treatment_ci = model.conf_int().loc["treatment"]
    
    print(f"\nTreatment Effect (Digital Intervention vs Standard Care):")
    print(f"  Coefficient: {treatment_coef:.4f}")
    print(f"  Standard Error: {treatment_se:.4f}")
    print(f"  95% CI: [{treatment_ci[0]:.4f}, {treatment_ci[1]:.4f}]")
    print(f"  t-statistic: {treatment_t:.4f}")
    print(f"  p-value: {treatment_p:.6f}")
    
    # All coefficients
    print(f"\nAll Coefficients:")
    print(model.summary())
    
    return {
        "n_obs": len(df_complete),
        "excluded": len(df) - len(df_complete),
        "treatment_coef": treatment_coef,
        "treatment_se": treatment_se,
        "treatment_t": treatment_t,
        "treatment_p": treatment_p,
        "treatment_ci_lower": treatment_ci[0],
        "treatment_ci_upper": treatment_ci[1],
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_statistic": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "model": model,
    }


def unadjusted_ttest(df):
    """Unadjusted independent t-test."""
    print("\n" + "=" * 80)
    print("SECONDARY ANALYSIS: Unadjusted Independent T-Test")
    print("=" * 80)
    
    digital = df[df["Treatment_Arm"] == "Digital Intervention"]["Followup_PSS10"].dropna()
    standard = df[df["Treatment_Arm"] == "Standard Care"]["Followup_PSS10"].dropna()
    
    print(f"\nDigital Intervention n={len(digital)}, mean={digital.mean():.2f}, sd={digital.std():.2f}")
    print(f"Standard Care n={len(standard)}, mean={standard.mean():.2f}, sd={standard.std():.2f}")
    
    # Welch's t-test (unequal variances)
    t_stat, p_value = sp_stats.ttest_ind(digital, standard, equal_var=False)
    
    # Pooled variance t-test
    t_pooled, p_pooled = sp_stats.ttest_ind(digital, standard, equal_var=True)
    
    # Effect size (Cohen's d)
    pooled_sd = np.sqrt(((len(digital) - 1) * digital.var() + (len(standard) - 1) * standard.var()) / (len(digital) + len(standard) - 2))
    cohens_d = (digital.mean() - standard.mean()) / pooled_sd
    
    print(f"\nWelch's t-test (unequal variances):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    
    print(f"\nPooled variance t-test (equal variances):")
    print(f"  t-statistic: {t_pooled:.4f}")
    print(f"  p-value: {p_pooled:.6f}")
    
    print(f"\nCohen's d: {cohens_d:.4f}")
    
    return {
        "n_digital": len(digital),
        "n_standard": len(standard),
        "mean_digital": digital.mean(),
        "mean_standard": standard.mean(),
        "t_welch": t_stat,
        "p_welch": p_value,
        "t_pooled": t_pooled,
        "p_pooled": p_pooled,
        "cohens_d": cohens_d,
    }


def mann_whitney(df):
    """Mann-Whitney U test."""
    print("\n" + "=" * 80)
    print("SECONDARY ANALYSIS: Mann-Whitney U Test")
    print("=" * 80)
    
    digital = df[df["Treatment_Arm"] == "Digital Intervention"]["Followup_PSS10"].dropna()
    standard = df[df["Treatment_Arm"] == "Standard Care"]["Followup_PSS10"].dropna()
    
    u_stat, p_value = sp_stats.mannwhitneyu(digital, standard, alternative="two-sided")
    
    print(f"\nMann-Whitney U test:")
    print(f"  U statistic: {u_stat:.0f}")
    print(f"  p-value: {p_value:.6f}")
    
    return {
        "n_digital": len(digital),
        "n_standard": len(standard),
        "u_statistic": u_stat,
        "p_value": p_value,
    }


def anova(df):
    """One-way ANOVA."""
    print("\n" + "=" * 80)
    print("SECONDARY ANALYSIS: One-Way ANOVA")
    print("=" * 80)
    
    # Use formula interface
    formula = "Followup_PSS10 ~ C(Treatment_Arm)"
    df_anova = df.dropna(subset=["Followup_PSS10", "Treatment_Arm"])
    model = smf.ols(formula, data=df_anova).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    print(f"\nANOVA Table:")
    print(anova_table)
    
    f_stat = anova_table.loc["C(Treatment_Arm)", "F"]
    p_value = anova_table.loc["C(Treatment_Arm)", "PR(>F)"]
    
    print(f"\nF-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    
    return {
        "f_statistic": f_stat,
        "p_value": p_value,
    }


def baseline_adjusted_ttest(df):
    """Baseline-adjusted t-test using residuals."""
    print("\n" + "=" * 80)
    print("SECONDARY ANALYSIS: Baseline-Adjusted T-Test (Residuals)")
    print("=" * 80)
    
    # Regress Followup_PSS10 on Baseline_PSS10 and get residuals
    df_adj = df.dropna(subset=["Followup_PSS10", "Baseline_PSS10", "Treatment_Arm"])
    model = smf.ols("Followup_PSS10 ~ Baseline_PSS10", data=df_adj).fit()
    residuals = model.resid
    
    # T-test on residuals by treatment group
    digital_resid = residuals[df_adj["Treatment_Arm"] == "Digital Intervention"]
    standard_resid = residuals[df_adj["Treatment_Arm"] == "Standard Care"]
    
    t_stat, p_value = sp_stats.ttest_ind(digital_resid, standard_resid, equal_var=False)
    
    print(f"\nBaseline-adjusted t-test (residuals):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
    }


def pearson_correlation(df):
    """Pearson correlation between baseline and follow-up PSS-10."""
    print("\n" + "=" * 80)
    print("SECONDARY ANALYSIS: Pearson Correlation (Baseline vs Follow-up PSS-10)")
    print("=" * 80)
    
    df_corr = df.dropna(subset=["Baseline_PSS10", "Followup_PSS10"])
    r, p_value = sp_stats.pearsonr(df_corr["Baseline_PSS10"], df_corr["Followup_PSS10"])
    
    print(f"\nPearson correlation:")
    print(f"  r = {r:.4f}")
    print(f"  p-value = {p_value:.6f}")
    
    return {
        "r": r,
        "p_value": p_value,
    }


def spearman_correlation(df):
    """Spearman correlation between baseline and follow-up PSS-10."""
    print("\n" + "=" * 80)
    print("SECONDARY ANALYSIS: Spearman Correlation (Baseline vs Follow-up PSS-10)")
    print("=" * 80)
    
    df_corr = df.dropna(subset=["Baseline_PSS10", "Followup_PSS10"])
    rho, p_value = sp_stats.spearmanr(df_corr["Baseline_PSS10"], df_corr["Followup_PSS10"])
    
    print(f"\nSpearman correlation:")
    print(f"  rho = {rho:.4f}")
    print(f"  p-value = {p_value:.6f}")
    
    return {
        "rho": rho,
        "p_value": p_value,
    }


def data_quality(df):
    """Data quality checks."""
    print("\n" + "=" * 80)
    print("DATA QUALITY CHECKS")
    print("=" * 80)
    
    # Extreme values
    print(f"\nAge range: {df['Age'].min():.0f} - {df['Age'].max():.0f}")
    print(f"BMI range: {df['BMI'].min():.1f} - {df['BMI'].max():.1f}")
    print(f"Baseline PSS-10 range: {df['Baseline_PSS10'].min()} - {df['Baseline_PSS10'].max()}")
    print(f"Follow-up PSS-10 range: {df['Followup_PSS10'].min():.1f} - {df['Followup_PSS10'].max():.1f}")
    print(f"Baseline Cortisol range: {df['Baseline_Cortisol_mcgdL'].min():.2f} - {df['Baseline_Cortisol_mcgdL'].max():.2f}")
    print(f"Follow-up Cortisol range: {df['Followup_Cortisol_mcgdL'].min():.2f} - {df['Followup_Cortisol_mcgdL'].max():.2f}")
    
    # Negative cortisol values
    neg_cortisol = df[df["Followup_Cortisol_mcgdL"] < 0]
    if len(neg_cortisol) > 0:
        print(f"\nWARNING: {len(neg_cortisol)} negative follow-up cortisol values")
        print(neg_cortisol[["Participant_ID", "Followup_Cortisol_mcgdL"]])
    
    # Extreme age
    extreme_age = df[(df["Age"] < 18) | (df["Age"] > 100)]
    if len(extreme_age) > 0:
        print(f"\nWARNING: {len(extreme_age)} extreme age values")
        print(extreme_age[["Participant_ID", "Age"]])
    
    # Extreme BMI
    extreme_bmi = df[(df["BMI"] < 15) | (df["BMI"] > 50)]
    if len(extreme_bmi) > 0:
        print(f"\nWARNING: {len(extreme_bmi)} extreme BMI values")
        print(extreme_bmi[["Participant_ID", "BMI"]])
    
    # Extreme baseline cortisol
    extreme_cortisol = df[df["Baseline_Cortisol_mcgdL"] > 30]
    if len(extreme_cortisol) > 0:
        print(f"\nWARNING: {len(extreme_cortisol)} extreme baseline cortisol values (>30 mcg/dL)")
        print(extreme_cortisol[["Participant_ID", "Baseline_Cortisol_mcgdL"]])
    
    # PSS-10 range check (should be 10-50)
    pss_out_of_range = df[(df["Baseline_PSS10"] < 10) | (df["Baseline_PSS10"] > 50)]
    if len(pss_out_of_range) > 0:
        print(f"\nWARNING: {len(pss_out_of_range)} baseline PSS-10 values out of range (10-50)")
        print(pss_out_of_range[["Participant_ID", "Baseline_PSS10"]])
    
    return None


def main():
    """Run all analyses."""
    df = load_data()
    
    # Data quality checks first
    data_quality(df)
    
    # Primary analysis
    primary = primary_analysis(df)
    
    # Secondary analyses
    ttest = unadjusted_ttest(df)
    mw = mann_whitney(df)
    anova_results = anova(df)
    adj_ttest = baseline_adjusted_ttest(df)
    pearson = pearson_correlation(df)
    spearman = spearman_correlation(df)
    
    print("\n" + "=" * 80)
    print("SUMMARY OF INDEPENDENT VALIDATION RESULTS")
    print("=" * 80)
    print(f"\nPrimary Analysis (Linear Regression with Baseline Adjustment):")
    print(f"  Treatment effect: {primary['treatment_coef']:.4f} (SE={primary['treatment_se']:.4f})")
    print(f"  95% CI: [{primary['treatment_ci_lower']:.4f}, {primary['treatment_ci_upper']:.4f}]")
    print(f"  p-value: {primary['treatment_p']:.6f}")
    print(f"  R-squared: {primary['r_squared']:.4f}")
    
    print(f"\nUnadjusted T-Test:")
    print(f"  t-statistic: {ttest['t_welch']:.4f}")
    print(f"  p-value: {ttest['p_welch']:.6f}")
    
    print(f"\nMann-Whitney U Test:")
    print(f"  U statistic: {mw['u_statistic']:.0f}")
    print(f"  p-value: {mw['p_value']:.6f}")
    
    print(f"\nOne-Way ANOVA:")
    print(f"  F-statistic: {anova_results['f_statistic']:.4f}")
    print(f"  p-value: {anova_results['p_value']:.6f}")
    
    print(f"\nBaseline-Adjusted T-Test (Residuals):")
    print(f"  t-statistic: {adj_ttest['t_statistic']:.4f}")
    print(f"  p-value: {adj_ttest['p_value']:.6f}")
    
    print(f"\nPearson Correlation (Baseline vs Follow-up PSS-10):")
    print(f"  r = {pearson['r']:.4f}")
    print(f"  p-value = {pearson['p_value']:.6f}")
    
    print(f"\nSpearman Correlation (Baseline vs Follow-up PSS-10):")
    print(f"  rho = {spearman['rho']:.4f}")
    print(f"  p-value = {spearman['p_value']:.6f}")


if __name__ == "__main__":
    main()
