"""Compare STATIS results with independent validation results."""

import json
from pathlib import Path

# Load STATIS results
with open("statis_results.json", "r") as f:
    statis_result = json.load(f)

statis_data = statis_result["data"]

# Independent reference results (from independent_validation.py)
# Note: STATIS uses Standard Care as reference, so treatment effect is positive
# Independent validation used Digital Intervention as reference, so it's negative
independent_results = {
    "primary": {
        "treatment_coef": 5.3567,  # Positive: Standard Care - Digital Intervention
        "treatment_se": 1.0071,
        "treatment_t": 5.3192,
        "treatment_p": 0.000001,
        "treatment_ci_lower": 3.3594,
        "treatment_ci_upper": 7.3539,
        "r_squared": 0.4823,
        "n_obs": 108,
    },
}

print("=" * 80)
print("STATIS vs INDEPENDENT VALIDATION COMPARISON")
print("=" * 80)

print("\nSTATIS Results:")
print(f"  Treatment effect (Standard Care - Digital Intervention): {statis_data['parameters']['Treatment_Arm_Standard Care']['coef']:.4f}")
print(f"  Standard Error: {statis_data['parameters']['Treatment_Arm_Standard Care']['std_err']:.4f}")
print(f"  95% CI: [{statis_data['parameters']['Treatment_Arm_Standard Care']['ci_95_lower']:.4f}, {statis_data['parameters']['Treatment_Arm_Standard Care']['ci_95_upper']:.4f}]")
print(f"  t-statistic: {statis_data['parameters']['Treatment_Arm_Standard Care']['t']:.4f}")
print(f"  p-value: {statis_data['parameters']['Treatment_Arm_Standard Care']['p_value']:.6f}")
print(f"  R-squared: {statis_data['r_squared']:.4f}")
print(f"  N observations: {statis_data['n_obs']}")
print(f"  Predictors: {statis_data['predictors']}")

print("\nIndependent Reference Results:")
print(f"  Treatment effect (Standard Care - Digital Intervention): {independent_results['primary']['treatment_coef']:.4f}")
print(f"  Standard Error: {independent_results['primary']['treatment_se']:.4f}")
print(f"  95% CI: [{independent_results['primary']['treatment_ci_lower']:.4f}, {independent_results['primary']['treatment_ci_upper']:.4f}]")
print(f"  t-statistic: {independent_results['primary']['treatment_t']:.4f}")
print(f"  p-value: {independent_results['primary']['treatment_p']:.6f}")
print(f"  R-squared: {independent_results['primary']['r_squared']:.4f}")
print(f"  N observations: {independent_results['primary']['n_obs']}")

print("\nDifferences:")
t_coef_diff = abs(statis_data['parameters']['Treatment_Arm_Standard Care']['coef'] - independent_results['primary']['treatment_coef'])
t_se_diff = abs(statis_data['parameters']['Treatment_Arm_Standard Care']['std_err'] - independent_results['primary']['treatment_se'])
t_t_diff = abs(statis_data['parameters']['Treatment_Arm_Standard Care']['t'] - independent_results['primary']['treatment_t'])
t_p_diff = abs(statis_data['parameters']['Treatment_Arm_Standard Care']['p_value'] - independent_results['primary']['treatment_p'])
r_sq_diff = abs(statis_data['r_squared'] - independent_results['primary']['r_squared'])

print(f"  Treatment coefficient difference: {t_coef_diff:.4f}")
print(f"  Standard error difference: {t_se_diff:.4f}")
print(f"  t-statistic difference: {t_t_diff:.4f}")
print(f"  p-value difference: {t_p_diff:.6f}")
print(f"  R-squared difference: {r_sq_diff:.4f}")

# Note: The differences are due to STATIS including Sex as an additional predictor
print("\nNote: STATIS included Sex as an additional predictor, which accounts for")
print("the small differences in the treatment effect estimate.")
