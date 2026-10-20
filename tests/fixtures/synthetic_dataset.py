"""Phase 2J — Deterministic Synthetic Research Dataset Fixture.

Generates a reproducible synthetic medical research cohort representative
of a real de-identified clinical dataset. Free of PHI. Seeded RNG.

Contains:
- Continuous, binary, categorical, ordinal variables
- Grouping variable (treatment arm)
- Continuous outcome, binary outcome, paired measurements
- Predictor variables with confounding-like relationships
- Missing values, mild outliers, skewed variables
- Identifier-like columns, high-cardinality variables
- Dates, time-to-event variable, censoring indicator
- Two survival groups

Data dictionary and ground-truth expectations included.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


SEED = 42
N_OBSERVATIONS = 600


def generate_synthetic_cohort(n: int = N_OBSERVATIONS, seed: int = SEED) -> pd.DataFrame:
    """Generate the deterministic synthetic research cohort.

    Returns a DataFrame with all columns documented in SYNTHETIC_DATA_DICTIONARY.
    Deterministic: same seed → same data.
    """
    rng = np.random.RandomState(seed)

    # --- Identifier-like columns (must never get analytical roles) ---
    patient_id = [f"PT-{i:05d}" for i in range(1, n + 1)]
    mrn = [f"MRN{100000 + i}" for i in range(1, n + 1)]
    site_code = rng.choice(["SITE-A", "SITE-B", "SITE-C", "SITE-D"], size=n, p=[0.35, 0.30, 0.20, 0.15])
    visit_date = pd.date_range("2023-01-15", periods=n, freq="D").strftime("%Y-%m-%d").tolist()

    # --- Demographics ---
    age = rng.normal(62, 14, n).clip(25, 95).round(1)
    sex = rng.choice(["M", "F"], size=n, p=[0.52, 0.48])
    race_ethnicity = rng.choice(
        ["White", "Black", "Hispanic", "Asian", "Other"],
        size=n, p=[0.55, 0.18, 0.14, 0.08, 0.05]
    )
    insurance = rng.choice(["Medicare", "Medicaid", "Private", "Self-pay"], size=n, p=[0.42, 0.22, 0.30, 0.06])

    # --- High-cardinality variable ---
    zip_code = rng.choice([f"{10000 + i}" for i in range(200)], size=n)

    # --- Treatment assignment (the grouping variable) ---
    # Treatment A = 0, Treatment B = 1
    treatment = rng.binomial(1, 0.48, n)
    treatment_label = np.where(treatment == 1, "Treatment_B", "Treatment_A")

    # --- Clinical measurements (continuous) ---
    # BMI: mildly skewed right
    bmi = np.exp(rng.normal(3.2, 0.18, n)).clip(16, 55).round(1)
    # Systolic BP: depends on age and treatment
    systolic_bp = (
        120 + 0.35 * (age - 62) + 4.5 * treatment + rng.normal(0, 12, n)
    ).clip(90, 200).round(1)
    # Diastolic BP
    diastolic_bp = (
        78 + 0.2 * (age - 62) + 2.0 * treatment + rng.normal(0, 8, n)
    ).clip(50, 130).round(1)
    # Cholesterol
    cholesterol = (
        210 - 0.5 * (age - 62) + rng.normal(0, 35, n)
    ).clip(120, 400).round(1)
    # Hemoglobin A1c
    hba1c = (
        6.5 + 0.4 * treatment + 0.01 * (age - 62) + rng.normal(0, 0.8, n)
    ).clip(4.5, 14).round(2)
    # Creatinine (skewed right)
    creatinine = np.exp(rng.normal(-0.2, 0.35, n)).clip(0.3, 5.0).round(2)
    # Duration of symptoms (days, skewed)
    symptom_duration = np.exp(rng.normal(4.5, 1.0, n)).clip(7, 1200).round(0)

    # --- Ordinal variable ---
    pain_severity = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.10, 0.22, 0.30, 0.24, 0.14])
    functional_status = rng.choice(
        ["Independent", "Modified", "Dependent", "Bedbound"],
        size=n, p=[0.40, 0.32, 0.20, 0.08]
    )

    # --- Comorbidities (binary) ---
    diabetes = rng.binomial(1, 0.32, n)
    hypertension = rng.binomial(1, 0.55, n)
    heart_failure = rng.binomial(1, 0.18, n)
    ckd = rng.binomial(1, 0.12, n)

    # --- Paired measurements (baseline and follow-up) ---
    # Baseline systolic BP and follow-up at 12 weeks
    sbp_baseline = systolic_bp.copy()
    treatment_effect_on_sbp = np.where(treatment == 1, -8.0, -3.0)
    sbp_followup = (
        sbp_baseline + treatment_effect_on_sbp + rng.normal(0, 10, n)
    ).clip(90, 200).round(1)

    # --- Continuous outcome: quality-of-life score (0-100) ---
    # Depends on treatment, age, pain severity, comorbidities
    qol_score = (
        80
        - 3.5 * treatment  # treatment effect
        - 0.15 * (age - 62)
        - 4.2 * pain_severity
        - 5.0 * diabetes
        - 3.0 * heart_failure
        + 0.3 * bmi
        + rng.normal(0, 8, n)
    ).clip(0, 100).round(1)

    # --- Binary outcome: 30-day readmission ---
    # Depends on treatment, heart failure, creatinine, age
    logit_readmission = (
        -2.5
        - 0.6 * treatment
        + 0.8 * heart_failure
        + 0.5 * ckd
        + 0.02 * (age - 62)
        + 0.3 * diabetes
        + rng.normal(0, 0.4, n)
    )
    prob_readmission = 1 / (1 + np.exp(-logit_readmission))
    readmission_30d = rng.binomial(1, prob_readmission, n)

    # --- Time-to-event data: survival time (days) and censoring ---
    # Exponential survival times with treatment effect
    baseline_hazard = 0.004
    hazard_multiplier = np.where(treatment == 1, 0.7, 1.0)  # treatment reduces hazard
    hazard = baseline_hazard * hazard_multiplier * (
        1 + 0.3 * heart_failure + 0.2 * ckd
    )
    survival_time = rng.exponential(1.0 / hazard)
    # Censoring time (uniform 30-365 days)
    censoring_time = rng.uniform(30, 365, n)
    # Observed time = min(survival, censoring)
    time_to_event = np.minimum(survival_time, censoring_time).round(1)
    # Event indicator: 1 if survival < censoring, 0 if censored
    event_observed = (survival_time < censoring_time).astype(int)

    # --- Introduce missing values (~3-5% per variable) ---
    missing_mask_hba1c = rng.random(n) < 0.04
    missing_mask_creatinine = rng.random(n) < 0.03
    missing_mask_cholesterol = rng.random(n) < 0.05
    missing_mask_qol = rng.random(n) < 0.03
    missing_mask_bmi = rng.random(n) < 0.02

    hba1c = np.where(missing_mask_hba1c, np.nan, hba1c)
    creatinine = np.where(missing_mask_creatinine, np.nan, creatinine)
    cholesterol = np.where(missing_mask_cholesterol, np.nan, cholesterol)
    qol_score = np.where(missing_mask_qol, np.nan, qol_score)
    bmi = np.where(missing_mask_bmi, np.nan, bmi)

    # --- Introduce mild outliers (~1-2%) ---
    outlier_indices = rng.choice(n, size=int(n * 0.015), replace=False)
    systolic_bp[outlier_indices] = systolic_bp[outlier_indices] + rng.choice([-30, 30], size=len(outlier_indices))

    # --- Build DataFrame ---
    df = pd.DataFrame({
        "patient_id": patient_id,
        "mrn": mrn,
        "site_code": site_code,
        "visit_date": visit_date,
        "zip_code": zip_code,
        "age": age,
        "sex": sex,
        "race_ethnicity": race_ethnicity,
        "insurance": insurance,
        "treatment": treatment,
        "treatment_label": treatment_label,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "cholesterol": cholesterol,
        "hba1c": hba1c,
        "creatinine": creatinine,
        "symptom_duration": symptom_duration,
        "pain_severity": pain_severity,
        "functional_status": functional_status,
        "diabetes": diabetes,
        "hypertension": hypertension,
        "heart_failure": heart_failure,
        "ckd": ckd,
        "sbp_baseline": sbp_baseline,
        "sbp_followup": sbp_followup,
        "qol_score": qol_score,
        "readmission_30d": readmission_30d,
        "time_to_event": time_to_event,
        "event_observed": event_observed,
    })

    return df


def save_synthetic_cohort(df: pd.DataFrame, path: Path) -> Path:
    """Save the synthetic cohort to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Data Dictionary
# ---------------------------------------------------------------------------

SYNTHETIC_DATA_DICTIONARY = {
    "patient_id": {
        "type": "string (identifier)",
        "description": "Unique patient identifier. NEVER assign analytical roles.",
        "role": "identifier",
    },
    "mrn": {
        "type": "string (identifier)",
        "description": "Medical record number. NEVER assign analytical roles.",
        "role": "identifier",
    },
    "site_code": {
        "type": "categorical",
        "description": "Enrollment site code (4 sites).",
        "role": "potential grouping/confounder",
    },
    "visit_date": {
        "type": "date string",
        "description": "Date of enrollment visit.",
        "role": "metadata",
    },
    "zip_code": {
        "type": "string (high-cardinality)",
        "description": "Postal code (200 unique values).",
        "role": "potential confounder",
    },
    "age": {
        "type": "continuous",
        "description": "Age in years (25-95).",
        "role": "predictor/confounder",
    },
    "sex": {
        "type": "binary categorical",
        "description": "Sex (M/F).",
        "role": "predictor/confounder",
    },
    "race_ethnicity": {
        "type": "categorical",
        "description": "Race/ethnicity (5 categories).",
        "role": "demographic",
    },
    "insurance": {
        "type": "categorical",
        "description": "Insurance type (4 categories).",
        "role": "demographic",
    },
    "treatment": {
        "type": "binary",
        "description": "Treatment assignment: 0=Treatment_A, 1=Treatment_B.",
        "role": "exposure/grouping",
    },
    "treatment_label": {
        "type": "string categorical",
        "description": "Treatment label: Treatment_A or Treatment_B.",
        "role": "exposure/grouping (label form)",
    },
    "bmi": {
        "type": "continuous (skewed right)",
        "description": "Body mass index (16-55). ~2% missing.",
        "role": "predictor/confounder",
    },
    "systolic_bp": {
        "type": "continuous",
        "description": "Systolic blood pressure mmHg (90-200). ~1.5% outliers.",
        "role": "predictor/outcome",
    },
    "diastolic_bp": {
        "type": "continuous",
        "description": "Diastolic blood pressure mmHg (50-130).",
        "role": "predictor",
    },
    "cholesterol": {
        "type": "continuous",
        "description": "Total cholesterol mg/dL (120-400). ~5% missing.",
        "role": "predictor",
    },
    "hba1c": {
        "type": "continuous",
        "description": "Hemoglobin A1c percent (4.5-14). ~4% missing.",
        "role": "predictor",
    },
    "creatinine": {
        "type": "continuous (skewed right)",
        "description": "Serum creatinine mg/dL (0.3-5.0). ~3% missing.",
        "role": "predictor",
    },
    "symptom_duration": {
        "type": "continuous (skewed right)",
        "description": "Duration of symptoms in days (7-1200).",
        "role": "predictor",
    },
    "pain_severity": {
        "type": "ordinal (1-5)",
        "description": "Pain severity score (1=mild, 5=severe).",
        "role": "predictor/confounder",
    },
    "functional_status": {
        "type": "ordinal categorical",
        "description": "Functional status (Independent/Modified/Dependent/Bedbound).",
        "role": "predictor",
    },
    "diabetes": {
        "type": "binary",
        "description": "Diabetes comorbidity (0/1).",
        "role": "confounder/comorbidity",
    },
    "hypertension": {
        "type": "binary",
        "description": "Hypertension comorbidity (0/1).",
        "role": "confounder/comorbidity",
    },
    "heart_failure": {
        "type": "binary",
        "description": "Heart failure comorbidity (0/1).",
        "role": "confounder/comorbidity",
    },
    "ckd": {
        "type": "binary",
        "description": "Chronic kidney disease comorbidity (0/1).",
        "role": "confounder/comorbidity",
    },
    "sbp_baseline": {
        "type": "continuous",
        "description": "Baseline systolic BP (identical to systolic_bp before treatment).",
        "role": "paired measurement (baseline)",
    },
    "sbp_followup": {
        "type": "continuous",
        "description": "12-week follow-up systolic BP. Treatment effect: -8 (B) vs -3 (A).",
        "role": "paired measurement (follow-up)",
    },
    "qol_score": {
        "type": "continuous (0-100)",
        "description": "Quality-of-life score. ~3% missing. Continuous outcome.",
        "role": "outcome",
    },
    "readmission_30d": {
        "type": "binary",
        "description": "30-day readmission (0/1). Binary outcome.",
        "role": "outcome",
    },
    "time_to_event": {
        "type": "continuous",
        "description": "Observed time in days (min of event/censoring time).",
        "role": "survival time",
    },
    "event_observed": {
        "type": "binary",
        "description": "Event indicator: 1=event, 0=censored.",
        "role": "survival event",
    },
}


# ---------------------------------------------------------------------------
# Ground-Truth Expectations
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    "n_observations": N_OBSERVATIONS,
    "seed": SEED,
    "treatment_prevalence": 0.48,
    "expected_treatment_effect_qol": -3.5,
    "expected_treatment_effect_readmission_logit": -0.6,
    "expected_heart_failure_readmission_logit": 0.8,
    "expected_pain_severity_qol": -4.2,
    "expected_diabetes_qol": -5.0,
    "expected_treatment_effect_sbp": -8.0,
    "missing_hba1c_rate": 0.04,
    "missing_creatinine_rate": 0.03,
    "missing_cholesterol_rate": 0.05,
    "missing_qol_rate": 0.03,
    "missing_bmi_rate": 0.02,
    "outlier_rate_systolic_bp": 0.015,
    "n_unique_zip_codes": 200,
    "n_sites": 4,
    "survival_groups": ["Treatment_A", "Treatment_B"],
    "notes": [
        "Treatment_B reduces QoL by ~3.5 points relative to Treatment_A",
        "Treatment_B reduces 30-day readmission log-odds by ~0.6",
        "Treatment_B reduces SBP by ~8 mmHg more than Treatment_A",
        "Higher pain_severity reduces QoL by ~4.2 points per level",
        "Heart failure increases readmission log-odds by ~0.8",
        "Survival: Treatment_B has lower hazard (hazard multiplier 0.7)",
        "1.5% mild outliers injected into systolic_bp",
        "Missing values injected in hba1c, creatinine, cholesterol, qol_score, bmi",
    ],
}
