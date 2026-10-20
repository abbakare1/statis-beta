"""Reproduce the analysis planner defect with clinical trial dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from app.agents.analysis_planner import AnalysisPlannerAgent

# Load the clinical trial dataset
csv_path = Path("data/uploads/e825c062-891f-469e-a57c-c0bf7cb01758_rigorous_clinical_trial_data.csv")
df = pd.read_csv(csv_path)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print()

# Build data dictionary
variables = []
for col in df.columns:
    series = df[col]
    variables.append({
        "name": col,
        "dtype": str(series.dtype),
        "n_unique": int(series.nunique()),
        "n_total": len(series),
        "n_missing": int(series.isna().sum()),
        "missing_pct": float(series.isna().sum() / len(series) * 100),
    })

data_dict = {
    "n_rows": len(df),
    "n_cols": len(df.columns),
    "variables": variables,
}

# Research question
research_question = "Among participants enrolled in the two-arm clinical trial, does the Digital Intervention produce a greater reduction in perceived stress from baseline to follow-up compared with Standard Care?"

# Build context
context = {
    "project_id": "test-project",
    "dataset_id": "test-dataset",
    "file_path": str(csv_path),
    "data_dictionary": data_dict,
    "research_question": research_question,
    "study_design": "randomized controlled trial",
}

# Build data profile
planner = AnalysisPlannerAgent()
profile = planner._build_data_profile(context)

print(f"Outcome variable: {profile.outcome_variable}")
print(f"Outcome type: {profile.outcome_type}")
print(f"Grouping variables: {profile.grouping_variables}")
print(f"Grouping types: {profile.grouping_types}")
print(f"Predictor variables: {profile.predictor_variables}")
print(f"Predictor types: {profile.predictor_types}")
print(f"Covariates: {profile.covariates}")
print(f"Identifier variables: {profile.identifier_variables}")
print(f"Problem type: {profile.problem_type}")
print()

# Generate candidates
candidate_set = planner.candidate_generator.generate(profile)

print(f"Primary candidates: {[c.method_id for c in candidate_set.primary_candidates]}")
print(f"Sensitivity candidates: {[c.method_id for c in candidate_set.sensitivity_candidates]}")
print(f"Exploratory candidates: {[c.method_id for c in candidate_set.exploratory_candidates]}")
print()

# Build plan and inspect specifications
plan = planner._build_plan(context, profile, candidate_set, None)

for entry in plan.analyses:
    spec = entry.get_current_spec()
    print(f"Analysis: {entry.analysis_name}")
    print(f"  Method: {spec.analysis_method}")
    print(f"  Outcome: {spec.primary_outcome}")
    print(f"  Predictors: {list(spec.predictors)}")
    print(f"  Subgroup variables: {list(spec.subgroup_variables)}")
    print(f"  Covariates: {list(spec.covariates)}")
    print()
