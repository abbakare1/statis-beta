"""Debug variable role resolution."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from app.statistics.variable_roles import ResearchQuestionParser, RoleResolver

# Research question
research_question = "Among participants enrolled in the two-arm clinical trial, does the Digital Intervention produce a greater reduction in perceived stress from baseline to follow-up compared with Standard Care?"

# Parse research question
rq_intent = ResearchQuestionParser.parse(research_question)
print(f"Research question intent: {rq_intent}")
print()

# Load the clinical trial dataset
csv_path = Path("data/uploads/e825c062-891f-469e-a57c-c0bf7cb01758_rigorous_clinical_trial_data.csv")
df = pd.read_csv(csv_path)

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

# Resolve variable roles
resolutions = RoleResolver.resolve_all(variables, research_question)

for resolution in resolutions.resolutions:
    print(f"{resolution.variable_name}: {resolution.proposed_role.value} (confidence: {resolution.confidence})")
    if resolution.conflicts:
        print(f"  Conflicts: {resolution.conflicts}")
    if resolution.reasoning_summary:
        print(f"  Reasoning: {resolution.reasoning_summary}")
