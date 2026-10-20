"""Execute STATIS analysis through the API and capture results."""

import asyncio
import json
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add STATIS to path
sys.path.insert(0, str(Path(__file__).parent))

from app.api.main import app

CSV_PATH = Path("data/uploads/e825c062-891f-469e-a57c-c0bf7cb01758_rigorous_clinical_trial_data.csv")
RESEARCH_QUESTION = "Does the Digital Intervention produce a greater reduction in perceived stress from baseline to follow-up compared with Standard Care?"


def main():
    """Execute STATIS analysis and capture results."""
    print("=" * 80)
    print("EXECUTING STATIS ANALYSIS")
    print("=" * 80)
    
    client = TestClient(app)
    
    # Create project
    r = client.post("/api/projects", data={"name": "Validation Project", "description": "Independent numerical validation"})
    assert r.status_code == 200, f"Failed to create project: {r.text}"
    project = r.json()
    pid = project["id"]
    print(f"\nProject created: {pid}")
    
    # Upload dataset
    with open(CSV_PATH, "rb") as f:
        r = client.post(f"/api/projects/{pid}/upload", files={"file": f})
    assert r.status_code == 200, f"Failed to upload dataset: {r.text}"
    upload = r.json()
    did = upload["id"]
    print(f"Dataset uploaded: {did}")
    
    # Create analysis plan
    r = client.post("/api/analysis-plan/create", json={
        "project_id": pid,
        "dataset_id": did,
        "research_question": RESEARCH_QUESTION,
        "study_design": "parallel_group_trial",
    })
    assert r.status_code == 200, f"Failed to create analysis plan: {r.text}"
    plan = r.json()
    plan_id = plan["plan_id"]
    print(f"Analysis plan created: {plan_id}")
    print(f"Number of analyses: {len(plan['analyses'])}")
    
    # Print analysis plan
    print("\nAnalysis Plan:")
    for analysis in plan["analyses"]:
        print(f"  - {analysis['analysis_name']} ({analysis['method']})")
        print(f"    Outcome: {analysis['outcome']}")
        print(f"    Predictors: {analysis['predictors']}")
        print(f"    Covariates: {analysis['covariates']}")
        print(f"    Subgroup: {analysis.get('subgroup_variables', [])}")
    
    # Find and execute the linear regression analysis
    regression_analysis = None
    for analysis in plan["analyses"]:
        if analysis["method"] == "linear_regression":
            regression_analysis = analysis
            break
    
    if regression_analysis:
        aid = regression_analysis["analysis_id"]
        print(f"\nExecuting linear regression analysis: {aid}")
        
        # Approve and lock
        r = client.post(f"/api/analysis-plan/{aid}/approve", json={"plan_id": plan_id})
        assert r.status_code == 200, f"Failed to approve: {r.text}"
        
        r = client.post(f"/api/analysis-plan/{aid}/lock", json={"plan_id": plan_id})
        assert r.status_code == 200, f"Failed to lock: {r.text}"
        
        # Execute
        r = client.post(f"/api/analysis-plan/{aid}/execute", json={"plan_id": plan_id})
        assert r.status_code == 200, f"Failed to execute: {r.text}"
        result = r.json()
        
        print(f"\nExecution result:")
        print(json.dumps(result, indent=2))
        
        # Save results to file for comparison
        with open("statis_results.json", "w") as f:
            json.dump(result, f, indent=2)
        print("\nResults saved to statis_results.json")
    else:
        print("\nNo linear regression analysis found in plan")


if __name__ == "__main__":
    main()
