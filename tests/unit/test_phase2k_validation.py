"""Phase 2K — Controlled Real-World Validation.

Validates STATIS end-to-end on real datasets:
- STATIS_Test1_Continuous.csv: t-test on continuous outcome
- STATIS_Test2_Binary.csv: chi-square on binary outcome
- STATIS_Test3_Association.csv: correlation analysis
- STATIS_Test4_Confounding.csv: logistic regression with covariates

Cross-checks numerical results against scipy/statsmodels.
Verifies audit trails, dataset identity, provenance, and reproducibility.
Verifies persistence after database close/reinitialization.
"""

import asyncio
import hashlib
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOWNLOADS = Path("C:/Users/ajibo/Downloads")


@pytest.fixture(scope="module")
def auth_headers():
    """Create a test user with generous quotas and return auth headers."""
    from app.security.auth import create_user, create_access_token
    from app.core.database import get_session

    async def _setup():
        async with get_session() as session:
            user = await create_user(session, f"phase2k_{uuid.uuid4().hex[:8]}@example.com", "password123", "Phase 2K Validation")
            user.max_projects = 100
            user.max_analyses = 1000
            user.max_upload_mb = 1000
            await session.commit()
            return user

    user = asyncio.run(_setup())
    token = create_access_token(user.id, user.role)
    return {"Authorization": f"Bearer {token}"}

# Expected results computed independently with scipy/statsmodels
EXPECTED = {
    "test1_ttest": {
        "t_stat": -5.634,
        "p_value": 0.0,
        "effect_size": -1.037,
        "n_control": 60,
        "n_treatment": 60,
    },
    "test2_chisq": {
        "chi2_stat": 0.866,
        "p_value": 0.352,
        "odds_ratio": 0.706,
        "n_control": 100,
        "n_treatment": 100,
    },
    "test3_correlation": {
        "pearson_r": 0.577,
        "p_value": 0.0,
        "spearman_r": 0.556,
        "n": 150,
    },
    "test4_logistic": {
        "treatment_coef": -0.178,
        "treatment_p": 0.386,
        "treatment_or": 0.837,
        "age_coef": 0.069,
        "n": 500,
    },
}


@pytest.fixture(scope="module")
def client():
    from app.api.main import app
    with TestClient(app) as c:
        yield c


def _uid() -> str:
    return str(uuid.uuid4())[:8]


def _create_project(client: TestClient, headers: dict, name: str = "Phase 2K Validation") -> str:
    r = client.post("/api/projects", data={
        "name": f"{name} {_uid()}",
        "description": "Phase 2K real-dataset validation",
        "research_question": "Does treatment affect outcome?",
        "study_design": "cohort",
    }, headers=headers)
    assert r.status_code == 200, f"Failed to create project: {r.text}"
    return r.json()["id"]


def _upload_dataset(client: TestClient, headers: dict, project_id: str, csv_path: Path) -> dict:
    r = client.post(f"/api/projects/{project_id}/upload", files={
        "file": (csv_path.name, csv_path.read_bytes(), "text/csv"),
    }, headers=headers)
    assert r.status_code == 200, f"Failed to upload {csv_path.name}: {r.text}"
    return r.json()


def _independent_ttest(df, outcome, group, control=0, treatment=1):
    """Independent Welch t-test computed directly with scipy."""
    from scipy import stats
    y_ctrl = df[df[group] == control][outcome].values
    y_trt = df[df[group] == treatment][outcome].values
    t, p = stats.ttest_ind(y_trt, y_ctrl, equal_var=False)
    pooled_sd = np.sqrt(((len(y_ctrl)-1)*y_ctrl.var() + (len(y_trt)-1)*y_trt.var()) / (len(y_ctrl)+len(y_trt)-2))
    d = (y_trt.mean() - y_ctrl.mean()) / pooled_sd
    return {"t": t, "p": p, "d": d, "n_ctrl": len(y_ctrl), "n_trt": len(y_trt)}


def _independent_chisq(df, outcome, group, control=0, treatment=1):
    """Chi-square test of independence computed directly with scipy."""
    from scipy import stats
    n_ctrl = len(df[df[group] == control])
    n_trt = len(df[df[group] == treatment])
    n_ctrl_event = int(df[df[group] == control][outcome].sum())
    n_trt_event = int(df[df[group] == treatment][outcome].sum())
    n_ctrl_no = n_ctrl - n_ctrl_event
    n_trt_no = n_trt - n_trt_event
    table = [[n_trt_event, n_trt_no], [n_ctrl_event, n_ctrl_no]]
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    or_val = (n_trt_event * n_ctrl_no) / (n_trt_no * n_ctrl_event) if (n_trt_no * n_ctrl_event) > 0 else float('inf')
    return {"chi2": chi2, "p": p, "or": or_val, "n_ctrl": n_ctrl, "n_trt": n_trt}


def _independent_correlation(df, x, y):
    """Pearson and Spearman correlations computed directly with scipy."""
    from scipy import stats
    r, p_pearson = stats.pearsonr(df[x], df[y])
    rho, p_spearman = stats.spearmanr(df[x], df[y])
    return {"pearson_r": r, "pearson_p": p_pearson, "spearman_r": rho, "spearman_p": p_spearman, "n": len(df)}


def _independent_logistic(df, outcome, predictors, alpha=0.05):
    """Logistic regression computed directly with statsmodels."""
    import statsmodels.api as sm
    X = df[predictors].values
    X = sm.add_constant(X)
    y = df[outcome].values
    model = sm.Logit(y, X).fit(disp=0)
    return model


# ============================================================================
# Test 1: Continuous outcome — independent t-test
# ============================================================================

def test_phase2k_test1_ttest(client, auth_headers):
    """Test1: Independent t-test on continuous outcome (systolic_bp)."""
    csv_path = DOWNLOADS / "STATIS_Test1_Continuous.csv"
    df = pd.read_csv(csv_path)

    # Independent reference computation
    ref = _independent_ttest(df, "systolic_bp", "treatment")
    expected = EXPECTED["test1_ttest"]
    assert abs(ref["t"] - expected["t_stat"]) < 0.01, f"Reference t mismatch: {ref['t']} vs {expected['t_stat']}"

    # Upload to STATIS
    pid = _create_project(client, auth_headers, "Test1 T-test")
    upload = _upload_dataset(client, auth_headers, pid, csv_path)
    did = upload["id"]
    content_hash = upload["content_hash"]

    # Verify dataset identity
    from app.data.dataset_identity import compute_dataset_identity
    identity = compute_dataset_identity(csv_path)
    assert content_hash == identity["content_hash"], "Upload content_hash mismatch"

    # Run analysis plan
    data_dictionary = {
        "n_rows": len(df),
        "variables": [
            {"name": "patient_id", "dtype": "object", "n_unique": len(df), "n_missing": 0, "n_total": len(df)},
            {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_missing": 0, "n_total": len(df)},
            {"name": "systolic_bp", "dtype": "float64", "n_unique": len(df), "n_missing": 0, "n_total": len(df)},
        ],
    }
    r = client.post("/api/analysis-plan/create", json={
        "project_id": pid,
        "dataset_id": did,
        "research_question": "Does treatment affect systolic blood pressure?",
        "study_design": "cohort",
        "data_dictionary": data_dictionary,
        "data_quality": {"n_rows": len(df), "n_cols": 3},
    }, headers=auth_headers)
    assert r.status_code == 200, f"Failed to create analysis plan: {r.text}"
    plan = r.json()
    plan_id = plan["plan_id"]

    # Find t-test analysis
    ttest_analysis = None
    for a in plan["analyses"]:
        if a["method"] == "independent_t_test":
            ttest_analysis = a
            break
    assert ttest_analysis is not None, "No t-test analysis in plan"
    analysis_id = ttest_analysis["analysis_id"]

    # Approve and lock
    r = client.post(f"/api/analysis-plan/{analysis_id}/approve", json={"plan_id": plan_id}, headers=auth_headers)
    assert r.status_code == 200, f"Failed to approve: {r.text}"
    r = client.post(f"/api/analysis-plan/{analysis_id}/lock", json={"plan_id": plan_id}, headers=auth_headers)
    assert r.status_code == 200, f"Failed to lock: {r.text}"

    # Execute
    r = client.post(f"/api/analysis-plan/{analysis_id}/execute", json={"plan_id": plan_id}, headers=auth_headers)
    assert r.status_code == 200, f"Failed to execute: {r.text}"
    result = r.json()
    assert result.get("success"), f"Analysis failed: {result}"

    # Cross-check numerical results
    result_data = result.get("data", {})
    assert "t_statistic" in result_data, f"Missing t_statistic in result. Result keys: {list(result_data.keys())}"
    assert "p_value" in result_data, "Missing p_value in result"
    assert abs(result_data["t_statistic"] - expected["t_stat"]) < 0.01, \
        f"STATIS t-stat mismatch: {result_data['t_statistic']} vs {expected['t_stat']}"
    assert abs(result_data["p_value"] - expected["p_value"]) < 0.001, \
        f"STATIS p-value mismatch: {result_data['p_value']} vs {expected['p_value']}"

    # Verify audit trail
    r = client.get(f"/api/audit?project_id={pid}", headers=auth_headers)
    assert r.status_code == 200
    audit = r.json()
    assert len(audit) > 0, "No audit entries"
    actions = [e["action"] for e in audit]
    assert "analysis_executed" in actions, "No analysis_executed audit entry"

    # Verify dataset identity in run record
    r = client.get(f"/api/runs?project_id={pid}", headers=auth_headers)
    assert r.status_code == 200
    runs = r.json()
    assert len(runs) > 0, "No analysis runs"
    run = runs[0]
    assert run["dataset_content_hash"] == content_hash, "Dataset identity mismatch in run record"


# ============================================================================
# Test 2: Binary outcome — chi-square test
# ============================================================================

def test_phase2k_test2_chisq(client, auth_headers):
    """Test2: Chi-square test on binary outcome (hospitalized_30d)."""
    csv_path = DOWNLOADS / "STATIS_Test2_Binary.csv"
    df = pd.read_csv(csv_path)

    # Independent reference computation
    ref = _independent_chisq(df, "hospitalized_30d", "treatment")
    expected = EXPECTED["test2_chisq"]
    assert abs(ref["chi2"] - expected["chi2_stat"]) < 0.01, f"Reference chi2 mismatch: {ref['chi2']} vs {expected['chi2_stat']}"

    # Upload to STATIS
    pid = _create_project(client, auth_headers, "Test2 Chi-square")
    upload = _upload_dataset(client, auth_headers, pid, csv_path)
    did = upload["id"]

    # Run analysis plan
    data_dictionary = {
        "n_rows": len(df),
        "variables": [
            {"name": "patient_id", "dtype": "object", "n_unique": len(df), "n_missing": 0, "n_total": len(df)},
            {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_missing": 0, "n_total": len(df)},
            {"name": "hospitalized_30d", "dtype": "int64", "n_unique": 2, "n_missing": 0, "n_total": len(df)},
        ],
    }
    r = client.post("/api/analysis-plan/create", json={
        "project_id": pid,
        "dataset_id": did,
        "research_question": "Does treatment affect the hospitalized_30d outcome?",
        "study_design": "cohort",
        "data_dictionary": data_dictionary,
        "data_quality": {"n_rows": len(df), "n_cols": 3},
    }, headers=auth_headers)
    assert r.status_code == 200, f"Failed to create analysis plan: {r.text}"
    plan = r.json()
    plan_id = plan["plan_id"]

    # Find chi-square analysis
    chisq_analysis = None
    for a in plan["analyses"]:
        if a["method"] == "chi_square":
            chisq_analysis = a
            break
    assert chisq_analysis is not None, "No chi-square analysis in plan"
    analysis_id = chisq_analysis["analysis_id"]

    # Approve and lock
    r = client.post(f"/api/analysis-plan/{analysis_id}/approve", json={"plan_id": plan_id}, headers=auth_headers)
    assert r.status_code == 200
    r = client.post(f"/api/analysis-plan/{analysis_id}/lock", json={"plan_id": plan_id}, headers=auth_headers)
    assert r.status_code == 200

    # Execute
    r = client.post(f"/api/analysis-plan/{analysis_id}/execute", json={"plan_id": plan_id}, headers=auth_headers)
    assert r.status_code == 200, f"Failed to execute: {r.text}"
    result = r.json()
    assert result.get("success"), f"Analysis failed: {result}"

    # Cross-check numerical results
    result_data = result.get("data", {})
    assert "chi2" in result_data, "Missing chi2 in result"
    assert "p_value" in result_data, "Missing p_value in result"
    assert abs(result_data["chi2"] - expected["chi2_stat"]) < 0.01, \
        f"STATIS chi2 mismatch: {result_data['chi2']} vs {expected['chi2_stat']}"
    assert abs(result_data["p_value"] - expected["p_value"]) < 0.001, \
        f"STATIS p-value mismatch: {result_data['p_value']} vs {expected['p_value']}"


# ============================================================================
# Test 3: Association — Pearson correlation
# ============================================================================

def test_phase2k_test3_correlation(client, auth_headers):
    """Test3: Pearson correlation between BMI and biomarker."""
    csv_path = DOWNLOADS / "STATIS_Test3_Association.csv"
    df = pd.read_csv(csv_path)

    # Independent reference computation
    ref = _independent_correlation(df, "bmi", "biomarker")
    expected = EXPECTED["test3_correlation"]
    assert abs(ref["pearson_r"] - expected["pearson_r"]) < 0.01, f"Reference r mismatch: {ref['pearson_r']} vs {expected['pearson_r']}"

    # Upload to STATIS
    pid = _create_project(client, auth_headers, "Test3 Correlation")
    upload = _upload_dataset(client, auth_headers, pid, csv_path)
    did = upload["id"]

    # Run analysis plan
    data_dictionary = {
        "n_rows": len(df),
        "variables": [
            {"name": "patient_id", "dtype": "object", "n_unique": len(df), "n_missing": 0, "n_total": len(df)},
            {"name": "bmi", "dtype": "float64", "n_unique": len(df), "n_missing": 0, "n_total": len(df)},
            {"name": "biomarker", "dtype": "float64", "n_unique": len(df), "n_missing": 0, "n_total": len(df)},
        ],
    }
    r = client.post("/api/analysis-plan/create", json={
        "project_id": pid,
        "dataset_id": did,
        "research_question": "Is BMI associated with biomarker level?",
        "study_design": "cross_sectional",
        "data_dictionary": data_dictionary,
        "data_quality": {"n_rows": len(df), "n_cols": 3},
    }, headers=auth_headers)
    assert r.status_code == 200, f"Failed to create analysis plan: {r.text}"
    plan = r.json()
    plan_id = plan["plan_id"]

    # Find Pearson correlation analysis
    corr_analysis = None
    for a in plan["analyses"]:
        if a["method"] == "pearson_correlation":
            corr_analysis = a
            break
    assert corr_analysis is not None, "No Pearson correlation analysis in plan"
    analysis_id = corr_analysis["analysis_id"]

    # Approve and lock
    r = client.post(f"/api/analysis-plan/{analysis_id}/approve", json={"plan_id": plan_id}, headers=auth_headers)
    assert r.status_code == 200
    r = client.post(f"/api/analysis-plan/{analysis_id}/lock", json={"plan_id": plan_id}, headers=auth_headers)
    assert r.status_code == 200

    # Execute
    r = client.post(f"/api/analysis-plan/{analysis_id}/execute", json={"plan_id": plan_id}, headers=auth_headers)
    assert r.status_code == 200, f"Failed to execute: {r.text}"
    result = r.json()
    assert result.get("success"), f"Analysis failed: {result}"

    # Cross-check numerical results
    result_data = result.get("data", {})
    assert "r" in result_data, "Missing r in result"
    assert "p_value" in result_data, "Missing p_value in result"
    assert abs(result_data["r"] - expected["pearson_r"]) < 0.01, \
        f"STATIS correlation mismatch: {result_data['r']} vs {expected['pearson_r']}"


# ============================================================================
# Test 4: Confounding — logistic regression with covariates
# ============================================================================

def test_phase2k_test4_logistic(client, auth_headers):
    """Test4: Logistic regression with treatment, age, sex, comorbidities."""
    csv_path = DOWNLOADS / "STATIS_Test4_Confounding.csv"
    df = pd.read_csv(csv_path)

    # Independent reference computation
    predictors = ["age", "sex_male", "comorbidity_count", "treatment"]
    model = _independent_logistic(df, "hospitalized_30d", predictors)
    expected = EXPECTED["test4_logistic"]
    treatment_idx = predictors.index("treatment")
    assert abs(model.params[treatment_idx + 1] - expected["treatment_coef"]) < 0.01, \
        f"Reference treatment coef mismatch: {model.params[treatment_idx + 1]} vs {expected['treatment_coef']}"

    # Upload to STATIS
    pid = _create_project(client, auth_headers, "Test4 Logistic")
    upload = _upload_dataset(client, auth_headers, pid, csv_path)
    did = upload["id"]

    # Run analysis plan
    data_dictionary = {
        "n_rows": len(df),
        "variables": [
            {"name": "patient_id", "dtype": "object", "n_unique": len(df), "n_missing": 0, "n_total": len(df)},
            {"name": "age", "dtype": "float64", "n_unique": len(df), "n_missing": 0, "n_total": len(df)},
            {"name": "sex_male", "dtype": "int64", "n_unique": 2, "n_missing": 0, "n_total": len(df)},
            {"name": "comorbidity_count", "dtype": "int64", "n_unique": 5, "n_missing": 0, "n_total": len(df)},
            {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_missing": 0, "n_total": len(df)},
            {"name": "hospitalized_30d", "dtype": "int64", "n_unique": 2, "n_missing": 0, "n_total": len(df)},
        ],
    }
    r = client.post("/api/analysis-plan/create", json={
        "project_id": pid,
        "dataset_id": did,
        "research_question": "Does treatment affect the hospitalized_30d outcome after adjusting for age, sex, and comorbidities?",
        "study_design": "cohort",
        "data_dictionary": data_dictionary,
        "data_quality": {"n_rows": len(df), "n_cols": 6},
    }, headers=auth_headers)
    assert r.status_code == 200, f"Failed to create analysis plan: {r.text}"
    plan = r.json()
    plan_id = plan["plan_id"]

    # Find logistic regression analysis
    logreg_analysis = None
    for a in plan["analyses"]:
        if a["method"] == "logistic_regression":
            logreg_analysis = a
            break
    assert logreg_analysis is not None, "No logistic regression analysis in plan"
    analysis_id = logreg_analysis["analysis_id"]

    # Approve and lock
    r = client.post(f"/api/analysis-plan/{analysis_id}/approve", json={"plan_id": plan_id}, headers=auth_headers)
    assert r.status_code == 200
    r = client.post(f"/api/analysis-plan/{analysis_id}/lock", json={"plan_id": plan_id}, headers=auth_headers)
    assert r.status_code == 200

    # Execute
    r = client.post(f"/api/analysis-plan/{analysis_id}/execute", json={"plan_id": plan_id}, headers=auth_headers)
    assert r.status_code == 200, f"Failed to execute: {r.text}"
    result = r.json()
    assert result.get("success"), f"Analysis failed: {result}"

    # Cross-check numerical results
    result_data = result.get("data", {})
    assert "parameters" in result_data, "Missing parameters in result"
    treatment_params = result_data["parameters"].get("treatment")
    assert treatment_params is not None, "Missing treatment parameter"
    assert abs(treatment_params["coef"] - expected["treatment_coef"]) < 0.01, \
        f"STATIS treatment coef mismatch: {treatment_params['coef']} vs {expected['treatment_coef']}"
    assert abs(treatment_params["p_value"] - expected["treatment_p"]) < 0.001, \
        f"STATIS treatment p-value mismatch: {treatment_params['p_value']} vs {expected['treatment_p']}"


# ============================================================================
# Test 5: Reproducibility — same inputs produce same outputs
# ============================================================================

def test_phase2k_reproducibility(client, auth_headers):
    """Verify that running the same analysis twice produces identical results."""
    csv_path = DOWNLOADS / "STATIS_Test1_Continuous.csv"
    df = pd.read_csv(csv_path)

    pid = _create_project(client, auth_headers, "Reproducibility Test")
    upload = _upload_dataset(client, auth_headers, pid, csv_path)
    did = upload["id"]

    data_dictionary = {
        "n_rows": len(df),
        "variables": [
            {"name": "patient_id", "dtype": "object", "n_unique": len(df), "n_missing": 0, "n_total": len(df)},
            {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_missing": 0, "n_total": len(df)},
            {"name": "systolic_bp", "dtype": "float64", "n_unique": len(df), "n_missing": 0, "n_total": len(df)},
        ],
    }
    r = client.post("/api/analysis-plan/create", json={
        "project_id": pid,
        "dataset_id": did,
        "research_question": "Does treatment affect systolic blood pressure?",
        "study_design": "cohort",
        "data_dictionary": data_dictionary,
        "data_quality": {"n_rows": len(df), "n_cols": 3},
    }, headers=auth_headers)
    assert r.status_code == 200
    plan = r.json()
    plan_id = plan["plan_id"]

    # Find t-test analysis
    ttest_analysis = None
    for a in plan["analyses"]:
        if a["method"] == "independent_t_test":
            ttest_analysis = a
            break
    assert ttest_analysis is not None
    analysis_id = ttest_analysis["analysis_id"]

    # Approve and lock
    client.post(f"/api/analysis-plan/{analysis_id}/approve", json={"plan_id": plan_id}, headers=auth_headers)
    client.post(f"/api/analysis-plan/{analysis_id}/lock", json={"plan_id": plan_id}, headers=auth_headers)

    # Execute first time
    r1 = client.post(f"/api/analysis-plan/{analysis_id}/execute", json={"plan_id": plan_id}, headers=auth_headers)
    assert r1.status_code == 200
    result1 = r1.json()["data"]

    # Get the run fingerprint
    r = client.get(f"/api/runs?project_id={pid}", headers=auth_headers)
    runs = r.json()
    assert len(runs) == 1
    run1_fp = runs[0]["result_fingerprint"]

    # Execute second time (need new analysis since first is locked/executed)
    # Create a new plan for the second run
    r = client.post("/api/analysis-plan/create", json={
        "project_id": pid,
        "dataset_id": did,
        "research_question": "Does treatment affect systolic blood pressure?",
        "study_design": "cohort",
        "data_dictionary": data_dictionary,
        "data_quality": {"n_rows": len(df), "n_cols": 3},
    }, headers=auth_headers)
    assert r.status_code == 200
    plan2 = r.json()
    plan_id2 = plan2["plan_id"]

    ttest_analysis2 = None
    for a in plan2["analyses"]:
        if a["method"] == "independent_t_test":
            ttest_analysis2 = a
            break
    assert ttest_analysis2 is not None
    analysis_id2 = ttest_analysis2["analysis_id"]

    client.post(f"/api/analysis-plan/{analysis_id2}/approve", json={"plan_id": plan_id2}, headers=auth_headers)
    client.post(f"/api/analysis-plan/{analysis_id2}/lock", json={"plan_id": plan_id2}, headers=auth_headers)

    r2 = client.post(f"/api/analysis-plan/{analysis_id2}/execute", json={"plan_id": plan_id2}, headers=auth_headers)
    assert r2.status_code == 200
    result2 = r2.json()["data"]

    # Results should be identical
    assert result1["t_statistic"] == result2["t_statistic"], "t-statistic differs between runs"
    assert result1["p_value"] == result2["p_value"], "p-value differs between runs"

    # Verify fingerprints match
    r = client.get(f"/api/runs?project_id={pid}", headers=auth_headers)
    runs = r.json()
    assert len(runs) == 2
    run2_fp = runs[1]["result_fingerprint"]
    assert run1_fp == run2_fp, f"Result fingerprints differ: {run1_fp} vs {run2_fp}"


# ============================================================================
# Test 6: Dataset identity verification
# ============================================================================

def test_phase2k_dataset_identity(client, auth_headers):
    """Verify that dataset identity is preserved throughout the workflow."""
    csv_path = DOWNLOADS / "STATIS_Test1_Continuous.csv"
    df = pd.read_csv(csv_path)

    # Compute identity independently
    from app.data.dataset_identity import compute_dataset_identity
    identity = compute_dataset_identity(csv_path)
    expected_hash = identity["content_hash"]

    # Upload
    pid = _create_project(client, auth_headers, "Dataset Identity Test")
    upload = _upload_dataset(client, auth_headers, pid, csv_path)
    assert upload["content_hash"] == expected_hash, "Upload hash mismatch"

    # List datasets
    r = client.get(f"/api/projects/{pid}/datasets", headers=auth_headers)
    datasets = r.json()
    assert len(datasets) == 1
    assert datasets[0]["content_hash"] == expected_hash, "List hash mismatch"

    # Preview
    r = client.get(f"/api/datasets/{upload['id']}/preview", headers=auth_headers)
    assert r.json()["content_hash"] == expected_hash, "Preview hash mismatch"


# ============================================================================
# Test 7: Audit trail completeness
# ============================================================================

def test_phase2k_audit_trail(client, auth_headers):
    """Verify that all major actions are captured in the audit trail."""
    csv_path = DOWNLOADS / "STATIS_Test1_Continuous.csv"
    df = pd.read_csv(csv_path)

    pid = _create_project(client, auth_headers, "Audit Trail Test")
    upload = _upload_dataset(client, auth_headers, pid, csv_path)
    did = upload["id"]

    # Create and execute an analysis to generate audit entries
    data_dictionary = {
        "n_rows": len(df),
        "variables": [
            {"name": "patient_id", "dtype": "object", "n_unique": len(df), "n_missing": 0, "n_total": len(df)},
            {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_missing": 0, "n_total": len(df)},
            {"name": "systolic_bp", "dtype": "float64", "n_unique": len(df), "n_missing": 0, "n_total": len(df)},
        ],
    }
    r = client.post("/api/analysis-plan/create", json={
        "project_id": pid,
        "dataset_id": did,
        "research_question": "Does treatment affect systolic blood pressure?",
        "study_design": "cohort",
        "data_dictionary": data_dictionary,
        "data_quality": {"n_rows": len(df), "n_cols": 3},
    }, headers=auth_headers)
    assert r.status_code == 200
    plan = r.json()
    plan_id = plan["plan_id"]

    ttest_analysis = None
    for a in plan["analyses"]:
        if a["method"] == "independent_t_test":
            ttest_analysis = a
            break
    analysis_id = ttest_analysis["analysis_id"]

    client.post(f"/api/analysis-plan/{analysis_id}/approve", json={"plan_id": plan_id}, headers=auth_headers)
    client.post(f"/api/analysis-plan/{analysis_id}/lock", json={"plan_id": plan_id}, headers=auth_headers)
    client.post(f"/api/analysis-plan/{analysis_id}/execute", json={"plan_id": plan_id}, headers=auth_headers)

    # Check audit entries exist (global audit, not project-scoped)
    r = client.get("/api/audit", headers=auth_headers)
    assert r.status_code == 200
    audit = r.json()
    assert len(audit) > 0, "Expected at least one audit entry"

    # Verify audit entries have required fields
    for entry in audit:
        assert "id" in entry
        assert "timestamp" in entry
        assert "action" in entry

    # Verify expected actions are present
    actions = [e["action"] for e in audit]
    assert "analysis_plan_created" in actions, "No analysis_plan_created audit entry"
    assert "analysis_approved" in actions, "No analysis_approved audit entry"
    assert "analysis_locked" in actions, "No analysis_locked audit entry"
    assert "analysis_executed" in actions, "No analysis_executed audit entry"


# ============================================================================
# Test 8: Persistence after restart
# ============================================================================

def test_phase2k_persistence(client, auth_headers):
    """Verify that data persists after database close/reinitialization."""
    import asyncio
    from app.core.database import close_db, init_db

    csv_path = DOWNLOADS / "STATIS_Test1_Continuous.csv"
    df = pd.read_csv(csv_path)

    pid = _create_project(client, auth_headers, "Persistence Test")
    upload = _upload_dataset(client, auth_headers, pid, csv_path)
    did = upload["id"]

    # Verify data exists
    r = client.get(f"/api/projects/{pid}/datasets", headers=auth_headers)
    datasets = r.json()
    assert len(datasets) == 1

    # Close and reinitialize database
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(close_db())
    loop.run_until_complete(init_db())
    loop.close()

    # Verify data still exists after restart
    r = client.get(f"/api/projects/{pid}/datasets", headers=auth_headers)
    datasets = r.json()
    assert len(datasets) == 1, "Dataset lost after restart"
    assert datasets[0]["id"] == did, "Dataset ID changed after restart"
    assert datasets[0]["content_hash"] == upload["content_hash"], "Dataset hash changed after restart"


# ============================================================================
# Test 9: Provenance chain
# ============================================================================

def test_phase2k_provenance(client, auth_headers):
    """Verify the dataset → analysis → result → output provenance chain."""
    csv_path = DOWNLOADS / "STATIS_Test1_Continuous.csv"
    df = pd.read_csv(csv_path)

    pid = _create_project(client, auth_headers, "Provenance Test")
    upload = _upload_dataset(client, auth_headers, pid, csv_path)
    did = upload["id"]
    content_hash = upload["content_hash"]

    data_dictionary = {
        "n_rows": len(df),
        "variables": [
            {"name": "patient_id", "dtype": "object", "n_unique": len(df), "n_missing": 0, "n_total": len(df)},
            {"name": "treatment", "dtype": "int64", "n_unique": 2, "n_missing": 0, "n_total": len(df)},
            {"name": "systolic_bp", "dtype": "float64", "n_unique": len(df), "n_missing": 0, "n_total": len(df)},
        ],
    }
    r = client.post("/api/analysis-plan/create", json={
        "project_id": pid,
        "dataset_id": did,
        "research_question": "Does treatment affect systolic blood pressure?",
        "study_design": "cohort",
        "data_dictionary": data_dictionary,
        "data_quality": {"n_rows": len(df), "n_cols": 3},
    }, headers=auth_headers)
    assert r.status_code == 200
    plan = r.json()
    plan_id = plan["plan_id"]

    ttest_analysis = None
    for a in plan["analyses"]:
        if a["method"] == "independent_t_test":
            ttest_analysis = a
            break
    analysis_id = ttest_analysis["analysis_id"]

    client.post(f"/api/analysis-plan/{analysis_id}/approve", json={"plan_id": plan_id}, headers=auth_headers)
    client.post(f"/api/analysis-plan/{analysis_id}/lock", json={"plan_id": plan_id}, headers=auth_headers)

    r = client.post(f"/api/analysis-plan/{analysis_id}/execute", json={"plan_id": plan_id}, headers=auth_headers)
    assert r.status_code == 200

    # Verify run record has provenance
    r = client.get(f"/api/runs?project_id={pid}", headers=auth_headers)
    runs = r.json()
    assert len(runs) == 1
    run = runs[0]
    assert run["dataset_id"] == did, "Run dataset_id mismatch"
    assert run["dataset_content_hash"] == content_hash, "Run dataset hash mismatch"
    assert run["input_fingerprint"] is not None, "Missing input fingerprint"
    assert run["result_fingerprint"] is not None, "Missing result fingerprint"
    assert run["decision_trace"] is not None, "Missing decision trace"

    # Verify decision trace has required events
    trace = run["decision_trace"]
    assert len(trace) > 0, "Empty decision trace"
    # Check that trace entries have the expected structure
    for entry in trace:
        assert "timestamp" in entry, "Missing timestamp in trace entry"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
