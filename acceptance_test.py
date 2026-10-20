import sys
sys.path.insert(0, '.')
import asyncio
import uuid
from fastapi.testclient import TestClient
from app.api.main import app
from app.security.auth import create_user, create_access_token
from app.core.database import get_session

client = TestClient(app)

# Create test user
async def setup_user():
    async with get_session() as session:
        user = await create_user(session, f"acceptance_{uuid.uuid4().hex[:8]}@example.com", "password123", "Acceptance Test User")
        user.max_projects = 100
        user.max_analyses = 1000
        user.max_upload_mb = 1000
        await session.commit()
        return user

user = asyncio.run(setup_user())
token = create_access_token(user.id, user.role)
auth_headers = {"Authorization": f"Bearer {token}"}

# 1. Create project
r = client.post('/api/projects', data={
    'name': 'Acceptance Test',
    'description': 'Testing workflow',
    'research_question': 'Does treatment affect outcome?',
    'study_design': 'cohort'
}, headers=auth_headers)
print(f'Create project: {r.status_code}')
if r.status_code != 200:
    print(f'  Error: {r.text}')
    sys.exit(1)
project_id = r.json()['id']

# 2. Upload dataset
csv_path = 'C:/Users/ajibo/Downloads/STATIS_Test1_Continuous.csv'
with open(csv_path, 'rb') as f:
    r = client.post(f'/api/projects/{project_id}/upload', files={'file': ('test.csv', f, 'text/csv')}, headers=auth_headers)
print(f'Upload dataset: {r.status_code}')
if r.status_code != 200:
    print(f'  Error: {r.text}')
    sys.exit(1)
dataset = r.json()
dataset_id = dataset['id']
print(f"  Rows: {dataset['row_count']}, Cols: {dataset['column_count']}")

# 3. Data quality assessment
r = client.post(f'/api/data-quality/{dataset_id}/assess', headers=auth_headers)
print(f'Data quality: {r.status_code}')

# 4. Create analysis plan
r = client.post('/api/analysis-plan/create', json={
    'project_id': project_id,
    'dataset_id': dataset_id,
    'research_question': 'Does treatment affect systolic blood pressure?',
    'study_design': 'cohort',
    'data_dictionary': {
        'n_rows': 120,
        'variables': [
            {'name': 'patient_id', 'dtype': 'object', 'n_unique': 120, 'n_missing': 0, 'n_total': 120},
            {'name': 'treatment', 'dtype': 'int64', 'n_unique': 2, 'n_missing': 0, 'n_total': 120},
            {'name': 'systolic_bp', 'dtype': 'float64', 'n_unique': 120, 'n_missing': 0, 'n_total': 120},
        ]
    },
    'data_quality': {'n_rows': 120, 'n_cols': 3}
}, headers=auth_headers)
print(f'Create plan: {r.status_code}')
if r.status_code != 200:
    print(f'  Error: {r.text}')
    sys.exit(1)
plan = r.json()
plan_id = plan['plan_id']
print(f"  Analyses: {len(plan['analyses'])}")

# 5. Find t-test analysis
ttest = None
for a in plan['analyses']:
    if a['method'] == 'independent_t_test':
        ttest = a
        break
if ttest:
    analysis_id = ttest['analysis_id']
    # Approve
    r = client.post(f'/api/analysis-plan/{analysis_id}/approve', json={'plan_id': plan_id}, headers=auth_headers)
    print(f'Approve: {r.status_code}')
    # Lock
    r = client.post(f'/api/analysis-plan/{analysis_id}/lock', json={'plan_id': plan_id}, headers=auth_headers)
    print(f'Lock: {r.status_code}')
    # Execute
    r = client.post(f'/api/analysis-plan/{analysis_id}/execute', json={'plan_id': plan_id}, headers=auth_headers)
    print(f'Execute: {r.status_code}')
    if r.status_code == 200:
        result = r.json()
        print(f"  Success: {result.get('success')}")
        if result.get('success'):
            data = result.get('data', {})
            print(f"  t-statistic: {data.get('t_statistic')}")
            print(f"  p-value: {data.get('p_value')}")
else:
    print('No t-test analysis found')

# 6. Check audit trail
r = client.get('/api/audit', headers=auth_headers)
print(f"Audit entries: {len(r.json())}")

# 7. Check runs
r = client.get(f'/api/runs?project_id={project_id}', headers=auth_headers)
print(f"Runs: {len(r.json())}")

print("\n=== ACCEPTANCE TEST COMPLETE ===")
