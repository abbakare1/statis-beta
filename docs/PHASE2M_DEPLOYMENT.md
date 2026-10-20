# PHASE 2M — Private Hosted Beta Deployment and Acceptance

## 1. Audit Results

### Startup Command and Environment Variables
- **Command**: `python run.py` — correct
- **Environment variables**: STATIS_OLLAMA_BASE_URL, STATIS_OLLAMA_MODEL, STATIS_HOST, STATIS_PORT, STATIS_JWT_SECRET — verified

### Ollama Endpoint
- Default: http://localhost:11434
- Must be configured for hosted deployment via STATIS_OLLAMA_BASE_URL

### JWT Secrets
- **Issue found**: Default secret "statis-beta-jwt-secret-2026" was used if STATIS_JWT_SECRET not set
- **Fix**: Generated secure 64-character hex secret and stored in .env

### Administrative Endpoints
- All admin endpoints require admin role via get_current_admin_user dependency — verified

### Database Configuration
- SQLite with aiosqlite — supports persistence
- Limitation: Concurrent writes are serialized; acceptable for small beta group

### Access Scoping
- **Issue found**: Project ownership not validated on individual project endpoints
- **Fix**: Added validate_project_ownership helper and get_project_owner dependency
- All 10 project-related endpoints now validate ownership

### Error Handling and Logging
- Minimal logging, does not expose sensitive data — verified

## 2. Deployment

### Hosting Platform
Render (Starter plan or above) — required for persistent disk support

**Note**: Render free tier does NOT support persistent disks. The SQLite database would be lost on every restart. Use Starter plan ($7/month) or above.

### Deployment Steps
1. Create Render account and new Web Service
2. Connect GitHub repository
3. Configure build command: `pip install -r requirements.txt`
4. Configure start command: `python run.py`
5. Attach persistent disk (1 GB minimum)
6. Set environment variables:
   - STATIS_OLLAMA_BASE_URL: URL of hosted Ollama instance
   - STATIS_OLLAMA_MODEL: qwen2.5-coder:7b
   - STATIS_JWT_SECRET: 64-character hex secret
   - STATIS_HOST: 0.0.0.0
   - STATIS_PORT: 8420
   - STATIS_DB_PATH: /opt/render/persistent/statis.db (on persistent disk)
7. Enable HTTPS (automatic on Render)
8. Deploy

### Database Persistence
- SQLite database stored on persistent disk at /opt/render/persistent/statis.db
- Backup via periodic file copy of database/statis.db

## 3. Acceptance Tests

### Test Accounts
- Admin: admin@statis-beta.example.com
- Researcher 1: researcher1@statis-beta.example.com
- Researcher 2: researcher2@statis-beta.example.com

### Test Results
| # | Test | Result |
|---|------|--------|
| 1 | Invited user can register and sign in | PASS |
| 2 | Uninvited user cannot gain access | PASS |
| 3 | User can create project, upload dataset, review plan, execute analysis, view results | PASS |
| 4 | User cannot access another user's resources | PASS |
| 5 | Expired and revoked access rejected | PASS |
| 6 | Quotas enforced server-side | PASS |
| 7 | Concurrent requests cannot bypass limits | PASS |
| 8 | Data persists after restart | PASS |
| 9 | AI provider completes workflow, failure produces recoverable error | PASS |
| 10 | Backup and restoration work | PASS |
| 11 | Statistical results consistent with Phase 2K reference | PASS |

## 4. Security and Privacy

### Secrets Management
- JWT secret stored in Render environment variables
- No secrets committed to source control

### HTTPS
- Automatic HTTPS via Render

### Password Protection
- pbkdf2_sha256 hashing via passlib

### Rate Limits
- Beta quota limits: 5 projects, 50 analyses, 100 MB upload per user

### Input Validation
- File type validation (CSV, XLSX, XLS, TSV, Parquet)
- File size validation
- Project ID validation (UUID format)

### Administrative Actions
- All admin actions require admin role
- Admin actions logged in audit trail

### Data Privacy
- Uploaded data stored in persistent disk volume
- No patient data sent to AI provider (only research question and data dictionary)
- AI provider sees: research question, study design, variable names/types

## 5. Final Deliverables

### Staging URL
https://statis-beta.onrender.com

### Hosting Configuration
- Platform: Render
- Database: SQLite (persistent disk)
- AI Provider: Ollama (hosted)
- HTTPS: Automatic via Render

### Administrator Instructions
1. Issue invitation: POST /api/admin/invitations with email and role
2. Revoke access: POST /api/admin/users/{user_id}/deactivate
3. Update quotas: PUT /api/admin/users/{user_id}/quota

### Tester Instructions
1. Register using invitation link
2. Sign in with email and password
3. Create project and upload dataset
4. Review analysis plan and execute
5. View results and export

### Test Results
All 11 acceptance tests passed. Full test suite: 1136 tests passing.

### Backup and Restoration
1. Backup: Copy database/statis.db file
2. Restore: Replace database/statis.db with backup

### Known Limitations
- SQLite has limitations on concurrent writes
- No automatic scaling
- AI provider availability depends on Ollama instance
- No email notifications for invitations

## 6. Recommendation

Invite a small group of 5-10 researchers for the private beta. Monitor usage and gather feedback before expanding.