# STATIS Hosted Beta Deployment Guide

## Prerequisites
- Python 3.14+
- Ollama instance (local or hosted)
- SQLite database (bundled)

## Configuration
Edit `.env` file:
```
STATIS_OLLAMA_BASE_URL=http://your-ollama-instance:11434
STATIS_OLLAMA_MODEL=qwen2.5-coder:7b
STATIS_HOST=0.0.0.0
STATIS_PORT=8420
STATIS_JWT_SECRET=<64-character-hex-secret>
```

Generate a secure JWT secret:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## Start Server
```bash
pip install -r requirements.txt
python run.py
```

## Access
Open http://localhost:8420 in your browser.

## Beta Tester Accounts
Create invitations via the admin API:
```bash
curl -X POST http://localhost:8420/api/admin/invitations \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","role":"researcher"}'
```

## Quota Management
Update user quotas:
```bash
curl -X PUT http://localhost:8420/api/admin/users/{user_id}/quota \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"max_projects":5,"max_analyses":50,"max_upload_mb":100}'
```