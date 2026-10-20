# Phase 2H Report — Output API Integration, Lifecycle Workflows, Persistence, and API Hardening

**Date**: 2026-10-02
**Status**: COMPLETE
**Baseline**: 754 tests passing
**Final**: 826/826 tests passing, 0 failures
**New tests**: 72

---

## 1. Scope

Phase 2H wires the Phase 2G publication output subsystem into the FastAPI application, providing:

- Controlled lifecycle management for figures, tables, reports, and reproducibility packages
- REST API endpoints with explicit Pydantic schemas
- Service layer that orchestrates validation → rendering → persistence → provenance → audit
- Lifecycle state machine with enforced transitions
- Dataset identity and analysis version verification before rendering
- Artifact storage with path traversal protection
- Idempotent rendering
- Immutable approved/archived outputs
- Comprehensive adversarial/security testing

The API layer does **not** duplicate statistical logic. Routes call into the service layer, which delegates to Phase 2G components.

---

## 2. Architecture

```
FastAPI Route
    ↓
Request Parsing (Pydantic schemas)
    ↓
Dependency Injection (database session)
    ↓
Output Service (app/outputs/service.py)
    ↓
Phase 2G Specification + Validation
    ↓
Phase 2G Deterministic Renderer
    ↓
Artifact Persistence (controlled storage)
    ↓
Provenance (app/outputs/provenance.py)
    ↓
Audit Ledger (app/memory/audit.py)
    ↓
Lifecycle Transition (app/outputs/lifecycle.py)
```

### New Modules

| Module | File | Purpose |
|---|---|---|
| Lifecycle state machine | `app/outputs/lifecycle.py` | Transition validation, immutability rules |
| Output service | `app/outputs/service.py` | Orchestration: create/validate/render/approve/archive |
| API schemas | `app/api/schemas.py` | Pydantic request/response contracts |
| API routers | `app/api/routers/outputs.py` | FastAPI routes for all output types |

---

## 3. Output Service Layer

**Module**: `app/outputs/service.py`

The `OutputService` class coordinates all output operations:

- **create()**: Validates specification, checks dataset binding, persists record
- **validate()**: Uses Phase 2G validators, transitions lifecycle
- **render()**: Revalidates, verifies dataset identity, dispatches to Phase 2G renderer, persists artifact + provenance
- **get_artifact()**: Retrieves artifact, verifies hash integrity
- **approve()**: Explicit approval with timestamp and actor
- **archive()**: Terminal immutable state
- **list()**: Filtered listing by project, dataset, status, classification

The `ReproducibilityService` handles package creation, manifest retrieval, and verification using the Phase 2G `ReproducibilityPackage` implementation.

---

## 4. Figure API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/outputs/figures` | Create a new figure |
| GET | `/api/outputs/figures` | List figures with filters |
| GET | `/api/outputs/figures/{id}` | Get a specific figure |
| POST | `/api/outputs/figures/{id}/validate` | Validate specification |
| POST | `/api/outputs/figures/{id}/render` | Render to artifact |
| GET | `/api/outputs/figures/{id}/artifact` | Get artifact metadata + integrity |
| GET | `/api/outputs/figures/{id}/artifact/download` | Download artifact file |
| POST | `/api/outputs/figures/{id}/review` | Request review |
| POST | `/api/outputs/figures/{id}/approve` | Approve output |
| POST | `/api/outputs/figures/{id}/archive` | Archive output |

---

## 5. Table API Endpoints

Same pattern as figures:

| Method | Path | Description |
|---|---|---|
| POST | `/api/outputs/tables` | Create a new table |
| GET | `/api/outputs/tables` | List tables |
| GET | `/api/outputs/tables/{id}` | Get a specific table |
| POST | `/api/outputs/tables/{id}/validate` | Validate |
| POST | `/api/outputs/tables/{id}/render` | Render |
| GET | `/api/outputs/tables/{id}/artifact` | Artifact metadata |
| GET | `/api/outputs/tables/{id}/artifact/download` | Download |
| POST | `/api/outputs/tables/{id}/review` | Request review |
| POST | `/api/outputs/tables/{id}/approve` | Approve |
| POST | `/api/outputs/tables/{id}/archive` | Archive |

---

## 6. Report API Endpoints

Same pattern as figures/tables. Reports reference analysis versions, figures, and tables — they never create new statistical analyses.

---

## 7. Reproducibility API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/outputs/reproducibility` | Create a reproducibility package |
| GET | `/api/outputs/reproducibility/{package_id}` | Get package info |
| GET | `/api/outputs/reproducibility/{package_id}/manifest` | Get manifest |
| POST | `/api/outputs/reproducibility/{package_id}/verify` | Verify package |
| GET | `/api/outputs/reproducibility/{package_id}/artifact` | Get package contents |

---

## 8. Lifecycle Enforcement

The state machine in `app/outputs/lifecycle.py` enforces:

```
DRAFT → VALIDATING → VALIDATED → RENDERING → GENERATED
  → REVIEW_REQUIRED → APPROVED → ARCHIVED

VALIDATING → VALIDATION_FAILED → VALIDATING (revalidation)
RENDERING → RENDERING_FAILED → VALIDATING (must revalidate)
```

**Rejected transitions** (HTTP 409):
- Approve DRAFT
- Archive DRAFT
- Render ARCHIVED
- Validate APPROVED
- Render VALIDATION_FAILED without revalidation
- Modify APPROVED/ARCHIVED outputs

---

## 9. Dataset Identity Enforcement

Before rendering, the service:
1. Loads the dataset record
2. Recomputes content hash and schema hash from the stored file
3. Compares against the hash bound to the output
4. Blocks rendering on mismatch (HTTP 409, `DATASET_IDENTITY_MISMATCH`)

---

## 10. Analysis Version Enforcement

Every output references immutable analysis version/run IDs. If the referenced analysis is missing or deleted, rendering fails safely with `ANALYSIS_VERSION_NOT_FOUND`. The system never silently substitutes the latest analysis.

---

## 11. Provenance

Every rendered output produces a provenance chain:

```
Project → Dataset → DatasetIdentity → AnalysisVersion
→ StatisticalResult → OutputSpecification → OutputArtifact
```

Provenance includes:
- Artifact hash (SHA-256)
- Specification hash
- Dataset content/schema hashes
- Result fingerprint
- Environment metadata (Python version, platform)
- Output fingerprint (for idempotency)

---

## 12. Artifact Storage

Artifacts are stored in a controlled directory structure:

```
outputs/
├── figures/
├── tables/
├── reports/
└── reproducibility/
```

The database stores artifact metadata (path, hash, format, size). Binary content is stored on disk.

**Path security**: All artifact paths are resolved and verified to stay within the base directory. Path traversal attempts are blocked.

---

## 13. Security

### Error Codes

| Code | HTTP | Meaning |
|---|---|---|
| `OUTPUT_NOT_FOUND` | 404 | Output does not exist |
| `INVALID_OUTPUT_STATE` | 409 | Lifecycle state prevents operation |
| `VALIDATION_FAILED` | 422 | Specification validation failed |
| `DATASET_NOT_FOUND` | 404 | Referenced dataset missing |
| `DATASET_IDENTITY_MISMATCH` | 409 | Dataset hash changed |
| `ANALYSIS_VERSION_NOT_FOUND` | 404 | Referenced analysis missing |
| `ARTIFACT_NOT_FOUND` | 404 | Artifact file missing |
| `ARTIFACT_INTEGRITY_FAILURE` | 409 | Artifact hash mismatch |
| `APPROVAL_NOT_ALLOWED` | 409 | Invalid approval state |
| `ARCHIVE_NOT_ALLOWED` | 409 | Invalid archive state |
| `CROSS_PROJECT_BINDING` | 409 | Dataset belongs to different project |
| `PATH_TRAVERSAL_BLOCKED` | 400 | Artifact path escapes allowed directory |

### Safeguards

1. **Path traversal blocked**: Artifact paths resolved and verified within base directory
2. **Cross-project binding prevented**: Dataset project must match output project
3. **AI boundary intact**: No endpoint allows AI to generate numerical results or approve outputs
4. **No p-value selection**: No endpoint accepts significance-based filtering
5. **PHI protection**: Default outputs are aggregate; reproducibility packages exclude raw data
6. **Idempotent rendering**: Equivalent renders return existing artifacts

---

## 14. Testing

### Test Count
- **Baseline**: 754 tests
- **New Phase 2H tests**: 72
- **Final**: 826 tests, 0 failures

### Test Categories

| Category | Tests | Coverage |
|---|---|---|
| Figure API CRUD | 10 | create, get, list, filters, validate, render, artifact, download |
| Figure lifecycle | 6 | approve, archive, immutability, idempotency, identity mismatch |
| Table API | 8 | create, get, list, validate, render, artifact, lifecycle |
| Report API | 7 | create, get, validate, render, artifact, lifecycle |
| Reproducibility API | 8 | create, get, manifest, verify, artifact, PHI, duplicates |
| Lifecycle enforcement | 5 | invalid transitions, full flow, 409 responses |
| Provenance | 3 | presence, specification hash, integrity detection |
| Adversarial | 12 | path traversal, malformed specs, LLM injection, PHI leakage, concurrency |
| Lifecycle unit | 5 | state machine transitions, immutability, revalidation |

---

## 15. Database Compatibility

Phase 2H uses the existing Phase 2G models without schema changes. The 20 database tables are unchanged. Both fresh databases and existing Phase 2G databases work correctly.

---

## 16. Known Limitations

- No multi-user authentication (single-user/local architecture)
- No websocket real-time updates
- No batch export bundles
- Report PDF generation deferred (HTML/Markdown works)
- No interactive figure editing
- Concurrency handled via lifecycle state checks (no distributed locking)

---

## 17. Deferred Work

- Report PDF generation via weasyprint
- Competing risks analysis
- Time-varying covariates
- Frailty models
- Parametric survival models
- Interactive figure editing
- CI/CD pipeline
- Multi-user authentication
- Batch export bundles

---

## 18. API Usage Examples

### Create and Render a Figure

```bash
# Create
curl -X POST http://127.0.0.1:8420/api/outputs/figures \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "your-dataset-id",
    "figure_type": "histogram",
    "variable": "age",
    "title": "Age Distribution",
    "output_format": "png"
  }'

# Validate
curl -X POST http://127.0.0.1:8420/api/outputs/figures/{id}/validate

# Render
curl -X POST http://127.0.0.1:8420/api/outputs/figures/{id}/render

# Download artifact
curl http://127.0.0.1:8420/api/outputs/figures/{id}/artifact/download -o figure.png

# Request review
curl -X POST http://127.0.0.1:8420/api/outputs/figures/{id}/review

# Approve
curl -X POST http://127.0.0.1:8420/api/outputs/figures/{id}/approve \
  -H "Content-Type: application/json" \
  -d '{"approved_by": "dr_smith"}'

# Archive
curl -X POST http://127.0.0.1:8420/api/outputs/figures/{id}/archive
```

### List with Filters

```bash
curl "http://127.0.0.1:8420/api/outputs/figures?project_id=...&status=approved"
curl "http://127.0.0.1:8420/api/outputs/tables?analysis_version_id=..."
```

---

## 19. Recovery Instructions

### To Resume Development

```bash
cd C:\Users\ajibo\STATIS
python -m pytest tests/unit/ -q     # Verify 826 tests pass
python run.py                        # Start server
python smoke_test.py                 # Verify components
```

### Key Files for Phase 2H

- `app/outputs/lifecycle.py` — Lifecycle state machine
- `app/outputs/service.py` — Output orchestration service
- `app/api/schemas.py` — API request/response schemas
- `app/api/routers/outputs.py` — FastAPI routes
- `app/api/main.py` — Router registration
- `tests/unit/test_phase2h_api.py` — 72 tests

### Documentation

- `docs/STATIS_MEMORY.md` — Canonical project memory
- `docs/STATIS_CHECKPOINT.json` — Machine-readable state
- `docs/PHASE2G_REPORT.md` — Phase 2G detailed report
- `docs/PHASE2H_REPORT.md` — This report

---

## 20. New Invariants (16–21)

16. **API routes cannot bypass output validation**: Rendering always revalidates using Phase 2G validators.
17. **Approved and archived outputs are immutable**: Once APPROVED/ARCHIVED, outputs cannot be modified.
18. **Rendering always revalidates dataset and analysis identity**: Hash verification before every render.
19. **Artifact retrieval cannot access arbitrary filesystem paths**: Path traversal blocked at service level.
20. **API-generated outputs preserve provenance and audit history**: Every operation writes to audit ledger.
21. **Equivalent output requests are idempotent or explicitly versioned**: Repeated renders return existing artifacts.

---

*End of Phase 2H Report*
