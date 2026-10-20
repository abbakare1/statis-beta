# STATIS — Canonical Project Memory

**Last updated**: 2026-10-16
**Current phase**: 2K (COMPLETE — controlled real-world validation)
**Test status**: 1120/1120 passing, 0 failures

> **Authority hierarchy**: Source code > Database/schema > Tests > Machine manifests > This file > Conversation history.
> If this file conflicts with the repository, inspect the repository and update this file.

> **Phase 2J status**: Building STATIS is COMPLETE (see `docs/PHASE2J_REPORT.md`). Phase 2K is controlled validation with real datasets — NOT another feature phase.
>
> **Phase 2K progress**: Dataset upload/preview UX complete (`tests/unit/test_upload_ux.py`, 18 tests; endpoints in `app/api/main.py`). Remaining: E2E validation runs on `Downloads/STATIS_Test{1..4}*.csv`, independent numeric cross-check, audit/provenance/reproducibility verification, `docs/PHASE2K_VALIDATION.md`.

---

## 1. Project Identity

| Field | Value |
|---|---|
| Name | STATIS — Scientific Statistical Analysis and Research Intelligence Platform |
| Purpose | AI-assisted statistical analysis platform with deterministic execution, audit trails, and defensibility guarantees |
| Users | Researchers, biostatisticians, data analysts |
| Python | 3.14 |
| Web framework | FastAPI on `127.0.0.1:8420` |
| Database | SQLite via async SQLAlchemy (`database/statis.db`) |
| LLM | Ollama local inference (default: `qwen2.5-coder:7b`) |
| Statistical libraries | scipy, statsmodels, lifelines, pingouin, scikit-learn |
| Test framework | pytest with pytest-asyncio |
| Platform | Windows 11 |

---

## 2. Product Vision — End-to-End Workflow

```
Research Question          ✓ implemented (agents/research_question.py)
        ↓
Study Design               ✓ implemented (agents/study_design.py, variable_roles.py)
        ↓
Data Ingestion             ✓ implemented (agents/data_ingestion.py)
        ↓
Data Dictionary            ✓ implemented (agents/data_dictionary.py)
        ↓
Data Quality               ✓ implemented (data_quality.py, data/data_quality_store.py)
        ↓
Missingness                ✓ implemented (missing_data.py)
        ↓
Outlier Detection          ✓ implemented (outlier_engine.py)
        ↓
Transformations            ✓ implemented (transformations.py)
        ↓
Variable Roles             ✓ implemented (variable_roles.py)
        ↓
Power / Information        ✓ implemented (agents/power_sample_size.py)
        ↓
Candidate Analyses         ✓ implemented (candidate_generator.py)
        ↓
Analysis Specification     ✓ implemented (specification.py)
        ↓
Validation                 ✓ implemented (validator.py)
        ↓
Researcher Approval        ✓ implemented (plan.py lifecycle)
        ↓
Deterministic Execution    ✓ implemented (executor.py, engine.py)
        ↓
Diagnostics                ✓ implemented (method_diagnostics.py, diagnostics.py)
        ↓
Defensibility              ✓ implemented (defensibility.py)
        ↓
Tables / Figures           ✓ implemented (visualization/, tables/ — Phase 2G)
        ↓
Reports                    ✓ implemented (reporting/ — Phase 2G)
        ↓
Reproducibility Package    ✓ implemented (reproducibility/ — Phase 2G)
        ↓
Output API / Lifecycle     ✓ implemented (api/routers/, outputs/service.py — Phase 2H)
```

---

## 3. Current Architecture — Module Map

### Core Infrastructure

| Module | File | Purpose |
|---|---|---|
| FastAPI app | `app/api/main.py` | REST API, 1007 lines |
| Database engine | `app/core/database.py` | SQLAlchemy async engine, init/close |
| Model provider | `app/core/model_provider.py` | Abstract LLM interface |
| Ollama adapter | `app/core/ollama_adapter.py` | Concrete LLM adapter |
| Ollama client | `app/core/ollama_client.py` | HTTP client for Ollama |
| Configuration | `config/settings.py` | Pydantic settings, paths, server config |
| Entry point | `run.py` | uvicorn launcher |
| Smoke test | `smoke_test.py` | End-to-end component verification |

### Statistical Engine (25 files in `app/statistics/`)

| Module | File | Lines | Purpose |
|---|---|---|---|
| Engine | `engine.py` | 695 | Deterministic computation layer (scipy/statsmodels) |
| Registry | `registry.py` | 440 | 15 method metadata entries |
| Candidate generator | `candidate_generator.py` | 511 | Registry-driven method selection |
| Executor | `executor.py` | 1013 | Guardrailed execution engine |
| Specification | `specification.py` | 156 | Formal analysis contract |
| Validator | `validator.py` | 355 | Pre-execution validation |
| Plan | `plan.py` | 259 | Lifecycle-managed analysis plans |
| Variable roles | `variable_roles.py` | 857 | RQ-aware variable classification |
| Study design state | `study_design_state.py` | 85 | Study design state machine |
| Survival | `survival.py` | 671 | KM, log-rank, Cox PH (lifelines) |
| Data quality | `data_quality.py` | 279 | Dataset quality profiling |
| Missing data | `missing_data.py` | 402 | Missingness analysis |
| Outlier engine | `outlier_engine.py` | 480 | Outlier detection (IQR/Z/MAD) |
| Transformations | `transformations.py` | 516 | Parameterized data transforms |
| Diagnostics | `diagnostics.py` | 147 | Diagnostic framework types |
| Method diagnostics | `method_diagnostics.py` | 1765 | Per-method assumption checks |
| Defensibility | `defensibility.py` | 122 | Methodological defensibility |
| Anti-p-hacking | `anti_phacking.py` | 184 | P-hacking prevention |
| Causal guard | `causal_guard.py` | 278 | Causal language classification |
| Result model | `result.py` | 175 | StatisticalResult contract |
| Decision trace | `decision_trace.py` | 247 | Immutable audit trails |
| Reproducibility | `reproducibility.py` | 155 | Reproducibility verification |
| Results comparator | `results_comparator.py` | 181 | Cross-specification comparison |
| Revalidation | `revalidation.py` | 85 | Post-transform revalidation |

### Publication Output Layer (Phase 2G + 2H)

| Module | File | Purpose |
|---|---|---|
| Output specifications | `app/outputs/specification.py` | Figure/Table/Report specs, validation, lifecycle |
| Output provenance | `app/outputs/provenance.py` | Artifact hashing, lineage tracking |
| Output lifecycle | `app/outputs/lifecycle.py` | State machine, transition validation |
| Output service | `app/outputs/service.py` | Orchestration: validation → render → persist → audit |
| Figure renderer | `app/visualization/renderer.py` | Deterministic matplotlib rendering |
| Table renderer | `app/tables/renderer.py` | Deterministic table rendering |
| Report renderer | `app/reporting/renderer.py` | Structured report generation |
| Reproducibility | `app/reproducibility/manifest.py` | Package creation, manifest, verification |
| API schemas | `app/api/schemas.py` | Pydantic request/response contracts |
| API routers | `app/api/routers/outputs.py` | FastAPI routes for figures/tables/reports/reproducibility |

### Agents (8 agents in `app/agents/`)

| Agent | File | Purpose |
|---|---|---|
| Base | `base.py` | Common agent interface |
| Data ingestion | `data_ingestion.py` | CSV/XLSX loading, type detection |
| Data dictionary | `data_dictionary.py` | Auto-construct data dictionary |
| Data quality | `data_quality.py` | Comprehensive quality assessment |
| Research question | `research_question.py` | RQ parsing via LLM |
| Study design | `study_design.py` | Study design classification via LLM |
| Power/sample size | `power_sample_size.py` | Dataset adequacy evaluation |
| Analysis planner | `analysis_planner.py` | Two-stage analysis planning |
| Statistical executor | `statistical_executor.py` | Analysis execution agent |

### Data Layer

| Module | File | Purpose |
|---|---|---|
| Dataset identity | `data/dataset_identity.py` | SHA-256 hashing for tabular data |
| Hashing utils | `data/hashing.py` | Hash/version utilities |
| Quality store | `data/data_quality_store.py` | Persistent quality profiles |
| Plan store | `data/plan_store.py` | Analysis plan CRUD (legacy, mostly dead) |
| Plan persistence | `data/plan_persistence.py` | Flat-dict DB store for analysis plans (Phase 2J, active) |
| Run store | `data/run_store.py` | Analysis run persistence |
| Audit | `memory/audit.py` | Immutable audit ledger |
| Memory store | `memory/store.py` | Persistent memory system |
| Orchestrator | `orchestration/orchestrator.py` | Agent pipeline routing |

### Database Models (`app/models.py`)

20 SQLAlchemy ORM models (see Section 8 for full list).

### Frontend Architecture (Phase 2I)

| Module | File | Purpose |
|---|---|---|
| API service layer | `frontend/static/js/api.js` | Centralized fetch, error handling, service modules |
| State management | `frontend/static/js/state.js` | Subscription-based state, loading/error/notification |
| Router | `frontend/static/js/router.js` | Hash-based routing with deep links |
| UI components | `frontend/static/js/components.js` | Reusable accessible components (badges, cards, states) |
| Views | `frontend/static/js/views.js` | Dashboard, Project workspace, global views |
| App entry | `frontend/static/js/app.js` | Initialization, navigation, keyboard shortcuts |
| Template | `frontend/templates/index.html` | SPA layout with sidebar navigation |
| Styles | `frontend/static/css/style.css` | Dark theme, responsive layout, accessibility |

**Frontend invariants**:
- Frontend never performs statistical computation
- Backend state is authoritative over frontend state
- Frontend cannot independently approve analyses or outputs
- Displayed workflow status corresponds to actual backend state
- Provenance navigation preserves dataset → analysis → result → output chain
- Frontend actions that mutate state generate backend audit events
- Frontend cannot silently modify source datasets or specifications

---

## 4. Test Suite

### Test Files

| File | Purpose | Approx. tests |
|---|---|---|
| `test_statistics_engine.py` | Core engine functions | ~80 |
| `test_validation_battery.py` | Validation battery (73 cases) | 73 |
| `test_phase2c.py` | Phase 2C candidate generation | ~30 |
| `test_phase2d.py` | Phase 2D data quality | ~40 |
| `test_phase2d1.py` | Phase 2D.1 persistence | ~30 |
| `test_phase2d2.py` | Phase 2D.2 variable roles | ~50 |
| `test_phase2e.py` | Phase 2E deterministic execution | ~40 |
| `test_phase2e_engine.py` | Phase 2E engine functions | ~30 |
| `test_phase2e1.py` | Phase 2E.1 diagnostics | ~30 |
| `test_phase2e2_regression.py` | Phase 2E.2 regression (65 cases) | 65 |
| `test_phase2f_survival.py` | Phase 2F survival (74 cases) | 74 |
| `test_phase2g_outputs.py` | Phase 2G outputs (72 cases) | 72 |
| `test_phase2h_api.py` | Phase 2H output API (72 cases) | 72 |
| `test_phase2i_frontend.py` | Phase 2I frontend integration (75 cases) | 75 |
| `test_phase2j_persistence.py` | Phase 2J plan persistence (INV 29) | 18 |
| `test_phase2j_comprehensive_persistence.py` | Phase 2J persistence (INV 30/31/34/35/36) | 28 |
| `test_phase2j_independent_validation.py` | Phase 2J independent validation (INV 33) | 27 |
| `test_phase2j_recommendation.py` | Phase 2J recommendation (INV 37) | 17 |
| `test_phase2j_anti_phacking.py` | Phase 2J anti-p-hacking (INV 38) | 19 |
| `test_phase2j_survival.py` | Phase 2J survival defensibility (INV 39) | 19 |
| `test_phase2j_output_integrity.py` | Phase 2J output integrity (INV 40) | 27 |
| `test_phase2j_llm_failure.py` | Phase 2J LLM failure isolation (INV 41) | 18 |
| `test_phase2j_restart_recovery.py` | Phase 2J restart/recovery (INV 42) | 12 |
| `test_phase2j_synthetic_e2e.py` | Phase 2J synthetic E2E workflow | 9 |
| `test_specification_validation.py` | Specification validation | ~20 |
| `test_hardening.py` | Hardening tests | ~20 |
| `test_model_provider.py` | Model provider tests | ~10 |
| `test_upload_ux.py` | Dataset upload/preview UX regression (Phase 2K) | 18 |
| `test_phase2k_validation.py` | Real-dataset E2E validation (Phase 2K) | 9 |

**Total**: 1120 tests, 0 failures (per-file exact counts from `pytest --collect-only`; see `docs/STATIS_CHECKPOINT.json`) — 29 test files

### Phase 2J Test Fixtures

| File | Purpose |
|---|---|
| `tests/fixtures/synthetic_dataset.py` | Deterministic synthetic cohort (seed=42, n=600, 30 columns, data dictionary, ground truth) |

### Test Infrastructure

- `tests/conftest.py`: DB init, canonical fixture hash tracking, `isolated_copy` factory
- Canonical test data: `data/uploads/228683e7-01c9-4050-afde-4d17cdb99e3f_STATIS_Test1_Continuous.csv` (120 patients, seed=42)
- All tests use `tmp_path` for isolation; canonical data is never modified

---

## 5. Statistical Methods Registry

### Registered Methods (15 total)

| # | method_id | Name | Engine | Software |
|---|-----------|------|--------|----------|
| 1 | `independent_t_test` | Independent samples t-test | `independent_t_test` | scipy.stats |
| 2 | `paired_t_test` | Paired samples t-test | `paired_t_test` | scipy.stats |
| 3 | `mann_whitney_u` | Mann-Whitney U test | `mann_whitney_u` | scipy.stats |
| 4 | `wilcoxon_signed_rank` | Wilcoxon signed-rank test | `wilcoxon_signed_rank` | scipy.stats |
| 5 | `pearson_correlation` | Pearson correlation | `correlation_test` | scipy.stats |
| 6 | `spearman_correlation` | Spearman rank correlation | `correlation_test` | scipy.stats |
| 7 | `chi_square` | Chi-square test of independence | `chi_square_test` | scipy.stats |
| 8 | `fisher_exact` | Fisher's exact test | `fisher_exact_test` | scipy.stats |
| 9 | `one_way_anova` | One-way ANOVA | `one_way_anova` | scipy + statsmodels |
| 10 | `linear_regression` | OLS Linear regression | `linear_regression` | statsmodels |
| 11 | `logistic_regression` | Logistic regression | `logistic_regression` | statsmodels |
| 12 | `descriptive` | Descriptive statistics | `descriptive_statistics` | numpy/pandas/scipy |
| 13 | `kaplan_meier` | Kaplan-Meier estimator | `kaplan_meier` | lifelines |
| 14 | `log_rank` | Log-rank test | `log_rank` | lifelines |
| 15 | `cox_proportional_hazards` | Cox proportional hazards regression | `cox_proportional_hazards` | lifelines |

### Method Capabilities by Phase

| Capability | Status |
|---|---|
| Execution | ✓ All 15 methods |
| Validation | ✓ All 15 methods |
| Diagnostics | ✓ All 15 methods |
| Defensibility | ✓ All 15 methods (survival-specific checks added in 2F) |
| Candidate generation | ✓ All 15 methods |
| Anti-p-hacking | ✓ All methods |

---

## 6. Invariants (Never Violate)

1. **Deterministic execution**: All statistical computation comes from `engine.py` or `survival.py`. No LLM computes statistics.
2. **Dataset identity**: Every execution verifies dataset hash before computing. Hash mismatch → BLOCKED.
3. **Locked analyses**: A locked `AnalysisVersionRecord` cannot be modified. Attempting modification creates a new version.
4. **No p-value-based method selection**: STATIS never selects the method with the smallest p-value.
5. **No causal language from observational data**: `causal_guard.py` blocks causal claims from non-experimental designs.
6. **No automatic outlier removal**: Outliers are detected and characterized, never silently removed.
7. **No automatic imputation**: Missing data is analyzed, not filled.
8. **Canonical test data immutability**: `tests/conftest.py` tracks hash of canonical fixture; tests must use `isolated_copy`.
9. **Identifier protection**: Patient/record identifiers are never assigned outcome, exposure, survival_time, or event_indicator roles.
10. **Audit trail**: Every analytical decision is captured in the audit ledger.
11. **Output immutability**: Publication outputs never recompute statistics or modify analysis results.
12. **No LLM numerics**: AI never generates numerical results in outputs.
13. **Output traceability**: Every output is traceable to dataset identity + analysis version + result fingerprint.
14. **PHI protection**: Default outputs exclude raw patient-level data.
15. **Analytical status preserved**: Outputs distinguish primary/secondary/sensitivity/exploratory analyses.
16. **API routes cannot bypass output validation**: Rendering always revalidates using the same Phase 2G validators; API routes contain no statistical logic.
17. **Approved and archived outputs are immutable**: Once APPROVED or ARCHIVED, specification, provenance, and artifacts cannot be modified. Changes require creating a new output version.
18. **Rendering always revalidates dataset and analysis identity**: Every render verifies dataset content/schema hash and analysis version binding before generating artifacts.
19. **Artifact retrieval cannot access arbitrary filesystem paths**: All artifact paths are resolved within the controlled `outputs/` directory; path traversal is blocked.
20. **API-generated outputs preserve provenance and audit history**: Every create/validate/render/approve/archive operation writes to the audit ledger with full lineage.
21. **Equivalent output requests are idempotent or explicitly versioned**: Repeated renders of the same (dataset identity + analysis version + specification) return the existing artifact rather than creating duplicates.
22. **Frontend numerical integrity**: The frontend never computes or modifies statistical results. All numerical values are displayed exactly as returned by the backend.
23. **Backend authority**: Backend state is authoritative over frontend state for analysis lifecycle, approval, datasets, outputs, and provenance.
24. **Researcher approval**: The frontend cannot independently approve an analysis or output. Approval always invokes the backend lifecycle.
25. **Workflow state integrity**: Displayed workflow status corresponds to actual backend state. Frontend never fabricates completion percentages or status.
26. **Provenance preservation**: Frontend navigation and presentation preserve dataset → analysis → result → output provenance. Provenance relationships come from actual backend records.
27. **Audit preservation**: Frontend actions that mutate application state generate the appropriate backend audit events.
28. **No silent data modification**: Frontend interactions cannot silently modify source datasets or analytical specifications.
29. **Analysis plan persistence**: Plans persist to the database via `plan_persistence.py` and survive application restart.
30. **Dataset persistence**: Dataset records and identity hashes persist across restart.
31. **Result persistence**: Analysis run results, fingerprints, and input fingerprints persist across restart.
32. **Output persistence**: Figures, tables, reports, and reproducibility packages persist across restart.
33. **Independent numerical validation**: All statistical methods match scipy/statsmodels/lifelines within documented tolerances.
34. **Researcher decision persistence**: Approve/reject decisions and decision traces persist across restart.
35. **Provenance integrity**: Dataset identity → analysis → result → output chain is traceable and persists.
36. **Audit persistence**: Audit log entries survive application restart.
37. **Recommendation independence**: Candidate method recommendations are driven by study design and variable types, never by p-values.
38. **Anti-p-hacking integrity**: STATIS does not manipulate data, methods, exclusions, or classifications to improve significance.
39. **Survival defensibility**: Zero events → NOT_DEFENSIBLE; PH violation → REVIEW_REQUIRED; sparse events <10 → REVIEW_REQUIRED; censoring >90% → REVIEW_REQUIRED.
40. **Output numerical integrity**: Stored results are consumed, not recomputed. JSON round-trip preserves all values. Engine is deterministic.
41. **LLM failure isolation**: LLM failures (timeout, malformed JSON, invalid method, invented variable/result) cause safe failure without data corruption.
42. **Full recovery**: Complete workflow (plan → run → result → audit → identity) survives application restart via real `close_db()` + `init_db()` cycle.

---

## 7. Database Schema

**Location**: `database/statis.db` (SQLite, created on first run)
**Engine**: `app/core/database.py` — async SQLAlchemy with aiosqlite

### Tables (20 models in `app/models.py`)

| Table | Model | Purpose | Key Fields |
|---|---|---|---|
| `audit_ledger` | `AuditEntry` | Immutable audit trail | id, timestamp, action, agent, decision, rationale |
| `projects` | `Project` | Research projects | id, name, research_question, study_design, status |
| `datasets` | `Dataset` | Uploaded datasets | id, project_id, content_hash, version, row_count |
| `analysis_specs` | `AnalysisSpec` | Analysis specifications | id, project_id, method, outcome, predictors, status |
| `analysis_results` | `AnalysisResult` | Computed results | id, spec_id, result_json, warnings |
| `memory` | `MemoryEntry` | Persistent memory | id, project_id, key, value, confidence |
| `analysis_plans` | `AnalysisPlanRecord` | Versioned plans | id, project_id, version, research_question |
| `analysis_entries` | `AnalysisEntryRecord` | Individual analyses | id, plan_id, analysis_name, prespecified_status |
| `analysis_versions` | `AnalysisVersionRecord` | Versioned specs | id, analysis_id, version_number, approval_status, lifecycle |
| `dataset_identities` | `DatasetIdentityRecord` | Identity chain | id, dataset_id, content_hash, schema_hash, parent_identity_id |
| `transformation_records` | `TransformationRecord` | Transformation lineage | id, operation, parameters, output_content_hash |
| `quality_profiles` | `QualityProfileRecord` | Data quality | id, dataset_id, profile_json |
| `transformation_proposals` | `TransformationProposalRecord` | Transform lifecycle | id, operation, status, reviewer, anti_p_hacking_verdict |
| `analysis_runs` | `AnalysisRunRecord` | Reproducible runs | id, input_fingerprint, result_fingerprint, success |
| `variable_role_resolutions` | `VariableRoleResolutionRecord` | Variable roles | id, variable_name, proposed_role, confidence |
| `figure_records` | `FigureRecord` | Publication figures (Phase 2G) | id, figure_type, specification_json, artifact_hash, lifecycle |
| `table_records` | `TableRecord` | Publication tables (Phase 2G) | id, table_type, specification_json, artifact_hash, lifecycle |
| `report_records` | `ReportRecord` | Publication reports (Phase 2G) | id, title, specification_json, artifact_hash, lifecycle |
| `output_artifacts` | `OutputArtifactRecord` | Generated artifacts (Phase 2G) | id, output_type, artifact_hash, dataset_content_hash |
| `reproducibility_packages` | `ReproducibilityPackageRecord` | Repro packages (Phase 2G) | id, package_id, manifest_json, manifest_hash |

---

## 8. Persistence Audit

| Artifact | Persisted? | Location | Recoverable? |
|---|---|---|---|
| Original datasets | ✓ | `data/uploads/` + `datasets` table | ✓ |
| Dataset identity/hash | ✓ | `dataset_identities` table | ✓ |
| Dataset versions | ✓ | `dataset_identities` (parent chain) | ✓ |
| Transformed datasets | ✓ | `data/working/` + `transformation_records` | ✓ |
| Transformation parameters | ✓ | `transformation_records.parameters` | ✓ |
| Quality profiles | ✓ | `quality_profiles` table | ✓ |
| Missingness assessments | ✓ | In quality profile JSON | ✓ |
| Outlier findings | ✓ | In quality profile JSON | ✓ |
| Researcher outlier decisions | ✓ | `transformation_proposals` table | ✓ |
| Variable role resolutions | ✓ | `variable_role_resolutions` table | ✓ |
| Research question | ✓ | `projects.research_question` | ✓ |
| Study design | ✓ | `projects.study_design` | ✓ |
| Candidate analyses | ✓ | In agent memory / plan JSON | ✓ |
| Analysis specifications | ✓ | `analysis_specs` + `analysis_versions` | ✓ |
| Validation results | ✓ | `analysis_versions.validation_*` | ✓ |
| Approvals | ✓ | `analysis_versions.approval_status` | ✓ |
| Locked analyses | ✓ | `analysis_versions.lifecycle` | ✓ |
| Executions | ✓ | `analysis_runs` table | ✓ |
| Statistical results | ✓ | `analysis_results` + `analysis_runs.result_json` | ✓ |
| Diagnostics | ✓ | In `analysis_runs.result_json.diagnostic_summary` | ✓ |
| Defensibility | ✓ | In `analysis_runs.result_json.defensibility_status` | ✓ |
| Audit events | ✓ | `audit_ledger` table | ✓ |
| Model/AI outputs | ✗ | In-memory only (not persisted after request) | ✗ |
| Researcher decisions (in-session) | Partial | In-memory during session, audit ledger after | Partial |
| Canonical test fixture hash | ✓ | `tests/conftest.py` (session-scoped) | ✓ per session |
| LLM model downloads | ✗ | Ollama local cache (external to STATIS) | ✗ |

---

## 9. Phase History

### Phase 1 — Foundation
- **Status**: COMPLETE
- **Purpose**: Core platform setup
- **Changes**: FastAPI app, SQLAlchemy models, basic agents, data ingestion, Ollama adapter
- **Tests**: ~50

### Phase 2A — Model Provider Architecture
- **Status**: COMPLETE
- **Purpose**: Abstract LLM interface
- **Changes**: `model_provider.py`, `ollama_adapter.py`, `ollama_client.py`
- **Tests**: ~10

### Phase 2B — Statistical Specifications & Registry
- **Status**: COMPLETE
- **Purpose**: Method registry, specification model, validation
- **Changes**: `registry.py`, `specification.py`, `validator.py`, 12 methods registered
- **Tests**: ~30

### Phase 2C — Candidate Generation & Analysis Plans
- **Status**: COMPLETE
- **Purpose**: Deterministic candidate generation, analysis plan lifecycle
- **Changes**: `candidate_generator.py`, `plan.py`, `causal_guard.py`, `anti_phacking.py`
- **Tests**: ~30

### Phase 2C Hardening — Dataset Identity & Persistence
- **Status**: COMPLETE
- **Purpose**: Dataset identity, persistence, executor protections
- **Changes**: `dataset_identity.py`, `plan_store.py`, `run_store.py`, hash verification in executor
- **Tests**: ~20

### Phase 2D — Data Quality, Missingness, Outliers, Transformations
- **Status**: COMPLETE
- **Purpose**: Data quality pipeline
- **Changes**: `data_quality.py`, `missing_data.py`, `outlier_engine.py`, `transformations.py`, `data_quality_store.py`
- **Tests**: ~40

### Phase 2D.1 — Persistence & Lifecycle Hardening
- **Status**: COMPLETE
- **Purpose**: Persistence layer, lineage, idempotency
- **Changes**: `TransformationProposalRecord`, `QualityProfileRecord`, `TransformationRecord` models
- **Tests**: ~30

### Phase 2D.2 — Variable Role Resolution
- **Status**: COMPLETE
- **Purpose**: Research-question-aware variable classification
- **Changes**: `variable_roles.py`, `study_design_state.py`, `data_dictionary.py` agent
- **Tests**: ~50

### Phase 2E — Deterministic Statistical Execution
- **Status**: COMPLETE
- **Purpose**: Guardrailed execution engine
- **Changes**: `executor.py`, `engine.py` enhancements, `result.py` model
- **Tests**: ~40

### Phase 2E.1 — Diagnostics & Defensibility
- **Status**: COMPLETE
- **Purpose**: Method-specific diagnostics, defensibility assessment
- **Changes**: `method_diagnostics.py`, `diagnostics.py`, `defensibility.py`, `reproducibility.py`, `decision_trace.py`
- **Tests**: ~30

### Phase 2E Validation Battery
- **Status**: COMPLETE
- **Purpose**: Validation integrity testing
- **Changes**: `test_validation_battery.py` (73 tests), discovered and fixed defects
- **Tests**: 73

### Phase 2E.2 — Validation Integrity & Execution Guard Hardening
- **Status**: COMPLETE
- **Purpose**: Fix 12 identified defects, harden guards
- **Changes**: `test_phase2e2_regression.py` (65 tests), fixes to variable_roles.py, conftest.py, test paths
- **Tests**: 65 (all 615 total passing at completion)

### Phase 2F — Survival Analysis Foundation
- **Status**: COMPLETE
- **Purpose**: Add survival analysis (KM, log-rank, Cox PH)
- **Changes**: `survival.py` (created), `variable_roles.py` (SURVIVAL_TIME, EVENT_INDICATOR roles), `registry.py` (3 survival methods), `executor.py` (survival dispatch), `method_diagnostics.py` (survival diagnostics), `candidate_generator.py` (survival-aware classification), `defensibility.py` (survival metadata), `test_phase2f_survival.py` (74 tests)
- **Tests**: 682 total (+67 from 615 baseline)

### Phase 2G — Publication Outputs, Visualization, Reporting, and Reproducibility
- **Status**: COMPLETE
- **Purpose**: Build publication output layer for figures, tables, reports, reproducibility packages
- **Changes**: `app/outputs/` (specification, provenance), `app/visualization/` (figure renderer), `app/tables/` (table renderer), `app/reporting/` (report renderer), `app/reproducibility/` (manifest, package), 5 new DB models (FigureRecord, TableRecord, ReportRecord, OutputArtifactRecord, ReproducibilityPackageRecord), `test_phase2g_outputs.py` (72 tests)
- **Tests**: 754 total (+72 from 682 baseline)

### Phase 2H — Output API Integration, Lifecycle Workflows, Persistence, and API Hardening
- **Status**: COMPLETE
- **Purpose**: Wire Phase 2G output subsystem into FastAPI with controlled lifecycle management
- **Changes**: `app/outputs/lifecycle.py` (state machine), `app/outputs/service.py` (orchestration service), `app/api/schemas.py` (Pydantic contracts), `app/api/routers/outputs.py` (32 endpoints for figures/tables/reports/reproducibility), routers registered in `main.py`, `test_phase2h_api.py` (72 tests)
- **Tests**: 826 total (+72 from 754 baseline)

### Phase 2I — Researcher-Facing Frontend Integration & User Experience
- **Status**: COMPLETE (see `docs/PHASE2I_REPORT.md`)
- **Purpose**: SPA frontend for researchers (dashboard, project workspace, provenance navigation)
- **Changes**: `frontend/static/js/{api,state,router,components,views,app}.js`, `frontend/templates/index.html`, `frontend/static/css/style.css`, `test_phase2i_frontend.py` (75 tests)
- **Tests**: 908 total (+82 from 826 baseline)

### Phase 2J — Real-World Research Validation & Persistence Hardening
- **Status**: COMPLETE (see `docs/PHASE2J_REPORT.md`)
- **Purpose**: Persistence hardening + independent numerical validation (INV 29–42)
- **Changes**: `app/data/plan_persistence.py` (DB-backed plans), `plan_json` column + manual `ALTER TABLE` migration in `init_db()`, 7 plan endpoints rewired off `_plan_store`, `tests/fixtures/synthetic_dataset.py`, 10 new test files (192 tests)
- **Tests**: 1100 reported at completion

### Phase 2K — Controlled Real-World Validation (COMPLETE)
- **Status**: COMPLETE — all validation objectives met
- **Purpose**: Validate STATIS end-to-end on real datasets, cross-check numbers independently, verify audit/provenance/reproducibility, assess researcher UX
- **Completed**: Dataset upload/preview UX (18 tests), E2E validation on 4 real datasets (Test1_Continuous, Test2_Binary, Test3_Association, Test4_Confounding), independent numeric cross-check against scipy/statsmodels, audit trail verification, dataset identity verification, reproducibility verification, persistence after restart, provenance chain verification, `docs/PHASE2K_VALIDATION.md` report
- **Bugs found + fixed**:
  - `config/settings.py:_base()` had one too many `.parent` (2026-10-08) — all runtime paths resolved to `C:\Users\ajibo` instead of project root
  - `app/statistics/engine.py:255` — chi-square test applied Yates' continuity correction by default; fixed with explicit `correction=False`
  - `tests/unit/test_phase2j_independent_validation.py:393` — Phase 2J test also expected Yates' correction; fixed to match
  - `tests/unit/test_phase2k_validation.py:439` — test expected "coefficients" key but STATIS returns "parameters"
  - `tests/unit/test_phase2k_validation.py:584` — audit trail test incomplete; updated to include full analysis workflow
- **Defect found + fixed (2026-10-08)**: `config/settings.py:_base()` had one too many `.parent` — all runtime paths resolved to `C:\Users\ajibo` instead of the project root. Fixed `_base()` → `.parent.parent`; migrated `data/`, `database/`, `outputs/` from the profile root into `STATIS/`; rewrote 1149 absolute paths in the DB (backup: `database/statis.db.pre2k.bak`); restored the canonical fixture to `data/uploads/` so the INV 8 hash check is active again; isolated tests from real data (see INV 43); fixed `test_phase2i_frontend.py` writing 3 CSVs into the project `data/uploads` (now `tmp_path`)
- **Tests**: 1120 total (9 Phase 2K validation tests + 18 upload-UX tests added to 1093 baseline)

---

## 10. Known Limitations & Deferred Work

### Not Implemented
- Competing risks analysis
- Time-varying covariates
- Frailty models
- Parametric survival (Weibull, exponential)
- SMR analysis
- Report generation via weasyprint (HTML/Markdown works; PDF generation deferred)
- Interactive figure editing
- Multi-format export bundles
- Multi-user authentication
- CI/CD pipeline

### Known Technical Debt
- `models.py` has 20 models in a single file — should be split
- Some agent files have LLM calls that could be mocked more thoroughly in tests
- `method_diagnostics.py` is 1765 lines — could benefit from decomposition
- No CI/CD pipeline configured
- No `.env` file committed (settings use defaults)

---

## 11. How to Resume Development

### Quick Start
```bash
cd C:\Users\ajibo\STATIS
python -m pytest tests/unit/ -q     # Verify all 1111 tests pass
python run.py                        # Start the server
python smoke_test.py                 # Verify components
```

### Before Starting New Work
1. Run `python -m pytest tests/unit/ -q` — confirm 1111 passing
2. Read this memory file for context
3. Inspect actual source files before assuming behavior
4. Check `docs/PHASE2H_REPORT.md` for output API details

### After Completing Work

**Mandatory completion checklist (all three are required before a phase is considered complete):**

1. Run full test suite — must remain at 0 failures
2. Update this memory file (`docs/STATIS_MEMORY.md`) — reflect any architecture changes, new methods, new invariants, updated test counts
3. Update `docs/STATIS_CHECKPOINT.json` — refresh test counts, phase status, registered methods, deferred work

**Recovery test**: Whenever the architecture changes substantially (new modules, new database tables, new statistical methods, changed invariants), the recovery test must be rerun:

```bash
# Verify tests still pass
python -m pytest tests/unit/ -q

# Verify smoke test passes
python smoke_test.py

# Verify checkpoint is consistent with repository
# (test count in STATIS_CHECKPOINT.json matches pytest output)
```

If the recovery test fails, fix the inconsistency before proceeding.

---

## 12. File Authoritativeness

| File | Role |
|---|---|
| `app/statistics/*.py` | Source of truth for statistical logic |
| `app/models.py` | Source of truth for database schema |
| `app/statistics/registry.py` | Source of truth for method registry |
| `tests/unit/test_*.py` | Source of truth for what works |
| `docs/STATIS_MEMORY.md` | This file — navigation aid, not source of truth |
| `docs/STATIS_CHECKPOINT.json` | Machine-readable state manifest |
| `docs/PHASE2F_REPORT.md` | Phase 2F detailed report |
| `docs/PHASE2G_REPORT.md` | Phase 2G detailed report |
| `docs/PHASE2H_REPORT.md` | Phase 2H detailed report |
| `requirements.txt` | Python dependencies |
| `config/settings.py` | Application configuration |
| `tests/conftest.py` | Test infrastructure and fixtures |
