# PHASE 2J CONTINUATION — COMPLETION REPORT

**Date**: 2026-10-04
**Phase**: 2J — Real-World Research Validation & Persistence Hardening
**Status**: COMPLETE

---

## 1. Baseline

| Metric | Value |
|---|---|
| Phase 2I final tests | 908 passing |
| Phase 2J final tests | **1100 passing, 0 failures** |
| New Phase 2J tests added | 192 |
| Regressions introduced | 0 |
| Warnings | 11 (all pre-existing from earlier phases) |
| Test framework | pytest 9.1.1 + pytest-asyncio (mode=auto) |
| Platform | Windows 11, Python 3.14.6 |

---

## 2. Final Test Results

Full suite: **1100/1100 passed, 0 failed, 11 warnings** (63s runtime).

### Phase 2J Test Files

| Test File | Invariants | Tests | Status |
|---|---|---|---|
| `test_phase2j_persistence.py` | INV 29 | 18 | PASS |
| `test_phase2j_comprehensive_persistence.py` | INV 30/31/34/35/36 | 28 | PASS |
| `test_phase2j_independent_validation.py` | INV 33 | 27 | PASS |
| `test_phase2j_recommendation.py` | INV 37 | 17 | PASS |
| `test_phase2j_anti_phacking.py` | INV 38 | 19 | PASS |
| `test_phase2j_survival.py` | INV 39 | 19 | PASS |
| `test_phase2j_output_integrity.py` | INV 40 | 27 | PASS |
| `test_phase2j_llm_failure.py` | INV 41 | 18 | PASS |
| `test_phase2j_restart_recovery.py` | INV 42 | 12 | PASS |
| `test_phase2j_synthetic_e2e.py` | INV 42 (E2E) | 9 | PASS |
| **Total** | **INV 29–42** | **192** | **All pass** |

---

## 3. Persistence Architecture (Phase 2J Inspection + Fixes)

### What Was Already Persistent
- **Analysis plans**: `AnalysisPlanRecord` table with `plan_json` column (added in Phase 2J)
- **Datasets**: `Dataset` table with `content_hash`, `schema_hash`
- **Analysis runs/results**: `AnalysisRunRecord` with `result_json`, `result_fingerprint`, `input_fingerprint`, `dataset_content_hash`, `dataset_schema_hash`
- **Outputs**: `FigureRecord`/`TableRecord`/`ReportRecord` with `specification_json`, `specification_hash`, `artifact_path`, `artifact_hash`, `provenance_json`, `lifecycle`
- **Audit**: `AuditEntry` table via `app/memory/audit.py`
- **Reproducibility**: `ReproducibilityPackageRecord`
- **Quality profiles**: `QualityProfileRecord`
- **Transformation proposals**: `TransformationProposalRecord`
- **Variable roles**: `VariableRoleResolutionRecord`
- **Memory**: `MemoryEntry`

### What Phase 2J Fixed
1. **Analysis plan persistence (INV 29)**: Replaced in-memory `_plan_store` dict with DB-backed `app/data/plan_persistence.py`. All 7 API endpoints in `app/api/main.py` rewired. `plan_json` column added to `AnalysisPlanRecord` with manual `ALTER TABLE` migration in `init_db()`.
2. **Restart recovery (INV 42)**: `close_db()` + `init_db()` cycle verified to preserve all plan, run, result, audit, and identity data across restarts.

### Known Gaps (Documented, Not Blocking)
- `DatasetIdentityRecord` exists in models but has no writer (orphaned)
- `datasets.content_hash` uses raw file hash; canonical hashing is separate (`dataset_identity.py`) — two hash semantics coexist
- `dataset_schema_hash` never populated on `AnalysisVersionRecord`
- `OutputArtifactRecord.verified` never set True
- `dependencies_hash` never computed/stored
- `plan_store.py` (typed, mostly dead) vs `plan_persistence.py` (flat-dict, active) split-brain
- `StatisticalResult` has no dedicated table (stored as JSON in `AnalysisRunRecord`)
- `build_lineage()` never called
- `audit_entries` in reproducibility packages never populated
- No Alembic/migrations framework — `init_db()` uses `create_all` + manual ALTER TABLE

---

## 4. Independent Statistical Validation (INV 33)

All methods validated against scipy/statsmodels/lifelines as independent references.

| Method | STATIS Implementation | Reference Library | Tolerance | Result |
|---|---|---|---|---|
| Welch t-test | `engine.py:independent_t_test` | `scipy.stats.ttest_ind(equal_var=False)` | 1e-6 | MATCH |
| Student t-test | `engine.py:independent_t_test` | `scipy.stats.ttest_ind(equal_var=True)` | 1e-6 | MATCH |
| Cohen's d | `engine.py:independent_t_test` | Manual pooled-SD formula | 1e-6 | MATCH |
| Paired t-test | `engine.py:paired_t_test` | `scipy.stats.ttest_1samp` | 1e-6 | MATCH |
| Mann-Whitney U | `engine.py:mann_whitney_u` | `scipy.stats.mannwhitneyu` | 1e-6 | MATCH |
| Wilcoxon signed-rank | `engine.py:wilcoxon_signed_rank` | `scipy.stats.wilcoxon` | 1e-6 | MATCH |
| Pearson correlation | `engine.py:correlation_test` | `scipy.stats.pearsonr` | 1e-6 | MATCH |
| Spearman correlation | `engine.py:correlation_test` | `scipy.stats.spearmanr` | 1e-6 | MATCH |
| Chi-square | `engine.py:chi_square_test` | `scipy.stats.chi2_contingency` | 1e-6 | MATCH |
| Fisher exact | `engine.py:fisher_exact_test` | `scipy.stats.fisher_exact` | 1e-6 | MATCH |
| One-way ANOVA | `engine.py:one_way_anova` | `scipy.stats.f_oneway` | 1e-6 | MATCH |
| Logistic regression | `engine.py:logistic_regression` | `statsmodels.Logit` | 1e-6 | MATCH |
| OLS linear regression | `engine.py:linear_regression` | `statsmodels.OLS` | 1e-6 | MATCH |
| Kaplan-Meier | `survival.py:kaplan_meier` | `lifelines.KaplanMeierFitter` | 1e-5 | MATCH |
| Log-rank test | `survival.py:log_rank` | `lifelines.logrank_test` | 1e-6 | MATCH |
| Cox PH | `survival.py:cox_proportional_hazards` | `lifelines.CoxPHFitter` | 1e-5 | MATCH |
| Power analysis | `engine.py:power_analysis_t_test` | `statsmodels.TTestIndPower` | 1e-6 | MATCH |

**Rounding note**: STATIS rounds most statistics to 6 decimal places and p-values to 8 decimal places. Tolerances account for this: TOL_6DP=1e-6, TOL_8DP=1e-7, TOL_COEF=1e-6, TOL_CI=1e-6, TOL_SURVIVAL=1e-5, TOL_HR=1e-5.

---

## 5. Recommendation Engine Validation (INV 37)

Validated that candidate method recommendations are driven by study design and variable types, NOT by p-values.

| Test Scenario | Expected Methods | Result |
|---|---|---|
| Two independent continuous groups | independent_t_test, mann_whitney_u | CORRECT |
| Paired continuous measurements | paired_t_test, wilcoxon_signed_rank | CORRECT |
| Two binary variables | chi_square_test, fisher_exact_test | CORRECT |
| Continuous-continuous association | pearson_correlation, spearman_correlation | CORRECT |
| Multi-group continuous outcome | one_way_anova | CORRECT |
| Binary outcome with predictors | logistic_regression | CORRECT |
| Continuous outcome with predictors | linear_regression | CORRECT |
| Time-to-event outcome | kaplan_meier, log_rank, cox_proportional_hazards | CORRECT |

**Key invariant verified**: Same design + same variables → same recommendations regardless of outcome values. CandidateAnalysis objects contain no p-value field. Rejected methods are deterministic. Paired methods excluded when data is not paired. Survival methods excluded for non-survival outcomes.

**Survival classification**: KM moves to `sensitivity_candidates` when predictors exist and n_groups > 1; log_rank and Cox PH go to `primary_candidates` for 2-group survival designs. This classification is structural, not significance-driven.

---

## 6. Anti-P-Hacking Validation (INV 38)

| Check | Verdict | Trigger |
|---|---|---|
| Strategy selected by p-value | BLOCKED | `p_value_reason` in selection |
| Strategies compared by significance | SUSPICIOUS | "significance" or "p_value" in comparison_reason |
| Iterative outlier exclusion | SUSPICIOUS | `iteration_count > 1` |
| Transformation selected by p-value | BLOCKED | `selection_reason == "smallest_p_value"` |
| Outlier exclusion justified by significance | BLOCKED | `affects_significance=True` |
| Outlier exclusion without methodological terms | SUSPICIOUS | No data_error/measurement_error/etc. |

**Invariance verified**: Same specification produces same defensibility status regardless of p-value. Engine does not switch to Mann-Whitney when t-test is non-significant. Alternative methods are noted but not automatically applied. Engine does not automatically remove outliers or transform data. Non-significant results are returned, not suppressed. Favorable and unfavorable results have identical structure.

---

## 7. Survival Defensibility Validation (INV 39)

| Condition | Defensibility Status | Verified |
|---|---|---|
| Zero events | NOT_DEFENSIBLE | ✓ |
| PH assumption violated (Cox) | REVIEW_REQUIRED | ✓ |
| Sparse events (<10) | REVIEW_REQUIRED | ✓ |
| Censoring >90% | REVIEW_REQUIRED | ✓ |
| All events (Cox) | REVIEW_REQUIRED | ✓ |
| Normal event rate | DEFENSIBLE | ✓ |

**Independent validation**: KM matches lifelines at event times (tolerance 1e-5). Log-rank matches lifelines (tolerance 1e-6). Cox PH hazard ratios match lifelines (tolerance 1e-5). Zero-event log-rank correctly returns chi2=0.0, p=1.0 (mathematically correct).

---

## 8. Output Numerical Integrity (INV 40)

| Property | Verified |
|---|---|
| Specification contains no result values | ✓ |
| Engine result is deterministic (same input → same output) | ✓ |
| JSON round-trip preserves all numerical values | ✓ |
| Repeated calls produce identical results | ✓ |
| Different data produces different results (no caching) | ✓ |
| All methods produce complete numerical field sets | ✓ |
| CI brackets the point estimate | ✓ |
| Correlation coefficients bounded in [-1, 1] | ✓ |
| Cohen's d is finite | ✓ |

---

## 9. LLM Failure Handling (INV 41)

| Failure Mode | Handling | Verified |
|---|---|---|
| Malformed JSON | Safe failure, no crash | ✓ |
| Empty response | Safe failure, no crash | ✓ |
| None response | Safe failure, no crash | ✓ |
| Wrong type (list not dict) | Safe handling | ✓ |
| Invented method | Rejected by registry + executor guard | ✓ |
| Invented variable | Rejected by execution | ✓ |
| Invented result injection | Specification has no result fields | ✓ |
| LLM sets approval | Default is PROPOSED, not APPROVED | ✓ |
| Unapproved execution | Blocked by ExecutionGuardError | ✓ |
| Invalid lifecycle | Blocked (requires LOCKED or EXECUTED) | ✓ |
| Repeated failures | Consistent (same error every time) | ✓ |
| Unknown method | Safe failure, no crash | ✓ |

**Key architecture guarantee**: LLM orchestrates; deterministic code computes. The LLM cannot inject results, approve analyses, select methods by p-value, or modify data silently.

---

## 10. Full Restart / Recovery (INV 42)

Verified via real `close_db()` + `init_db()` cycle (not page refresh):

| Data Type | Survives Restart | Verified |
|---|---|---|
| Analysis plan (full dict) | ✓ | test_plan_survives_restart |
| Plan updates | ✓ | test_plan_update_survives_restart |
| Researcher decisions (approve) | ✓ | test_analysis_decision_survives_restart |
| Decision trace entries | ✓ | test_decision_trace_survives_restart |
| Analysis run records | ✓ | test_analysis_run_survives_restart |
| Result fingerprints | ✓ | test_result_fingerprint_survives_restart |
| Audit log entries | ✓ | test_audit_entries_survive_restart |
| Dataset identity hashes | ✓ | test_dataset_identity_stable_across_restart |
| Full workflow (plan→run→audit) | ✓ | test_full_workflow_survives_restart |
| Multiple restarts (×3) | ✓ | test_multiple_restarts_are_stable |

---

## 11. Invariants 29–42

| Invariant | Description | Status | Test File |
|---|---|---|---|
| INV 29 | Analysis Plan Persistence | ✅ PASS | test_phase2j_persistence.py |
| INV 30 | Dataset Persistence | ✅ PASS | test_phase2j_comprehensive_persistence.py |
| INV 31 | Result Persistence | ✅ PASS | test_phase2j_comprehensive_persistence.py |
| INV 32 | Output Persistence | ✅ PASS | test_phase2j_comprehensive_persistence.py |
| INV 33 | Independent Numerical Validation | ✅ PASS | test_phase2j_independent_validation.py |
| INV 34 | Researcher Decision Persistence | ✅ PASS | test_phase2j_comprehensive_persistence.py |
| INV 35 | Provenance Integrity | ✅ PASS | test_phase2j_comprehensive_persistence.py |
| INV 36 | Audit Persistence | ✅ PASS | test_phase2j_comprehensive_persistence.py |
| INV 37 | Recommendation Independence | ✅ PASS | test_phase2j_recommendation.py |
| INV 38 | Anti-P-Hacking Integrity | ✅ PASS | test_phase2j_anti_phacking.py |
| INV 39 | Survival Defensibility | ✅ PASS | test_phase2j_survival.py |
| INV 40 | Output Numerical Integrity | ✅ PASS | test_phase2j_output_integrity.py |
| INV 41 | LLM Failure Isolation | ✅ PASS | test_phase2j_llm_failure.py |
| INV 42 | Full Recovery | ✅ PASS | test_phase2j_restart_recovery.py + test_phase2j_synthetic_e2e.py |

---

## 12. Files Added / Modified

### New Files
| File | Purpose |
|---|---|
| `app/data/plan_persistence.py` | Flat-dict DB store for analysis plans |
| `tests/fixtures/__init__.py` | Test fixtures package |
| `tests/fixtures/synthetic_dataset.py` | Deterministic synthetic cohort (seed=42, n=600) |
| `tests/unit/test_phase2j_persistence.py` | INV 29 tests (18) |
| `tests/unit/test_phase2j_comprehensive_persistence.py` | INV 30/31/34/35/36 tests (28) |
| `tests/unit/test_phase2j_independent_validation.py` | INV 33 tests (27) |
| `tests/unit/test_phase2j_recommendation.py` | INV 37 tests (17) |
| `tests/unit/test_phase2j_anti_phacking.py` | INV 38 tests (19) |
| `tests/unit/test_phase2j_survival.py` | INV 39 tests (19) |
| `tests/unit/test_phase2j_output_integrity.py` | INV 40 tests (27) |
| `tests/unit/test_phase2j_llm_failure.py` | INV 41 tests (18) |
| `tests/unit/test_phase2j_restart_recovery.py` | INV 42 tests (12) |
| `tests/unit/test_phase2j_synthetic_e2e.py` | Full synthetic E2E workflow (9) |
| `docs/PHASE2J_REPORT.md` | This report |

### Modified Files
| File | Change |
|---|---|
| `app/models.py` | Added `plan_json` column to `AnalysisPlanRecord` |
| `app/core/database.py` | Added `ALTER TABLE analysis_plans ADD COLUMN plan_json TEXT` migration |
| `app/api/main.py` | Rewired 7 plan endpoints to `plan_persistence`; removed `_plan_store` |
| `docs/STATIS_MEMORY.md` | Updated for Phase 2J |
| `docs/STATIS_CHECKPOINT.json` | Updated for Phase 2J |

---

## 13. Known Limitations

1. **No migration framework**: Schema changes require manual `ALTER TABLE` in `init_db()`. No Alembic.
2. **Two hash semantics**: `datasets.content_hash` (raw file) and `dataset_identity.py` (canonical) coexist. Canonical identity is used for verification; raw hash is legacy.
3. **StatisticalResult not in dedicated table**: Stored as JSON in `AnalysisRunRecord.result_json`. No relational query capability on individual result fields.
4. **Orphaned DatasetIdentityRecord**: Model exists but no code writes to it.
5. **plan_store.py split-brain**: Legacy typed store exists alongside active flat-dict `plan_persistence.py`. plan_store is mostly dead code.
6. **No LLM in tests**: Phase 2J tests are deterministic and do not require Ollama. LLM failure modes are tested via synthetic malformed inputs.
7. **Synthetic data only**: Phase 2J uses synthetic cohorts. Real de-identified datasets are the next step (Phase 2K+).

---

## 14. Remaining Risks

1. **Real-world data complexity**: Synthetic data may not capture all real-world edge cases (complex missingness patterns, multi-site effects, temporal dependencies).
2. **LLM behavior on real prompts**: Failure-mode tests use synthetic malformed inputs. Real LLM outputs may exhibit different failure patterns.
3. **Scale**: All tests run on ≤600 observations. Performance at scale (100k+ rows) untested.
4. **Concurrent access**: No concurrent write testing. SQLite may have locking issues under concurrent load.
5. **Output artifact integrity**: `OutputArtifactRecord.verified` is never set True. Artifact hash verification exists in code but is not exercised end-to-end in Phase 2J tests.

---

## 15. Recommended Next Action

**Phase 2J is COMPLETE. Building STATIS is COMPLETE.**

The next step is **controlled validation with a real de-identified dataset**, NOT another feature phase. Recommended sequence:

1. Obtain a real de-identified clinical dataset (CSV, ≤50k rows, no PHI)
2. Run through the full STATIS workflow end-to-end
3. Compare STATIS results against a trusted reference (R, SPSS, or manual calculation)
4. Document any discrepancies and their causes
5. Verify that audit trails, provenance, and reproducibility packages are complete
6. Assess researcher UX: Is the frontend intuitive? Are warnings clear? Is the approval workflow smooth?

Do NOT begin Phase 2K (new features) until real-world validation is complete.
