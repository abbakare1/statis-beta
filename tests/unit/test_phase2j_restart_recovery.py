"""Phase 2J — Full Restart / Recovery (Invariant 42).

Validates that the full workflow survives a simulated application restart:
- analysis plan persists across restart
- dataset identity persists across restart
- analysis run and result persists across restart
- audit log persists across restart
- researcher decision persists across restart

Uses real re-initialization of database context, not a page refresh.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.database import get_engine, get_session, init_db, close_db
from app.data import plan_persistence
from app.data.run_store import (
    create_analysis_run, complete_analysis_run, get_analysis_run,
    compute_result_fingerprint,
)
from app.data.dataset_identity import compute_dataset_identity
from app.memory.audit import log_audit, get_audit_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _reinit_db():
    """Simulate a full restart: dispose engine, re-run init_db."""
    await close_db()
    await init_db()


# ===========================================================================
# Plan Persistence Across Restart
# ===========================================================================

class TestPlanRestartRecovery:
    async def test_plan_survives_restart(self):
        """Analysis plan persists across a full restart."""
        await init_db()
        plan_id = "restart-plan-001"
        plan = {
            "plan_id": plan_id,
            "project_id": "restart-proj",
            "research_question": "Does X affect Y?",
            "study_design": "observational",
            "analyses": [
                {
                    "analysis_id": "restart-analysis-001",
                    "method": "independent_t_test",
                    "outcome": "value",
                    "lifecycle": "draft",
                    "approval": "proposed",
                }
            ],
            "warnings": [],
            "rejected_methods": [],
        }
        await plan_persistence.save_plan(plan)

        # Simulate restart
        await _reinit_db()

        # Plan must still be loadable
        loaded = await plan_persistence.load_plan(plan_id)
        assert loaded is not None
        assert loaded["plan_id"] == plan_id
        assert loaded["research_question"] == "Does X affect Y?"
        assert len(loaded["analyses"]) == 1
        assert loaded["analyses"][0]["method"] == "independent_t_test"

    async def test_plan_update_survives_restart(self):
        """Plan update persists across restart."""
        await init_db()
        plan_id = "restart-plan-002"
        plan = {
            "plan_id": plan_id,
            "project_id": "restart-proj",
            "analyses": [
                {
                    "analysis_id": "restart-analysis-002",
                    "method": "independent_t_test",
                    "lifecycle": "draft",
                    "approval": "proposed",
                }
            ],
        }
        await plan_persistence.save_plan(plan)

        # Update plan fields
        await plan_persistence.update_plan(plan_id, {"research_question": "Updated question"})

        # Restart
        await _reinit_db()

        loaded = await plan_persistence.load_plan(plan_id)
        assert loaded["research_question"] == "Updated question"

    async def test_analysis_decision_survives_restart(self):
        """Researcher decision (approve/reject) persists across restart."""
        await init_db()
        plan_id = "restart-plan-003"
        analysis_id = "restart-analysis-003"
        plan = {
            "plan_id": plan_id,
            "project_id": "restart-proj",
            "analyses": [
                {
                    "analysis_id": analysis_id,
                    "method": "independent_t_test",
                    "lifecycle": "draft",
                    "approval": "proposed",
                }
            ],
        }
        await plan_persistence.save_plan(plan)

        # Researcher approves
        await plan_persistence.update_analysis(plan_id, analysis_id, {
            "approval": "approved",
            "lifecycle": "approved",
        })

        # Restart
        await _reinit_db()

        result = await plan_persistence.find_analysis(analysis_id)
        assert result is not None
        plan_loaded, analysis = result
        assert analysis["approval"] == "approved"
        assert analysis["lifecycle"] == "approved"

    async def test_decision_trace_survives_restart(self):
        """Decision trace entries persist across restart."""
        await init_db()
        plan_id = "restart-plan-004"
        analysis_id = "restart-analysis-004"
        plan = {
            "plan_id": plan_id,
            "project_id": "restart-proj",
            "analyses": [
                {
                    "analysis_id": analysis_id,
                    "method": "independent_t_test",
                    "lifecycle": "draft",
                    "approval": "proposed",
                }
            ],
        }
        await plan_persistence.save_plan(plan)

        # Record decisions
        await plan_persistence.update_analysis(plan_id, analysis_id, {
            "approval": "approved",
            "lifecycle": "locked",
            "decision": "approved_by_researcher",
            "decision_timestamp": "2026-10-01T10:00:00Z",
        })

        # Restart
        await _reinit_db()

        result = await plan_persistence.find_analysis(analysis_id)
        assert result is not None
        _, analysis = result
        assert analysis["decision"] == "approved_by_researcher"


# ===========================================================================
# Run / Result Persistence Across Restart
# ===========================================================================

class TestRunRestartRecovery:
    async def test_analysis_run_survives_restart(self):
        """Analysis run record persists across restart."""
        await init_db()
        spec = {"analysis_method": "independent_t_test", "primary_outcome": "value"}
        result_payload = {"t_statistic": 2.5, "p_value": 0.014}

        run_id = await create_analysis_run(
            project_id="restart-proj-001",
            analysis_type="independent_t_test",
            specification=spec,
            dataset_id="restart-ds-001",
            dataset_content_hash="abc123",
            dataset_schema_hash="schema456",
        )
        assert run_id is not None

        await complete_analysis_run(run_id, result=result_payload)

        # Restart
        await _reinit_db()

        loaded = await get_analysis_run(run_id)
        assert loaded is not None
        assert loaded["result"]["t_statistic"] == 2.5
        assert loaded["result"]["p_value"] == 0.014

    async def test_result_fingerprint_survives_restart(self):
        """Result fingerprint persists across restart (integrity check)."""
        await init_db()
        result_payload = {"t_statistic": 2.5, "p_value": 0.014}

        run_id = await create_analysis_run(
            project_id="restart-proj-002",
            analysis_type="independent_t_test",
            specification={"analysis_method": "independent_t_test"},
            dataset_id="restart-ds-002",
            dataset_content_hash="def456",
            dataset_schema_hash="schema789",
        )
        await complete_analysis_run(run_id, result=result_payload)

        # Restart
        await _reinit_db()

        loaded = await get_analysis_run(run_id)
        recomputed = compute_result_fingerprint(result_payload)
        assert loaded["result_fingerprint"] == recomputed


# ===========================================================================
# Audit Persistence Across Restart
# ===========================================================================

class TestAuditRestartRecovery:
    async def test_audit_entries_survive_restart(self):
        """Audit log entries persist across restart."""
        await init_db()
        await log_audit(
            action="researcher_decision",
            agent="researcher",
            decision="approved",
            project_id="restart-proj-audit",
        )

        # Restart
        await _reinit_db()

        entries = await get_audit_log(project_id="restart-proj-audit")
        assert len(entries) > 0
        assert any(e["action"] == "researcher_decision" for e in entries)

    async def test_multiple_audit_entries_survive(self):
        """Multiple audit entries all survive restart."""
        await init_db()
        for i in range(5):
            await log_audit(
                action=f"test_action_{i}",
                agent="test_agent",
                decision=f"decision_{i}",
                project_id="restart-multi-audit",
            )

        # Restart
        await _reinit_db()

        entries = await get_audit_log(project_id="restart-multi-audit")
        assert len(entries) >= 5


# ===========================================================================
# Dataset Identity Persistence Across Restart
# ===========================================================================

class TestDatasetIdentityRestartRecovery:
    async def test_dataset_identity_stable_across_restart(self, tmp_path):
        """Dataset identity hash is identical before and after restart."""
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({
            "value": np.random.RandomState(42).normal(50, 10, 100),
            "group": ["A"] * 50 + ["B"] * 50,
        })
        path = tmp_path / "dataset.csv"
        df.to_csv(path, index=False)

        identity_before = compute_dataset_identity(path)

        # Restart
        await _reinit_db()

        identity_after = compute_dataset_identity(path)
        assert identity_before["content_hash"] == identity_after["content_hash"]
        assert identity_before["schema_hash"] == identity_after["schema_hash"]

    async def test_dataset_identity_distinct_for_different_data(self, tmp_path):
        """Different datasets have distinct identities after restart."""
        import pandas as pd

        df1 = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        df2 = pd.DataFrame({"value": [1, 2, 3, 4, 6]})
        p1 = tmp_path / "d1.csv"
        p2 = tmp_path / "d2.csv"
        df1.to_csv(p1, index=False)
        df2.to_csv(p2, index=False)

        id1 = compute_dataset_identity(p1)
        id2 = compute_dataset_identity(p2)

        # Restart
        await _reinit_db()

        id1_after = compute_dataset_identity(p1)
        id2_after = compute_dataset_identity(p2)

        assert id1["content_hash"] == id1_after["content_hash"]
        assert id2["content_hash"] == id2_after["content_hash"]
        assert id1["content_hash"] != id2["content_hash"]


# ===========================================================================
# Full Workflow Restart Recovery
# ===========================================================================

class TestFullWorkflowRestartRecovery:
    async def test_full_workflow_survives_restart(self, tmp_path):
        """Complete workflow: plan → run → result → audit → restart → all recoverable."""
        import pandas as pd
        import numpy as np
        from app.statistics.engine import independent_t_test

        await init_db()

        # 1. Create dataset
        df = pd.DataFrame({
            "value": np.random.RandomState(42).normal(50, 12, 80),
            "group": ["A"] * 40 + ["B"] * 40,
        })
        data_path = tmp_path / "workflow_data.csv"
        df.to_csv(data_path, index=False)
        ds_identity = compute_dataset_identity(data_path)

        # 2. Create and approve plan
        plan_id = "full-workflow-plan"
        analysis_id = "full-workflow-analysis"
        plan = {
            "plan_id": plan_id,
            "project_id": "full-workflow-proj",
            "research_question": "Does group affect value?",
            "study_design": "observational",
            "analyses": [
                {
                    "analysis_id": analysis_id,
                    "method": "independent_t_test",
                    "outcome": "value",
                    "predictors": ["group"],
                    "lifecycle": "draft",
                    "approval": "proposed",
                }
            ],
        }
        await plan_persistence.save_plan(plan)
        await plan_persistence.update_analysis(plan_id, analysis_id, {
            "approval": "approved",
            "lifecycle": "approved",
        })

        # 3. Execute and record run
        result = independent_t_test(data_path, outcome="value", group="group",
                                    group1="A", group2="B")

        run_id = await create_analysis_run(
            project_id="full-workflow-proj",
            analysis_type="independent_t_test",
            specification={"analysis_method": "independent_t_test", "primary_outcome": "value"},
            dataset_id="full-workflow-ds",
            dataset_content_hash=ds_identity["content_hash"],
            dataset_schema_hash=ds_identity["schema_hash"],
        )
        await complete_analysis_run(run_id, result=result)

        # 4. Log audit
        await log_audit(
            action="analysis_executed",
            agent="statistical_executor",
            decision=f"completed with p={result['p_value']:.4f}",
            project_id="full-workflow-proj",
        )

        # 5. SIMULATE RESTART
        await _reinit_db()

        # 6. VERIFY EVERYTHING IS RECOVERABLE
        # Plan
        loaded_plan = await plan_persistence.load_plan(plan_id)
        assert loaded_plan is not None
        assert len(loaded_plan["analyses"]) == 1

        # Analysis decision
        result_tuple = await plan_persistence.find_analysis(analysis_id)
        assert result_tuple is not None
        _, analysis = result_tuple
        assert analysis["approval"] == "approved"

        # Run and result
        loaded_run = await get_analysis_run(run_id)
        assert loaded_run is not None
        assert loaded_run["result"]["t_statistic"] == result["t_statistic"]
        assert loaded_run["result"]["p_value"] == result["p_value"]

        # Fingerprint integrity
        recomputed_fp = compute_result_fingerprint(result)
        assert loaded_run["result_fingerprint"] == recomputed_fp

        # Audit
        entries = await get_audit_log(project_id="full-workflow-proj")
        assert len(entries) > 0
        assert any(e["action"] == "analysis_executed" for e in entries)

        # Dataset identity
        ds_identity_after = compute_dataset_identity(data_path)
        assert ds_identity_after["content_hash"] == ds_identity["content_hash"]

    async def test_multiple_restarts_are_stable(self):
        """Data remains stable across multiple restarts (not just one)."""
        await init_db()
        plan_id = "multi-restart-plan"
        plan = {
            "plan_id": plan_id,
            "project_id": "restart-proj",
            "analyses": [
                {
                    "analysis_id": "multi-restart-analysis",
                    "method": "independent_t_test",
                    "lifecycle": "draft",
                }
            ],
        }
        await plan_persistence.save_plan(plan)

        for restart_count in range(3):
            await _reinit_db()
            loaded = await plan_persistence.load_plan(plan_id)
            assert loaded is not None, f"Plan lost after restart #{restart_count + 1}"
            assert loaded["plan_id"] == plan_id
