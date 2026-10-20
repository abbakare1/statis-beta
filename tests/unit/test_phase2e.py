"""Phase 2E Tests — Reproducible Analysis Runs, Decision Traces, and Results Comparison.

Covers:
- AnalysisRun creation, completion, and retrieval
- Input fingerprinting (deterministic, invariant to order)
- Result fingerprinting (deterministic, cross-platform stable)
- Reproducibility verification (same inputs → same result)
- Decision trace construction (every step captured)
- Results comparison (field-by-field, conclusion heuristics)
- Orchestrator run tracking integration
"""

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path):
    """Create a deterministic sample CSV."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "id": range(100),
        "age": rng.normal(50, 15, 100),
        "bmi": rng.lognormal(3, 0.5, 100),
        "bp": rng.normal(120, 20, 100),
        "group": rng.choice(["A", "B"], 100),
        "outcome": rng.choice([0, 1], 100),
    })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path


def _uid() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


# ===========================================================================
# INPUT FINGERPRINT TESTS
# ===========================================================================

class TestInputFingerprint:
    def test_same_inputs_same_fingerprint(self):
        from app.data.run_store import compute_input_fingerprint

        spec = {"method": "t_test", "outcome": "bmi", "predictors": ["age"]}
        fp1 = compute_input_fingerprint(spec, "hash_a", "schema_a")
        fp2 = compute_input_fingerprint(spec, "hash_a", "schema_a")
        assert fp1 == fp2

    def test_different_inputs_different_fingerprint(self):
        from app.data.run_store import compute_input_fingerprint

        spec = {"method": "t_test", "outcome": "bmi"}
        fp1 = compute_input_fingerprint(spec, "hash_a", "schema_a")
        fp2 = compute_input_fingerprint(spec, "hash_b", "schema_a")
        assert fp1 != fp2

    def test_dict_key_order_invariant(self):
        from app.data.run_store import compute_input_fingerprint

        spec1 = {"method": "t_test", "outcome": "bmi", "alpha": 0.05}
        spec2 = {"alpha": 0.05, "outcome": "bmi", "method": "t_test"}
        fp1 = compute_input_fingerprint(spec1, "h", "s")
        fp2 = compute_input_fingerprint(spec2, "h", "s")
        assert fp1 == fp2

    def test_float_normalization(self):
        from app.data.run_store import compute_input_fingerprint

        # Floats that differ beyond 10 decimal places should normalize
        spec1 = {"alpha": 0.12345678901}
        spec2 = {"alpha": 0.12345678902}
        fp1 = compute_input_fingerprint(spec1, "h", "s")
        fp2 = compute_input_fingerprint(spec2, "h", "s")
        assert fp1 == fp2

    def test_list_order_matters(self):
        from app.data.run_store import compute_input_fingerprint

        spec1 = {"predictors": ["age", "bmi"]}
        spec2 = {"predictors": ["bmi", "age"]}
        fp1 = compute_input_fingerprint(spec1, "h", "s")
        fp2 = compute_input_fingerprint(spec2, "h", "s")
        assert fp1 != fp2  # List order matters for reproducibility

    def test_fingerprint_is_sha256(self):
        from app.data.run_store import compute_input_fingerprint

        fp = compute_input_fingerprint({}, "h", "s")
        assert len(fp) == 64  # SHA-256 hex digest


# ===========================================================================
# RESULT FINGERPRINT TESTS
# ===========================================================================

class TestResultFingerprint:
    def test_same_result_same_fingerprint(self):
        from app.data.run_store import compute_result_fingerprint

        result = {"p_value": 0.03, "statistic": 2.15, "n": 100}
        fp1 = compute_result_fingerprint(result)
        fp2 = compute_result_fingerprint(result)
        assert fp1 == fp2

    def test_different_result_different_fingerprint(self):
        from app.data.run_store import compute_result_fingerprint

        fp1 = compute_result_fingerprint({"p_value": 0.03})
        fp2 = compute_result_fingerprint({"p_value": 0.04})
        assert fp1 != fp2

    def test_float_normalization_in_results(self):
        from app.data.run_store import compute_result_fingerprint

        fp1 = compute_result_fingerprint({"p_value": 0.12345678901})
        fp2 = compute_result_fingerprint({"p_value": 0.12345678902})
        assert fp1 == fp2

    def test_nested_dict_fingerprint(self):
        from app.data.run_store import compute_result_fingerprint

        r1 = {"model": {"coef": 1.5, "p": 0.01}, "r_squared": 0.45}
        r2 = {"r_squared": 0.45, "model": {"p": 0.01, "coef": 1.5}}
        fp1 = compute_result_fingerprint(r1)
        fp2 = compute_result_fingerprint(r2)
        assert fp1 == fp2


# ===========================================================================
# ENVIRONMENT CAPTURE TESTS
# ===========================================================================

class TestEnvironmentCapture:
    def test_capture_environment(self):
        from app.data.run_store import capture_environment

        env = capture_environment()
        assert "python_version" in env
        assert "platform" in env
        assert "engine_version" in env
        assert env["engine_version"] == "2e.1"


# ===========================================================================
# ANALYSIS RUN CRUD TESTS
# ===========================================================================

class TestAnalysisRunCRUD:
    @pytest.mark.asyncio
    async def test_create_and_get_run(self, sample_csv):
        from app.data.run_store import create_analysis_run, get_analysis_run

        run_id = await create_analysis_run(
            project_id="proj_test",
            analysis_type="t_test",
            specification={"method": "t_test", "outcome": "bmi"},
            dataset_id="ds1",
            dataset_content_hash="abc123",
            dataset_schema_hash="def456",
            file_path=str(sample_csv),
        )
        assert run_id is not None

        run = await get_analysis_run(run_id)
        assert run is not None
        assert run["run_id"] == run_id
        assert run["analysis_type"] == "t_test"
        assert run["input_fingerprint"] is not None
        assert run["success"] is True  # Default before completion

    @pytest.mark.asyncio
    async def test_complete_run(self, sample_csv):
        from app.data.run_store import create_analysis_run, complete_analysis_run, get_analysis_run

        run_id = await create_analysis_run(
            project_id="proj_test",
            analysis_type="t_test",
            specification={"method": "t_test"},
            dataset_id="ds1",
            dataset_content_hash="abc",
            dataset_schema_hash="def",
        )

        result = {"p_value": 0.03, "statistic": 2.15}
        completed = await complete_analysis_run(
            run_id,
            result=result,
            success=True,
            validation_status="pass",
            lifecycle="executed",
        )
        assert completed is not None
        assert completed["success"] is True
        assert completed["result_fingerprint"] is not None
        assert completed["completed_at"] is not None

        # Verify retrieval
        run = await get_analysis_run(run_id)
        assert run["result"]["p_value"] == 0.03
        assert run["validation_status"] == "pass"

    @pytest.mark.asyncio
    async def test_complete_run_with_warnings(self, sample_csv):
        from app.data.run_store import create_analysis_run, complete_analysis_run

        run_id = await create_analysis_run(
            project_id="proj_test", analysis_type="t_test",
            specification={}, dataset_id="ds1",
            dataset_content_hash="a", dataset_schema_hash="b",
        )
        completed = await complete_analysis_run(
            run_id, result={"p_value": 0.01},
            warnings=["Small sample size", "High missingness"],
        )
        assert completed["warnings"] == ["Small sample size", "High missingness"]

    @pytest.mark.asyncio
    async def test_complete_run_with_decision_trace(self, sample_csv):
        from app.data.run_store import create_analysis_run, complete_analysis_run

        run_id = await create_analysis_run(
            project_id="proj_test", analysis_type="t_test",
            specification={}, dataset_id="ds1",
            dataset_content_hash="a", dataset_schema_hash="b",
        )
        trace = [
            {"step": "ingestion", "agent": "data_ingestion"},
            {"step": "execution", "agent": "statistical_engine"},
        ]
        completed = await complete_analysis_run(
            run_id, result={"p_value": 0.01}, decision_trace=trace,
        )
        assert completed["decision_trace"] == trace

    @pytest.mark.asyncio
    async def test_list_runs_by_project(self, sample_csv):
        from app.data.run_store import create_analysis_run, list_analysis_runs

        unique = _uid()
        for _ in range(3):
            await create_analysis_run(
                project_id=unique, analysis_type="t_test",
                specification={}, dataset_id="ds1",
                dataset_content_hash="a", dataset_schema_hash="b",
            )
        other = _uid()
        await create_analysis_run(
            project_id=other, analysis_type="anova",
            specification={}, dataset_id="ds1",
            dataset_content_hash="a", dataset_schema_hash="b",
        )

        proj_runs = await list_analysis_runs(project_id=unique)
        assert len(proj_runs) == 3

    @pytest.mark.asyncio
    async def test_list_runs_by_type(self, sample_csv):
        from app.data.run_store import create_analysis_run, list_analysis_runs

        unique = _uid()
        await create_analysis_run(
            project_id=unique, analysis_type="t_test",
            specification={}, dataset_id="ds1",
            dataset_content_hash="a", dataset_schema_hash="b",
        )
        await create_analysis_run(
            project_id=unique, analysis_type="anova",
            specification={}, dataset_id="ds1",
            dataset_content_hash="a", dataset_schema_hash="b",
        )

        t_runs = await list_analysis_runs(project_id=unique, analysis_type="t_test")
        assert len(t_runs) == 1
        assert t_runs[0]["analysis_type"] == "t_test"

    @pytest.mark.asyncio
    async def test_nonexistent_run_returns_none(self):
        from app.data.run_store import get_analysis_run
        result = await get_analysis_run("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_find_runs_by_fingerprint(self, sample_csv):
        from app.data.run_store import (
            create_analysis_run, find_runs_by_fingerprint, compute_input_fingerprint,
        )

        unique = _uid()
        spec = {"method": "t_test", "unique": unique}
        await create_analysis_run(
            project_id=unique, analysis_type="t_test",
            specification=spec, dataset_id="ds1",
            dataset_content_hash="h1", dataset_schema_hash="s1",
        )

        fp = compute_input_fingerprint(spec, "h1", "s1")
        runs = await find_runs_by_fingerprint(fp)
        assert len(runs) >= 1
        assert runs[0]["input_fingerprint"] == fp


# ===========================================================================
# REPRODUCIBILITY TESTS
# ===========================================================================

class TestReproducibility:
    @pytest.mark.asyncio
    async def test_new_run_status(self, sample_csv):
        from app.statistics.reproducibility import verify_reproducibility, ReproducibilityStatus

        status = await verify_reproducibility(
            {"method": "t_test"}, "hash1", "schema1",
        )
        assert status["status"] == ReproducibilityStatus.NEW_RUN
        assert len(status["matching_runs"]) == 0

    @pytest.mark.asyncio
    async def test_verified_with_same_result(self, sample_csv):
        from app.data.run_store import create_analysis_run, complete_analysis_run
        from app.statistics.reproducibility import verify_reproducibility, ReproducibilityStatus

        unique = _uid()
        spec = {"method": "t_test", "unique": unique}
        run_id = await create_analysis_run(
            project_id=unique, analysis_type="t_test",
            specification=spec, dataset_id="ds1",
            dataset_content_hash="h1", dataset_schema_hash="s1",
        )
        result = {"p_value": 0.03, "statistic": 2.15}
        await complete_analysis_run(run_id, result=result)

        # Verify with same result
        status = await verify_reproducibility(
            spec, "h1", "s1", current_result=result,
        )
        assert status["status"] == ReproducibilityStatus.VERIFIED
        assert len(status["matching_runs"]) >= 1

    @pytest.mark.asyncio
    async def test_mismatch_with_different_result(self, sample_csv):
        from app.data.run_store import create_analysis_run, complete_analysis_run
        from app.statistics.reproducibility import verify_reproducibility, ReproducibilityStatus

        spec = {"method": "t_test"}
        run_id = await create_analysis_run(
            project_id="p1", analysis_type="t_test",
            specification=spec, dataset_id="ds1",
            dataset_content_hash="h1", dataset_schema_hash="s1",
        )
        await complete_analysis_run(run_id, result={"p_value": 0.03})

        # Verify with different result
        status = await verify_reproducibility(
            spec, "h1", "s1", current_result={"p_value": 0.99},
        )
        assert status["status"] == ReproducibilityStatus.MISMATCH
        assert len(status["differences"]) > 0

    @pytest.mark.asyncio
    async def test_verified_without_current_result(self, sample_csv):
        from app.data.run_store import create_analysis_run, complete_analysis_run
        from app.statistics.reproducibility import verify_reproducibility, ReproducibilityStatus

        spec = {"method": "t_test"}
        run_id = await create_analysis_run(
            project_id="p1", analysis_type="t_test",
            specification=spec, dataset_id="ds1",
            dataset_content_hash="h1", dataset_schema_hash="s1",
        )
        await complete_analysis_run(run_id, result={"p_value": 0.03})

        # Verify without providing current result (just check prior exists)
        status = await verify_reproducibility(spec, "h1", "s1")
        assert status["status"] == ReproducibilityStatus.VERIFIED
        assert status["prior_result"] is not None


# ===========================================================================
# DECISION TRACE TESTS
# ===========================================================================

class TestDecisionTrace:
    def test_create_trace(self):
        from app.statistics.decision_trace import DecisionTrace

        trace = DecisionTrace("run_1", "proj_1", "t_test")
        assert trace.run_id == "run_1"
        assert trace.project_id == "proj_1"
        assert len(trace.steps) == 0

    def test_add_step(self):
        from app.statistics.decision_trace import DecisionTrace, TraceStep

        trace = DecisionTrace("run_1", "proj_1", "t_test")
        step = TraceStep("ingestion", "data_ingestion", deterministic=True)
        trace.add_step(step)
        assert len(trace.steps) == 1
        assert trace.steps[0].step_name == "ingestion"

    def test_add_data_ingestion(self):
        from app.statistics.decision_trace import DecisionTrace

        trace = DecisionTrace("run_1", "proj_1", "t_test")
        trace.add_data_ingestion(
            file_path="/data/test.csv",
            content_hash="abc123",
            schema_hash="def456",
            n_rows=100,
            n_cols=5,
            columns=["age", "bmi", "bp"],
        )
        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.step_name == "data_ingestion"
        assert step.deterministic is True
        assert step.outputs["n_rows"] == 100

    def test_add_method_selection(self):
        from app.statistics.decision_trace import DecisionTrace

        trace = DecisionTrace("run_1", "proj_1", "t_test")
        trace.add_method_selection(
            selected_method="independent_t_test",
            rationale="Two independent groups, normal distribution",
        )
        step = trace.steps[0]
        assert step.deterministic is False
        assert step.reasoning == "Two independent groups, normal distribution"

    def test_add_guardrail_check(self):
        from app.statistics.decision_trace import DecisionTrace

        trace = DecisionTrace("run_1", "proj_1", "t_test")
        trace.add_guardrail_check("dataset_exists", passed=True)
        trace.add_guardrail_check("lifecycle_locked", passed=False, reason="Still in draft")

        assert len(trace.steps) == 2
        assert trace.steps[1].guardrails_triggered == ["lifecycle_locked"]

    def test_to_list(self):
        from app.statistics.decision_trace import DecisionTrace

        trace = DecisionTrace("run_1", "proj_1", "t_test")
        trace.add_data_ingestion("/data.csv", "a", "b", 100, 5, ["age"])
        trace.add_execution({"success": True})

        steps = trace.to_list()
        assert len(steps) == 2
        assert steps[0]["step_name"] == "data_ingestion"
        assert steps[1]["step_name"] == "execution"

    def test_to_dict(self):
        from app.statistics.decision_trace import DecisionTrace

        trace = DecisionTrace("run_1", "proj_1", "t_test")
        trace.add_execution({"success": True})

        d = trace.to_dict()
        assert d["run_id"] == "run_1"
        assert d["n_steps"] == 1

    def test_summary(self):
        from app.statistics.decision_trace import DecisionTrace

        trace = DecisionTrace("run_1", "proj_1", "t_test")
        trace.add_data_ingestion("/data.csv", "a", "b", 100, 5, ["age"])
        trace.add_guardrail_check("check1", passed=True)
        trace.add_execution({"success": True})

        summary = trace.summary()
        assert "run_1" in summary
        assert "Steps: 3" in summary
        assert "[det]" in summary


# ===========================================================================
# RESULTS COMPARATOR TESTS
# ===========================================================================

class TestResultsComparator:
    @pytest.mark.asyncio
    async def test_identical_runs(self, sample_csv):
        from app.data.run_store import create_analysis_run, complete_analysis_run
        from app.statistics.results_comparator import compare_runs

        spec = {"method": "t_test"}
        common = dict(dataset_id="ds1", dataset_content_hash="h", dataset_schema_hash="s")

        run_a = await create_analysis_run(project_id="p", analysis_type="t", specification=spec, **common)
        await complete_analysis_run(run_a, result={"p_value": 0.03, "statistic": 2.1})

        run_b = await create_analysis_run(project_id="p", analysis_type="t", specification=spec, **common)
        await complete_analysis_run(run_b, result={"p_value": 0.03, "statistic": 2.1})

        comparison = await compare_runs(run_a, run_b)
        assert comparison["n_differences"] == 0
        assert comparison["conclusions_same"] is True

    @pytest.mark.asyncio
    async def test_different_results(self, sample_csv):
        from app.data.run_store import create_analysis_run, complete_analysis_run
        from app.statistics.results_comparator import compare_runs

        common = dict(dataset_id="ds1", dataset_content_hash="h", dataset_schema_hash="s",
                      specification={"method": "t_test"})

        run_a = await create_analysis_run(project_id="p", analysis_type="t", **common)
        await complete_analysis_run(run_a, result={"p_value": 0.03})

        run_b = await create_analysis_run(project_id="p", analysis_type="t", **common)
        await complete_analysis_run(run_b, result={"p_value": 0.85})

        comparison = await compare_runs(run_a, run_b)
        assert comparison["n_differences"] > 0
        assert comparison["conclusions_same"] is False  # One significant, one not

    @pytest.mark.asyncio
    async def test_same_p_value_same_conclusion(self, sample_csv):
        from app.data.run_store import create_analysis_run, complete_analysis_run
        from app.statistics.results_comparator import compare_runs

        common = dict(dataset_id="ds1", dataset_content_hash="h", dataset_schema_hash="s",
                      specification={"method": "t_test"})

        run_a = await create_analysis_run(project_id="p", analysis_type="t", **common)
        await complete_analysis_run(run_a, result={"p_value": 0.03})

        run_b = await create_analysis_run(project_id="p", analysis_type="t", **common)
        await complete_analysis_run(run_b, result={"p_value": 0.04})

        comparison = await compare_runs(run_a, run_b)
        assert comparison["conclusions_same"] is True  # Both significant

    @pytest.mark.asyncio
    async def test_nonexistent_run_raises(self):
        from app.statistics.results_comparator import compare_runs

        with pytest.raises(ValueError, match="not found"):
            await compare_runs("nonexistent_a", "nonexistent_b")

    @pytest.mark.asyncio
    async def test_compare_by_fingerprint(self, sample_csv):
        from app.data.run_store import create_analysis_run, complete_analysis_run
        from app.statistics.results_comparator import compare_by_fingerprint

        unique = _uid()
        spec = {"method": "t_test", "run_id": unique}
        run1 = await create_analysis_run(
            project_id=unique, analysis_type="t", specification=spec,
            dataset_id="ds1", dataset_content_hash="h", dataset_schema_hash="s",
        )
        await complete_analysis_run(run1, result={"p_value": 0.03})

        run2 = await create_analysis_run(
            project_id=unique, analysis_type="t", specification=spec,
            dataset_id="ds1", dataset_content_hash="h", dataset_schema_hash="s",
        )
        await complete_analysis_run(run2, result={"p_value": 0.03})

        result = await compare_by_fingerprint(spec, "h", "s")
        assert result["n_runs"] == 2
        assert result["all_identical"] is True


# ===========================================================================
# ORCHESTRATOR INTEGRATION TESTS
# ===========================================================================

class TestOrchestratorIntegration:
    @pytest.mark.asyncio
    async def test_run_single_analysis_creates_run(self, sample_csv):
        from app.orchestration.orchestrator import Orchestrator

        orch = Orchestrator()
        result = await orch.run_single_analysis({
            "analysis_type": "descriptive",
            "file_path": str(sample_csv),
            "project_id": "proj_test",
        })

        assert "run_id" in result
        assert "decision_trace" in result
        assert len(result["decision_trace"]) >= 1

    @pytest.mark.asyncio
    async def test_run_single_analysis_persists_run(self, sample_csv):
        from app.orchestration.orchestrator import Orchestrator
        from app.data.run_store import get_analysis_run

        orch = Orchestrator()
        result = await orch.run_single_analysis({
            "analysis_type": "descriptive",
            "file_path": str(sample_csv),
            "project_id": "proj_test",
        })

        run = await get_analysis_run(result["run_id"])
        assert run is not None
        assert run["analysis_type"] == "descriptive"
        assert run["success"] is True
        assert run["result"] is not None

    @pytest.mark.asyncio
    async def test_decision_trace_includes_execution(self, sample_csv):
        from app.orchestration.orchestrator import Orchestrator

        orch = Orchestrator()
        result = await orch.run_single_analysis({
            "analysis_type": "descriptive",
            "file_path": str(sample_csv),
            "project_id": "proj_test",
        })

        trace = result["decision_trace"]
        step_names = [s["step_name"] for s in trace]
        assert "data_ingestion" in step_names
        assert "execution" in step_names

    @pytest.mark.asyncio
    async def test_reproducibility_of_orchestrator_runs(self, sample_csv):
        from app.orchestration.orchestrator import Orchestrator
        from app.data.run_store import get_analysis_run, compute_input_fingerprint

        orch = Orchestrator()
        r1 = await orch.run_single_analysis({
            "analysis_type": "descriptive",
            "file_path": str(sample_csv),
            "project_id": "proj_test",
        })
        r2 = await orch.run_single_analysis({
            "analysis_type": "descriptive",
            "file_path": str(sample_csv),
            "project_id": "proj_test",
        })

        run1 = await get_analysis_run(r1["run_id"])
        run2 = await get_analysis_run(r2["run_id"])

        # Same inputs should produce same input fingerprint
        assert run1["input_fingerprint"] == run2["input_fingerprint"]

        # Both should succeed
        assert run1["success"] is True
        assert run2["success"] is True
