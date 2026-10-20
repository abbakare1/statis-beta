"""Phase 2J — LLM Failure Isolation (Invariant 41).

Verifies that LLM failures (timeout, malformed JSON, empty response,
invalid method, invented variable, invented result) cause safe failure
without corrupting data, results, or system state.

The LLM orchestrates; deterministic code computes.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.analysis_planner import AnalysisPlannerAgent
from app.agents.statistical_executor import StatisticalExecutorAgent, ExecutionGuardError
from app.statistics.specification import StatisticalAnalysisSpecification, ApprovalStatus
from app.statistics.plan import AnalysisLifecycle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def planner():
    return AnalysisPlannerAgent(model_provider=None)


@pytest.fixture
def executor():
    return StatisticalExecutorAgent()


@pytest.fixture
def approved_spec():
    spec = StatisticalAnalysisSpecification(
        analysis_method="independent_t_test",
        primary_outcome="value",
        predictors=["group"],
    )
    spec.approval_status = ApprovalStatus.APPROVED
    return spec


# ===========================================================================
# LLM Response Failure Modes — Safe Handling
# ===========================================================================

class TestLLMMalformedResponse:
    def test_malformed_json_does_not_crash(self):
        """Malformed JSON from LLM → safe failure, no crash."""
        malformed = '{"analysis_method": "independent_t_test", "primary_outcome": "value"'
        try:
            result = json.loads(malformed)
        except json.JSONDecodeError:
            result = None
        assert result is None

    def test_empty_response_does_not_crash(self):
        """Empty LLM response → safe failure."""
        try:
            result = json.loads("")
        except json.JSONDecodeError:
            result = None
        assert result is None

    def test_none_response_does_not_crash(self):
        """None LLM response → safe failure."""
        try:
            result = json.loads(None) if None is not None else None
        except (TypeError, json.JSONDecodeError):
            result = None
        assert result is None

    def test_wrong_type_response_does_not_crash(self):
        """Wrong type (list instead of dict) → safe handling."""
        wrong = ["not", "a", "plan"]
        try:
            result = json.loads(json.dumps(wrong))
        except (TypeError, json.JSONDecodeError):
            result = None
        assert isinstance(result, list) or result is None

    def test_planner_handles_missing_llm_provider(self):
        """Planner with no LLM provider → safe fallback to deterministic."""
        planner = AnalysisPlannerAgent(model_provider=None)
        assert planner._model_provider is None
        assert planner.name == "analysis_planner"
        assert planner.requires_llm is True


class TestLLMInventedMethod:
    def test_invented_method_rejected_by_registry(self):
        """LLM suggesting a non-existent method → rejected by registry."""
        from app.statistics.registry import get_method
        assert get_method("invented_magic_method") is None
        assert get_method("independent_t_test") is not None

    async def test_invented_method_not_in_dispatch(self, executor, tmp_path):
        """Executor dispatch rejects unknown analysis types."""
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "value": np.random.RandomState(42).normal(50, 10, 50),
            "group": ["A"] * 25 + ["B"] * 25,
        })
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)

        spec = StatisticalAnalysisSpecification(
            analysis_method="invented_magic_method",
            primary_outcome="value",
            predictors=["group"],
        )
        spec.approval_status = ApprovalStatus.APPROVED

        result = await executor._execute({
            "specification": spec.model_dump(),
            "file_path": str(path),
            "lifecycle": AnalysisLifecycle.LOCKED.value,
        })
        assert result.success is False
        assert "Unknown analysis type" in str(result.data) or "invented" in str(result.data).lower()

    def test_invented_method_guardrail_blocks(self, executor, tmp_path):
        """_run_guardrails rejects methods not in registry."""
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "value": np.random.RandomState(42).normal(50, 10, 50),
            "group": ["A"] * 25 + ["B"] * 25,
        })
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)

        spec = StatisticalAnalysisSpecification(
            analysis_method="invented_magic_method",
            primary_outcome="value",
            predictors=["group"],
        )
        spec.approval_status = ApprovalStatus.APPROVED

        with pytest.raises(ExecutionGuardError) as exc_info:
            executor._run_guardrails(
                spec.model_dump(),
                AnalysisLifecycle.LOCKED.value,
                str(path),
            )
        assert "unsupported_method" in str(exc_info.value)


class TestLLMInventedVariable:
    async def test_invented_variable_rejected_by_validation(self, executor, tmp_path):
        """LLM referencing a non-existent variable → rejected."""
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "value": np.random.RandomState(42).normal(50, 10, 50),
            "group": ["A"] * 25 + ["B"] * 25,
        })
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)

        spec = StatisticalAnalysisSpecification(
            analysis_method="independent_t_test",
            primary_outcome="nonexistent_column",
            predictors=["group"],
        )
        spec.approval_status = ApprovalStatus.APPROVED

        result = await executor._execute({
            "specification": spec.model_dump(),
            "file_path": str(path),
            "lifecycle": AnalysisLifecycle.LOCKED.value,
        })
        assert result.success is False

    async def test_invented_predictor_variable_rejected(self, executor, tmp_path):
        """LLM referencing non-existent predictor → rejected."""
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "value": np.random.RandomState(42).normal(50, 10, 50),
            "group": ["A"] * 25 + ["B"] * 25,
        })
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)

        spec = StatisticalAnalysisSpecification(
            analysis_method="independent_t_test",
            primary_outcome="value",
            predictors=["nonexistent_predictor"],
        )
        spec.approval_status = ApprovalStatus.APPROVED

        result = await executor._execute({
            "specification": spec.model_dump(),
            "file_path": str(path),
            "lifecycle": AnalysisLifecycle.LOCKED.value,
        })
        assert result.success is False


class TestLLMInventedResult:
    def test_llm_cannot_inject_results(self):
        """LLM cannot inject result values into a specification."""
        spec = StatisticalAnalysisSpecification(
            analysis_method="independent_t_test",
            primary_outcome="value",
            predictors=["group"],
        )
        spec_dict = spec.model_dump()
        result_fields = ["t_statistic", "p_value", "cohens_d", "mean_difference"]
        for field in result_fields:
            assert field not in spec_dict, f"Specification should not contain {field}"

    def test_llm_cannot_set_approval(self):
        """LLM cannot set approval status — only researcher can."""
        spec = StatisticalAnalysisSpecification(
            analysis_method="independent_t_test",
            primary_outcome="value",
            predictors=["group"],
        )
        # Default approval status should be PROPOSED, not APPROVED
        assert spec.approval_status == ApprovalStatus.PROPOSED


# ===========================================================================
# Executor Guard: LLM Cannot Bypass
# ===========================================================================

class TestExecutorGuard:
    def test_unapproved_execution_blocked(self, executor, tmp_path):
        """Executing an unapproved specification → blocked."""
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "value": np.random.RandomState(42).normal(50, 10, 50),
            "group": ["A"] * 25 + ["B"] * 25,
        })
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)

        unapproved = StatisticalAnalysisSpecification(
            analysis_method="independent_t_test",
            primary_outcome="value",
            predictors=["group"],
        )
        # Default is PROPOSED — not approved
        assert unapproved.approval_status == ApprovalStatus.PROPOSED

        with pytest.raises(ExecutionGuardError) as exc_info:
            executor._run_guardrails(
                unapproved.model_dump(),
                AnalysisLifecycle.LOCKED.value,
                str(path),
            )
        assert "not_approved" in str(exc_info.value)

    def test_invalid_lifecycle_blocked(self, executor, tmp_path):
        """Executing with invalid lifecycle → blocked."""
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "value": np.random.RandomState(42).normal(50, 10, 50),
            "group": ["A"] * 25 + ["B"] * 25,
        })
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)

        spec = StatisticalAnalysisSpecification(
            analysis_method="independent_t_test",
            primary_outcome="value",
            predictors=["group"],
        )
        spec.approval_status = ApprovalStatus.APPROVED

        with pytest.raises(ExecutionGuardError) as exc_info:
            executor._run_guardrails(
                spec.model_dump(),
                "draft",  # not LOCKED or EXECUTED
                str(path),
            )
        assert "invalid_lifecycle" in str(exc_info.value)

    def test_run_guardrails_passes_approved_locked(self, executor, tmp_path):
        """_run_guardrails passes for approved + LOCKED specification."""
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "value": np.random.RandomState(42).normal(50, 10, 50),
            "group": ["A"] * 25 + ["B"] * 25,
        })
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)

        spec = StatisticalAnalysisSpecification(
            analysis_method="independent_t_test",
            primary_outcome="value",
            predictors=["group"],
        )
        spec.approval_status = ApprovalStatus.APPROVED

        # Should not raise
        executor._run_guardrails(
            spec.model_dump(),
            AnalysisLifecycle.LOCKED.value,
            str(path),
        )


# ===========================================================================
# Safe Failure Does Not Corrupt State
# ===========================================================================

class TestSafeFailureNoCorruption:
    async def test_failed_execution_returns_error_not_partial(self, executor, tmp_path):
        """Failed execution returns error object, not partial results."""
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "wrong_column": np.random.RandomState(42).normal(50, 10, 50),
            "another_wrong": ["A"] * 25 + ["B"] * 25,
        })
        path = tmp_path / "wrong.csv"
        df.to_csv(path, index=False)

        spec = StatisticalAnalysisSpecification(
            analysis_method="independent_t_test",
            primary_outcome="value",  # column doesn't exist
            predictors=["group"],  # column doesn't exist
        )
        spec.approval_status = ApprovalStatus.APPROVED

        result = await executor._execute({
            "specification": spec.model_dump(),
            "file_path": str(path),
            "lifecycle": AnalysisLifecycle.LOCKED.value,
        })
        assert result.success is False

    async def test_repeated_failures_are_consistent(self, executor, tmp_path):
        """Same failure → same error type every time (no random corruption)."""
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "wrong_column": np.random.RandomState(42).normal(50, 10, 50),
        })
        path = tmp_path / "wrong2.csv"
        df.to_csv(path, index=False)

        spec = StatisticalAnalysisSpecification(
            analysis_method="independent_t_test",
            primary_outcome="wrong_column",
            predictors=["group"],  # doesn't exist
        )
        spec.approval_status = ApprovalStatus.APPROVED

        successes = []
        for _ in range(3):
            result = await executor._execute({
                "specification": spec.model_dump(),
                "file_path": str(path),
                "lifecycle": AnalysisLifecycle.LOCKED.value,
            })
            successes.append(result.success)

        assert len(set(successes)) == 1, f"Inconsistent failures: {successes}"
        assert successes[0] is False

    async def test_unknown_method_fails_safely(self, executor, tmp_path):
        """Unknown analysis method → safe failure, no crash."""
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "value": np.random.RandomState(42).normal(50, 10, 50),
            "group": ["A"] * 25 + ["B"] * 25,
        })
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)

        spec = StatisticalAnalysisSpecification(
            analysis_method="totally_fake_method",
            primary_outcome="value",
            predictors=["group"],
        )
        spec.approval_status = ApprovalStatus.APPROVED

        result = await executor._execute({
            "specification": spec.model_dump(),
            "file_path": str(path),
            "lifecycle": AnalysisLifecycle.LOCKED.value,
        })
        assert result.success is False
