"""Decision Trace Builder — construct complete audit trails from raw data to final results.

Every analytical decision is captured as a timestamped step with inputs, outputs,
reasoning, and confidence. The trace is immutable once constructed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class TraceStep:
    """A single step in the decision trace."""

    def __init__(
        self,
        step_name: str,
        agent: str,
        *,
        inputs: dict | None = None,
        outputs: dict | None = None,
        reasoning: str | None = None,
        confidence: str | None = None,
        warnings: list[str] | None = None,
        guardrails_triggered: list[str] | None = None,
        deterministic: bool = True,
    ):
        self.step_name = step_name
        self.agent = agent
        self.inputs = inputs or {}
        self.outputs = outputs or {}
        self.reasoning = reasoning
        self.confidence = confidence
        self.warnings = warnings or []
        self.guardrails_triggered = guardrails_triggered or []
        self.deterministic = deterministic
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "step_name": self.step_name,
            "agent": self.agent,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "guardrails_triggered": self.guardrails_triggered,
            "deterministic": self.deterministic,
            "timestamp": self.timestamp,
        }


class DecisionTrace:
    """Immutable record of every decision made during an analysis pipeline run."""

    def __init__(self, run_id: str, project_id: str, analysis_type: str):
        self.run_id = run_id
        self.project_id = project_id
        self.analysis_type = analysis_type
        self.steps: list[TraceStep] = []
        self.created_at = datetime.now(timezone.utc).isoformat()

    def add_step(self, step: TraceStep) -> None:
        """Append a step to the trace."""
        self.steps.append(step)

    def add_data_ingestion(
        self,
        file_path: str,
        content_hash: str,
        schema_hash: str,
        n_rows: int,
        n_cols: int,
        columns: list[str],
    ) -> None:
        """Record data ingestion step."""
        self.add_step(TraceStep(
            step_name="data_ingestion",
            agent="data_ingestion",
            inputs={"file_path": file_path},
            outputs={
                "content_hash": content_hash[:16],
                "schema_hash": schema_hash[:16],
                "n_rows": n_rows,
                "n_cols": n_cols,
                "columns": columns,
            },
            deterministic=True,
        ))

    def add_data_quality_assessment(
        self,
        n_issues: int,
        n_critical: int,
        missingness_summary: dict | None = None,
        outlier_summary: dict | None = None,
    ) -> None:
        """Record data quality assessment step."""
        self.add_step(TraceStep(
            step_name="data_quality_assessment",
            agent="data_quality",
            inputs={},
            outputs={
                "n_issues": n_issues,
                "n_critical": n_critical,
                "missingness_summary": missingness_summary,
                "outlier_summary": outlier_summary,
            },
            deterministic=True,
        ))

    def add_candidate_generation(
        self,
        n_candidates: int,
        primary_method: str,
        alternatives: list[str],
    ) -> None:
        """Record candidate generation step."""
        self.add_step(TraceStep(
            step_name="candidate_generation",
            agent="candidate_generator",
            inputs={},
            outputs={
                "n_candidates": n_candidates,
                "primary_method": primary_method,
                "alternatives": alternatives,
            },
            deterministic=True,
        ))

    def add_method_selection(
        self,
        selected_method: str,
        rationale: str,
        alternatives_rejected: list[dict] | None = None,
    ) -> None:
        """Record AI method selection step."""
        self.add_step(TraceStep(
            step_name="method_selection",
            agent="analysis_planner",
            inputs={},
            outputs={
                "selected_method": selected_method,
                "alternatives_rejected": alternatives_rejected or [],
            },
            reasoning=rationale,
            deterministic=False,
        ))

    def add_validation(
        self,
        status: str,
        checks_passed: list[str],
        warnings: list[str] | None = None,
        blockers: list[str] | None = None,
    ) -> None:
        """Record validation step."""
        self.add_step(TraceStep(
            step_name="validation",
            agent="statistical_validator",
            inputs={},
            outputs={
                "status": status,
                "checks_passed": checks_passed,
                "blockers": blockers or [],
            },
            warnings=warnings or [],
            deterministic=True,
        ))

    def add_guardrail_check(
        self,
        guard_name: str,
        passed: bool,
        reason: str | None = None,
    ) -> None:
        """Record a guardrail check."""
        self.add_step(TraceStep(
            step_name=f"guardrail_{guard_name}",
            agent="statistical_executor",
            inputs={"guard": guard_name},
            outputs={"passed": passed, "reason": reason},
            guardrails_triggered=[guard_name] if not passed else [],
            deterministic=True,
        ))

    def add_execution(
        self,
        result_summary: dict,
        warnings: list[str] | None = None,
    ) -> None:
        """Record analysis execution step."""
        self.add_step(TraceStep(
            step_name="execution",
            agent="statistical_engine",
            inputs={},
            outputs=result_summary,
            warnings=warnings or [],
            deterministic=True,
        ))

    def add_researcher_decision(
        self,
        decision: str,
        rationale: str | None = None,
    ) -> None:
        """Record a researcher decision (approval/rejection)."""
        self.add_step(TraceStep(
            step_name="researcher_decision",
            agent="researcher",
            inputs={},
            outputs={"decision": decision},
            reasoning=rationale,
            deterministic=False,
        ))

    def to_list(self) -> list[dict]:
        """Serialize the trace as a list of step dicts."""
        return [step.to_dict() for step in self.steps]

    def to_dict(self) -> dict:
        """Serialize the full trace."""
        return {
            "run_id": self.run_id,
            "project_id": self.project_id,
            "analysis_type": self.analysis_type,
            "created_at": self.created_at,
            "n_steps": len(self.steps),
            "steps": self.to_list(),
        }

    def summary(self) -> str:
        """Human-readable summary of the decision trace."""
        lines = [f"Decision Trace for run {self.run_id}"]
        lines.append(f"  Analysis: {self.analysis_type}")
        lines.append(f"  Steps: {len(self.steps)}")
        for i, step in enumerate(self.steps, 1):
            det = "det" if step.deterministic else "ai"
            status = "OK"
            if step.guardrails_triggered:
                status = f"GUARDRAIL: {', '.join(step.guardrails_triggered)}"
            elif step.warnings:
                status = f"WARN: {len(step.warnings)} warnings"
            lines.append(f"  {i}. [{det}] {step.step_name} ({step.agent}) — {status}")
        return "\n".join(lines)
