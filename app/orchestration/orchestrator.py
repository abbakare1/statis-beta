"""Agent Orchestrator — routes requests through the agent pipeline."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from app.agents.base import BaseAgent, AgentResult
from app.agents.data_ingestion import DataIngestionAgent
from app.agents.data_dictionary import DataDictionaryAgent
from app.agents.data_quality import DataQualityAgent
from app.agents.research_question import ResearchQuestionAgent
from app.agents.study_design import StudyDesignAgent
from app.agents.power_sample_size import PowerSampleSizeAgent
from app.agents.analysis_planner import AnalysisPlannerAgent
from app.agents.statistical_executor import StatisticalExecutorAgent
from app.memory.audit import log_audit


class PipelineState:
    """Track the state of an analysis pipeline run."""

    def __init__(self, project_id: str, session_id: str | None = None):
        self.project_id = project_id
        self.session_id = session_id or str(uuid.uuid4())
        self.steps: list[dict[str, Any]] = []
        self.context: dict[str, Any] = {"project_id": project_id, "session_id": self.session_id}
        self.status = "initialized"
        self.started_at = datetime.now(timezone.utc)

    def add_step(self, name: str, agent: str, status: str = "pending", result: dict | None = None):
        self.steps.append({
            "step_name": name,
            "agent": agent,
            "status": status,
            "result": result,
            "warnings": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def update_step(self, index: int, status: str, result: dict | None = None, warnings: list | None = None):
        if index < len(self.steps):
            self.steps[index]["status"] = status
            self.steps[index]["timestamp"] = datetime.now(timezone.utc).isoformat()
            if result:
                self.steps[index]["result"] = result
            if warnings:
                self.steps[index]["warnings"] = warnings

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "session_id": self.session_id,
            "steps": self.steps,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
        }


class Orchestrator:
    """Manages the full analysis pipeline, running agents in sequence."""

    def __init__(self):
        self.agents: dict[str, BaseAgent] = {
            "data_ingestion": DataIngestionAgent(),
            "data_dictionary": DataDictionaryAgent(),
            "data_quality": DataQualityAgent(),
            "research_question": ResearchQuestionAgent(),
            "study_design": StudyDesignAgent(),
            "power_sample_size": PowerSampleSizeAgent(),
            "analysis_planner": AnalysisPlannerAgent(),
            "statistical_executor": StatisticalExecutorAgent(),
        }

    async def run_full_pipeline(self, request: dict) -> dict:
        """Run the complete analysis pipeline from research question through analysis plan."""
        project_id = request["project_id"]
        state = PipelineState(project_id)

        logger.info(f"Starting pipeline for project {project_id}")

        # --- Step 1: Research Question ---
        state.add_step("research_question", "research_question", "running")
        rq_agent = self.agents["research_question"]
        rq_result = await rq_agent.run({
            "research_question": request["research_question"],
            "project_id": project_id,
            "session_id": state.session_id,
        })
        state.update_step(-1, "completed" if rq_result.success else "failed",
                          rq_result.data, rq_result.warnings)
        state.context["research_question_spec"] = rq_result.data
        state.context["research_question"] = request["research_question"]

        if not rq_result.success:
            state.status = "failed_at_research_question"
            return state.to_dict()

        # --- Step 2: Study Design ---
        state.add_step("study_design", "study_design", "running")
        sd_agent = self.agents["study_design"]
        sd_result = await sd_agent.run({
            "research_question": request["research_question"],
            "study_design": request.get("study_design", ""),
            "project_id": project_id,
            "session_id": state.session_id,
        })
        state.update_step(-1, "completed" if sd_result.success else "failed",
                          sd_result.data, sd_result.warnings)
        state.context["study_design_spec"] = sd_result.data

        # --- Step 3: Data Ingestion ---
        state.add_step("data_ingestion", "data_ingestion", "running")
        di_agent = self.agents["data_ingestion"]
        di_result = await di_agent.run({
            "file_path": request["file_path"],
            "original_filename": request.get("original_filename"),
            "project_id": project_id,
            "session_id": state.session_id,
            "version": 1,
        })
        state.update_step(-1, "completed" if di_result.success else "failed",
                          di_result.data, di_result.warnings)
        state.context["ingestion"] = di_result.data

        if not di_result.success:
            state.status = "failed_at_data_ingestion"
            return state.to_dict()

        file_path = di_result.data.get("file_path")

        # --- Step 4: Data Dictionary ---
        state.add_step("data_dictionary", "data_dictionary", "running")
        dd_agent = self.agents["data_dictionary"]
        dd_result = await dd_agent.run({
            "file_path": file_path,
            "project_id": project_id,
            "session_id": state.session_id,
            "research_question": request.get("research_question"),
        })
        state.update_step(-1, "completed" if dd_result.success else "failed",
                          dd_result.data, dd_result.warnings)
        state.context["data_dictionary"] = dd_result.data

        # --- Step 5: Data Quality ---
        state.add_step("data_quality", "data_quality", "running")
        dq_agent = self.agents["data_quality"]
        dq_result = await dq_agent.run({
            "file_path": file_path,
            "project_id": project_id,
            "session_id": state.session_id,
        })
        state.update_step(-1, "completed" if dq_result.success else "failed",
                          dq_result.data, dq_result.warnings)
        state.context["data_quality"] = dq_result.data

        # --- Step 6: Power / Sample Size ---
        state.add_step("power_sample_size", "power_sample_size", "running")
        ps_agent = self.agents["power_sample_size"]
        ps_result = await ps_agent.run({
            "file_path": file_path,
            "analysis_type": "t_test",
            "outcome": request.get("outcomes", [None])[0] if request.get("outcomes") else None,
            "group": request.get("variables_of_interest", [None])[0] if request.get("variables_of_interest") else None,
            "project_id": project_id,
            "session_id": state.session_id,
        })
        state.update_step(-1, "completed" if ps_result.success else "failed",
                          ps_result.data, ps_result.warnings)
        state.context["power_assessment"] = ps_result.data

        # --- Step 7: Analysis Planner ---
        state.add_step("analysis_planner", "analysis_planner", "running")
        ap_agent = self.agents["analysis_planner"]
        ap_result = await ap_agent.run({
            "research_question": request["research_question"],
            "study_design": request.get("study_design", ""),
            "data_dictionary": dd_result.data,
            "data_quality": dq_result.data,
            "power_assessment": ps_result.data,
            "project_id": project_id,
            "session_id": state.session_id,
        })
        state.update_step(-1, "completed" if ap_result.success else "failed",
                          ap_result.data, ap_result.warnings)
        state.context["analysis_plan"] = ap_result.data

        state.status = "completed"
        state.completed_at = datetime.now(timezone.utc)

        await log_audit(
            action="pipeline_completed",
            agent="orchestrator",
            decision=f"Pipeline completed with {len(state.steps)} steps",
            project_id=project_id,
            session_id=state.session_id,
        )

        logger.info(f"Pipeline completed for project {project_id}")
        return state.to_dict()

    async def run_single_analysis(self, request: dict) -> dict:
        """Run a single statistical analysis with run tracking."""
        from app.data.run_store import create_analysis_run, complete_analysis_run
        from app.data.dataset_identity import compute_dataset_identity
        from app.statistics.decision_trace import DecisionTrace

        file_path = request["file_path"]
        project_id = request.get("project_id", "unknown")
        analysis_type = request["analysis_type"]

        # Compute dataset identity for run tracking
        try:
            identity = compute_dataset_identity(file_path)
            dataset_content_hash = identity["content_hash"]
            dataset_schema_hash = identity["schema_hash"]
        except Exception:
            dataset_content_hash = "unknown"
            dataset_schema_hash = "unknown"

        # Build specification for run record
        specification = {
            "analysis_method": analysis_type,
            "outcome": request.get("outcome"),
            "predictors": request.get("predictors", []),
            "group": request.get("group"),
            "method": request.get("method", "pearson"),
            "alpha": request.get("alpha", 0.05),
        }

        # Create run record
        run_id = await create_analysis_run(
            project_id=project_id,
            analysis_type=analysis_type,
            specification=specification,
            dataset_id=None,
            dataset_content_hash=dataset_content_hash,
            dataset_schema_hash=dataset_schema_hash,
            file_path=file_path,
            session_id=request.get("session_id"),
        )

        # Build decision trace
        trace = DecisionTrace(run_id, project_id, analysis_type)
        trace.add_data_ingestion(
            file_path=file_path,
            content_hash=dataset_content_hash,
            schema_hash=dataset_schema_hash,
            n_rows=0,  # Will be filled by executor
            n_cols=0,
            columns=[],
        )

        # Execute analysis
        agent = self.agents["statistical_executor"]
        result = await agent.run({
            "analysis_type": analysis_type,
            "file_path": file_path,
            "outcome": request.get("outcome"),
            "group": request.get("group"),
            "group1": request.get("group1"),
            "group2": request.get("group2"),
            "outcome1": request.get("outcome1"),
            "outcome2": request.get("outcome2"),
            "variable1": request.get("variable1"),
            "variable2": request.get("variable2"),
            "x": request.get("x"),
            "y": request.get("y"),
            "predictors": request.get("predictors", []),
            "covariates": request.get("covariates"),
            "method": request.get("method", "pearson"),
            "equal_var": request.get("equal_var", False),
            "alternative": request.get("alternative", "two-sided"),
            "effect_size": request.get("effect_size"),
            "n_per_group": request.get("n_per_group"),
            "alpha": request.get("alpha", 0.05),
            "power": request.get("power", 0.80),
            "columns": request.get("columns"),
            "project_id": project_id,
        })

        # Add execution to trace
        trace.add_execution(
            result_summary={"success": result.success, "n_fields": len(result.data)},
            warnings=result.warnings,
        )

        # Complete run record
        await complete_analysis_run(
            run_id,
            result=result.data,
            success=result.success,
            error_message=result.data.get("error") if not result.success else None,
            warnings=result.warnings,
            decision_trace=trace.to_list(),
        )

        await log_audit(
            action="analysis_executed",
            agent="statistical_executor",
            decision=f"Executed {analysis_type}: {'success' if result.success else 'failed'}",
            project_id=project_id,
        )

        return {
            "success": result.success,
            "data": result.data,
            "warnings": result.warnings,
            "confidence": result.confidence,
            "run_id": run_id,
            "decision_trace": trace.to_list(),
        }
