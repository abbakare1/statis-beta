"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------- Enums ----------

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class AnalysisCategory(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SENSITIVITY = "sensitivity"
    EXPLORATORY = "exploratory"


class ResearcherDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    MODIFY = "modify"
    INVESTIGATE = "investigate"


class MemoryType(str, Enum):
    PROJECT = "project"
    SYSTEM = "system"
    SESSION = "session"


# ---------- Project ----------

class ProjectCreate(BaseModel):
    name: str
    description: str | None = None
    research_question: str | None = None
    study_design: str | None = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str | None
    research_question: str | None
    study_design: str | None
    status: str
    created_at: datetime
    updated_at: datetime


# ---------- Dataset ----------

class DatasetResponse(BaseModel):
    id: str
    project_id: str
    original_filename: str
    version: int
    content_hash: str
    row_count: int | None
    column_count: int | None
    description: str | None
    created_at: datetime


# ---------- Variable Info ----------

class VariableInfo(BaseModel):
    name: str
    label: str | None = None
    dtype: str
    role: str | None = None  # ai-generated classification
    role_source: str = "ai_interpretation"
    units: str | None = None
    n_unique: int | None = None
    n_missing: int | None = None
    missing_pct: float | None = None
    is_identifier: bool = False
    sample_values: list[Any] = []
    distribution_summary: dict[str, Any] = {}


# ---------- Data Quality ----------

class DataQualityIssue(BaseModel):
    issue_type: str  # missing, duplicate, outlier, invalid, encoding, etc.
    severity: str  # info, warning, critical
    column: str | None = None
    description: str
    n_affected: int | None = None
    recommendation: str | None = None


class DataQualityReport(BaseModel):
    dataset_id: str
    n_rows: int
    n_columns: int
    total_missing_cells: int
    missing_pct: float
    duplicate_rows: int
    duplicate_pct: float
    variables: list[VariableInfo]
    issues: list[DataQualityIssue]
    summary: str | None = None


# ---------- Research Question ----------

class ResearchQuestionSpec(BaseModel):
    population: str | None = None
    exposure: str | None = None
    intervention: str | None = None
    comparator: str | None = None
    outcome: str | None = None
    time_horizon: str | None = None
    hypothesis: str | None = None
    primary_objective: str | None = None
    secondary_objectives: list[str] = []
    ambiguities: list[str] = []
    clarifications_needed: list[str] = []


# ---------- Study Design ----------

class StudyDesignSpec(BaseModel):
    design_type: str | None = None
    design_description: str | None = None
    appropriate_for_question: bool | None = None
    concerns: list[str] = []
    recommendations: list[str] = []
    confidence: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT


# ---------- Analysis Plan ----------

class AnalysisMethod(BaseModel):
    name: str
    category: AnalysisCategory
    outcome: str
    predictors: list[str] = []
    covariates: list[str] = []
    method: str
    assumptions: list[str] = []
    software_engine: str = "python"
    rationale: str | None = None


class AnalysisPlan(BaseModel):
    project_id: str
    primary_analyses: list[AnalysisMethod] = []
    secondary_analyses: list[AnalysisMethod] = []
    sensitivity_analyses: list[AnalysisMethod] = []
    exploratory_analyses: list[AnalysisMethod] = []
    warnings: list[str] = []


# ---------- Audit ----------

class AuditEntryResponse(BaseModel):
    id: str
    timestamp: datetime
    action: str
    agent: str
    decision: str
    rationale: str | None
    confidence: str | None
    warnings: str | None
    researcher_decision: str | None


# ---------- Pipeline ----------

class PipelineRequest(BaseModel):
    project_id: str
    dataset_id: str
    research_question: str
    study_design: str | None = None
    hypotheses: list[str] = []
    variables_of_interest: list[str] = []
    covariates: list[str] = []
    outcomes: list[str] = []
    publication_target: str | None = None


class PipelineStep(BaseModel):
    step_name: str
    agent: str
    status: str  # pending, running, completed, failed, skipped
    result: dict[str, Any] | None = None
    warnings: list[str] = []
    timestamp: datetime | None = None


class PipelineState(BaseModel):
    project_id: str
    steps: list[PipelineStep] = []
    current_step: int = 0
    status: str = "initialized"
    started_at: datetime | None = None
    completed_at: datetime | None = None
