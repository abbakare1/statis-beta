"""Statistical Analysis Specification — the formal contract between AI reasoning and execution.

Every analysis must be fully specified in this structure BEFORE it runs.
This is the single source of truth for what will be computed.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PrespecifiedStatus(str, Enum):
    PRESPECIFIED = "prespecified"
    SECONDARY = "secondary"
    SENSITIVITY = "sensitivity"
    EXPLORATORY = "exploratory"
    POST_HOC = "post_hoc"


class ApprovalStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    LOCKED = "locked"


class ValidationStatus(str, Enum):
    PENDING = "pending"
    PASS = "pass"
    WARNING = "warning"
    BLOCK = "block"


class AnalysisAssumption(BaseModel):
    name: str
    description: str
    test_method: str | None = None
    status: str = "untested"  # untested, passed, failed, warning
    details: str | None = None


class AnalysisDiagnosis(BaseModel):
    check_name: str
    status: str  # pass, warning, fail
    value: Any = None
    threshold: Any = None
    description: str | None = None


class StatisticalAnalysisSpecification(BaseModel):
    """The complete, machine-readable specification for a single statistical analysis.

    This is the contract between AI reasoning and deterministic execution.
    """

    # Identifiers
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str | None = None
    dataset_id: str | None = None
    version: str = "1.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Research context
    research_question: str | None = None
    study_design: str | None = None

    # Variables
    primary_outcome: str
    secondary_outcomes: list[str] = []
    predictors: list[str] = []
    covariates: list[str] = []
    subgroup_variables: list[str] = []
    interaction_terms: list[str] = []

    # Method specification
    analysis_method: str
    method_category: str = "primary"  # primary, secondary, sensitivity, exploratory
    rationale: str | None = None

    # Assumptions
    assumptions: list[AnalysisAssumption] = []
    required_assumptions: list[str] = []

    # Data handling
    transformations: list[dict[str, Any]] = []
    missing_data_strategy: str = "complete_case"
    outlier_strategy: str = "flag_only"

    # Multiple testing
    multiple_testing_correction: str | None = None
    alpha: float = 0.05
    confidence_level: float = 0.95

    # Sensitivity analyses
    sensitivity_analyses: list[str] = []

    # Computation
    software_engine: str = "python"
    random_seed: int = 42

    # Status
    prespecified_status: PrespecifiedStatus = PrespecifiedStatus.PRESPECIFIED
    approval_status: ApprovalStatus = ApprovalStatus.PROPOSED

    # Validation
    validation_status: ValidationStatus = ValidationStatus.PENDING
    validation_warnings: list[str] = []
    validation_blockers: list[str] = []

    # Diagnostics (populated after execution)
    diagnostics: list[AnalysisDiagnosis] = []

    def approve(self):
        self.approval_status = ApprovalStatus.APPROVED

    def reject(self):
        self.approval_status = ApprovalStatus.REJECTED

    def lock(self):
        self.approval_status = ApprovalStatus.LOCKED

    def modify(self, new_version: str):
        self.approval_status = ApprovalStatus.MODIFIED
        self.version = new_version


class AnalysisDecisionTrace(BaseModel):
    """Records the reasoning chain for how an analysis method was selected.

    This allows the researcher to inspect why a particular method was chosen.
    """
    analysis_id: str
    steps: list[dict[str, Any]] = []
    final_method: str | None = None
    alternatives_considered: list[str] = []
    rejection_reasons: list[dict[str, str]] = []

    def add_step(self, step: str, detail: str, status: str = "info"):
        self.steps.append({
            "step": step,
            "detail": detail,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def reject_method(self, method: str, reason: str):
        self.rejection_reasons.append({"method": method, "reason": reason})

    def set_final(self, method: str):
        self.final_method = method
