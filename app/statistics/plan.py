"""Analysis Plan — lifecycle-managed, versioned analysis specification.

The AnalysisPlan is the top-level container for all analyses in a project.
Each plan contains one or more StatisticalAnalysisSpecifications, each with
its own lifecycle, version history, and approval chain.

Lifecycle: DRAFT → VALIDATING → VALIDATED → NEEDS_REVIEW → APPROVED → LOCKED → EXECUTED
Failure:   VALIDATION_FAILED / EXECUTED_FAILED
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.statistics.specification import (
    StatisticalAnalysisSpecification,
    AnalysisDecisionTrace,
    PrespecifiedStatus,
    ApprovalStatus,
    ValidationStatus,
)


# ---------------------------------------------------------------------------
# Lifecycle States
# ---------------------------------------------------------------------------

class AnalysisLifecycle(str, Enum):
    DRAFT = "draft"
    VALIDATING = "validating"
    VALIDATED = "validated"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"
    LOCKED = "locked"
    EXECUTED = "executed"
    VALIDATION_FAILED = "validation_failed"
    EXECUTION_FAILED = "execution_failed"


# Allowed transitions
_LIFECYCLE_TRANSITIONS: dict[str, list[str]] = {
    AnalysisLifecycle.DRAFT: [
        AnalysisLifecycle.VALIDATING,
        AnalysisLifecycle.VALIDATION_FAILED,
    ],
    AnalysisLifecycle.VALIDATING: [
        AnalysisLifecycle.VALIDATED,
        AnalysisLifecycle.VALIDATION_FAILED,
        AnalysisLifecycle.NEEDS_REVIEW,
    ],
    AnalysisLifecycle.VALIDATED: [
        AnalysisLifecycle.NEEDS_REVIEW,
        AnalysisLifecycle.APPROVED,
    ],
    AnalysisLifecycle.NEEDS_REVIEW: [
        AnalysisLifecycle.APPROVED,
        AnalysisLifecycle.DRAFT,  # researcher requests modification
        AnalysisLifecycle.VALIDATION_FAILED,
    ],
    AnalysisLifecycle.APPROVED: [
        AnalysisLifecycle.LOCKED,
        AnalysisLifecycle.DRAFT,  # researcher requests modification before locking
    ],
    AnalysisLifecycle.LOCKED: [
        AnalysisLifecycle.EXECUTED,
        AnalysisLifecycle.EXECUTION_FAILED,
    ],
    AnalysisLifecycle.EXECUTED: [],  # terminal
    AnalysisLifecycle.VALIDATION_FAILED: [
        AnalysisLifecycle.DRAFT,  # can be revised
    ],
    AnalysisLifecycle.EXECUTION_FAILED: [
        AnalysisLifecycle.LOCKED,  # can retry from locked
        AnalysisLifecycle.DRAFT,  # or revise
    ],
}


# ---------------------------------------------------------------------------
# Version History
# ---------------------------------------------------------------------------

class AnalysisVersion(BaseModel):
    """A single version of an analysis specification."""
    version_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version_number: str = "1.0"
    specification: StatisticalAnalysisSpecification
    validation_status: ValidationStatus = ValidationStatus.PENDING
    validation_warnings: list[str] = []
    validation_blockers: list[str] = []
    approval_status: ApprovalStatus = ApprovalStatus.PROPOSED
    lifecycle: AnalysisLifecycle = AnalysisLifecycle.DRAFT
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    modified_by: str | None = None
    modification_reason: str | None = None
    dataset_id: str | None = None
    dataset_version: int | None = None
    dataset_hash: str | None = None


class VersionHistory(BaseModel):
    """Tracks all versions of an analysis specification."""
    versions: list[AnalysisVersion] = []
    current_version_id: str | None = None

    def add_version(self, version: AnalysisVersion):
        self.versions.append(version)
        self.current_version_id = version.version_id

    def get_current(self) -> AnalysisVersion | None:
        if self.current_version_id is None:
            return None
        for v in self.versions:
            if v.version_id == self.current_version_id:
                return v
        return None

    def get_version(self, version_id: str) -> AnalysisVersion | None:
        for v in self.versions:
            if v.version_id == version_id:
                return v
        return None

    def get_version_by_number(self, version_number: str) -> AnalysisVersion | None:
        for v in self.versions:
            if v.version_number == version_number:
                return v
        return None


# ---------------------------------------------------------------------------
# Analysis Entry (single analysis within a plan)
# ---------------------------------------------------------------------------

class AnalysisEntry(BaseModel):
    """A single analysis within an AnalysisPlan, with full lifecycle tracking."""
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    analysis_name: str
    analysis_class: str  # primary, secondary, sensitivity, exploratory

    # Versioning
    version_history: VersionHistory = Field(default_factory=VersionHistory)

    # Decision trace
    decision_trace: AnalysisDecisionTrace | None = None

    # Prespecification
    prespecified_status: str = "unclear"  # prespecified, not_prespecified, unclear

    # Causal language guardrail
    causal_language_flag: str = "association"  # association, causation

    # Multiple testing
    multiple_testing_strategy: str = "none_required"

    # Power / information
    power_assessment: dict[str, Any] = Field(default_factory=dict)

    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def model_post_init(self, __context: Any) -> None:
        """Create a default v1.0 DRAFT version if none exists."""
        if not self.version_history.versions:
            default_spec = StatisticalAnalysisSpecification(
                analysis_method="",
                primary_outcome="",
            )
            default_version = AnalysisVersion(
                version_number="1.0",
                specification=default_spec,
                lifecycle=AnalysisLifecycle.DRAFT,
            )
            self.version_history.add_version(default_version)

    def get_current_spec(self) -> StatisticalAnalysisSpecification | None:
        """Get the current version's specification."""
        cv = self.version_history.get_current()
        return cv.specification if cv else None

    def get_lifecycle(self) -> AnalysisLifecycle:
        """Get the current lifecycle state."""
        cv = self.version_history.get_current()
        return cv.lifecycle if cv else AnalysisLifecycle.DRAFT

    def can_transition_to(self, target: AnalysisLifecycle) -> bool:
        """Check if a lifecycle transition is allowed."""
        current = self.get_lifecycle()
        allowed = _LIFECYCLE_TRANSITIONS.get(current, [])
        return target in allowed


# ---------------------------------------------------------------------------
# Analysis Plan (top-level container)
# ---------------------------------------------------------------------------

class AnalysisPlan(BaseModel):
    """Top-level analysis plan for a project.

    Contains all analyses, their specifications, lifecycle states, and audit trails.
    """
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    version: str = "1.0"

    # Research context
    research_question: str | None = None
    study_design: str | None = None
    hypotheses: list[str] = []

    # Analyses
    analyses: list[AnalysisEntry] = []

    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Overall plan warnings
    warnings: list[str] = []

    # Audit trail for the plan itself
    audit_events: list[dict[str, Any]] = []

    def add_analysis(self, entry: AnalysisEntry):
        self.analyses.append(entry)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def get_analysis(self, analysis_id: str) -> AnalysisEntry | None:
        for a in self.analyses:
            if a.analysis_id == analysis_id:
                return a
        return None

    def get_analyses_by_class(self, analysis_class: str) -> list[AnalysisEntry]:
        return [a for a in self.analyses if a.analysis_class == analysis_class]

    def get_primary_analyses(self) -> list[AnalysisEntry]:
        return self.get_analyses_by_class("primary")

    def get_executable_analyses(self) -> list[AnalysisEntry]:
        """Return analyses that are in LOCKED or EXECUTED state."""
        return [
            a for a in self.analyses
            if a.get_lifecycle() in (AnalysisLifecycle.LOCKED, AnalysisLifecycle.EXECUTED)
        ]

    def add_audit_event(self, event_type: str, analysis_id: str | None = None, details: dict | None = None):
        self.audit_events.append({
            "event_type": event_type,
            "analysis_id": analysis_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        })
