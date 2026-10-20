"""SQLAlchemy ORM models — single source of truth for the database schema."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


# ---------- Users ----------

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(sa.String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(sa.String(255))
    display_name: Mapped[str | None] = mapped_column(sa.String(128))
    role: Mapped[str] = mapped_column(sa.String(32), default="researcher")  # researcher, admin
    is_active: Mapped[bool] = mapped_column(sa.Boolean, default=True)
    # Beta quota limits
    max_projects: Mapped[int] = mapped_column(sa.Integer, default=5)
    max_analyses: Mapped[int] = mapped_column(sa.Integer, default=50)
    max_upload_mb: Mapped[int] = mapped_column(sa.Integer, default=100)
    # Account expiration for beta access
    access_expires_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


# ---------- Invitations ----------

class Invitation(Base):
    __tablename__ = "invitations"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(sa.String(255), index=True)
    token: Mapped[str] = mapped_column(sa.String(64), unique=True, index=True)
    role: Mapped[str] = mapped_column(sa.String(32), default="researcher")
    expires_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)


# ---------- Audit Ledger ----------

class AuditEntry(Base):
    __tablename__ = "audit_ledger"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=_uuid)
    timestamp: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    session_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)
    project_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)
    action: Mapped[str] = mapped_column(sa.String(128))
    agent: Mapped[str] = mapped_column(sa.String(128))
    input_summary: Mapped[str | None] = mapped_column(sa.Text)
    decision: Mapped[str] = mapped_column(sa.Text)
    rationale: Mapped[str | None] = mapped_column(sa.Text)
    output_summary: Mapped[str | None] = mapped_column(sa.Text)
    confidence: Mapped[str | None] = mapped_column(sa.String(32))
    warnings: Mapped[str | None] = mapped_column(sa.Text)
    researcher_decision: Mapped[str | None] = mapped_column(sa.String(64))


# ---------- Projects ----------

class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=_uuid)
    owner_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)  # User ID
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    name: Mapped[str] = mapped_column(sa.String(256))
    description: Mapped[str | None] = mapped_column(sa.Text)
    research_question: Mapped[str | None] = mapped_column(sa.Text)
    study_design: Mapped[str | None] = mapped_column(sa.String(128))
    status: Mapped[str] = mapped_column(sa.String(64), default="created")


# ---------- Datasets ----------

class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("projects.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    original_filename: Mapped[str] = mapped_column(sa.String(512))
    stored_path: Mapped[str] = mapped_column(sa.String(1024))
    version: Mapped[int] = mapped_column(sa.Integer, default=1)
    content_hash: Mapped[str] = mapped_column(sa.String(128))
    row_count: Mapped[int | None] = mapped_column(sa.Integer)
    column_count: Mapped[int | None] = mapped_column(sa.Integer)
    description: Mapped[str | None] = mapped_column(sa.Text)


# ---------- Analysis Specifications ----------

class AnalysisSpec(Base):
    __tablename__ = "analysis_specs"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("projects.id"), index=True)
    dataset_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("datasets.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    analysis_type: Mapped[str] = mapped_column(sa.String(128))
    category: Mapped[str] = mapped_column(sa.String(64))  # primary, secondary, sensitivity, exploratory
    outcome: Mapped[str | None] = mapped_column(sa.String(256))
    predictors: Mapped[str | None] = mapped_column(sa.Text)  # JSON
    covariates: Mapped[str | None] = mapped_column(sa.Text)  # JSON
    method: Mapped[str] = mapped_column(sa.String(128))
    assumptions: Mapped[str | None] = mapped_column(sa.Text)  # JSON
    missing_data_strategy: Mapped[str | None] = mapped_column(sa.String(128))
    outlier_strategy: Mapped[str | None] = mapped_column(sa.String(128))
    status: Mapped[str] = mapped_column(sa.String(64), default="proposed")


# ---------- Analysis Results ----------

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=_uuid)
    spec_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("analysis_specs.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    result_json: Mapped[str] = mapped_column(sa.Text)  # full structured result
    summary: Mapped[str | None] = mapped_column(sa.Text)
    warnings: Mapped[str | None] = mapped_column(sa.Text)
    execution_time_ms: Mapped[int | None] = mapped_column(sa.Integer)


# ---------- Memory ----------

class MemoryEntry(Base):
    __tablename__ = "memory"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)
    memory_type: Mapped[str] = mapped_column(sa.String(64), index=True)  # project, system, session
    category: Mapped[str] = mapped_column(sa.String(128))
    key: Mapped[str] = mapped_column(sa.String(256))
    value: Mapped[str] = mapped_column(sa.Text)
    source: Mapped[str] = mapped_column(sa.String(64))  # user, ai_interpretation, ai_recommendation, deterministic
    confidence: Mapped[str | None] = mapped_column(sa.String(32))
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


# ---------- Analysis Plans (Phase 2C persistence) ----------

class AnalysisPlanRecord(Base):
    __tablename__ = "analysis_plans"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("projects.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    version: Mapped[str] = mapped_column(sa.String(32), default="1.0")
    research_question: Mapped[str | None] = mapped_column(sa.Text)
    study_design: Mapped[str | None] = mapped_column(sa.String(128))
    hypotheses: Mapped[str | None] = mapped_column(sa.Text)  # JSON list
    warnings: Mapped[str | None] = mapped_column(sa.Text)  # JSON list
    audit_events: Mapped[str | None] = mapped_column(sa.Text)  # JSON list
    plan_json: Mapped[str | None] = mapped_column(sa.Text)  # Full flat-dict plan (API schema)


class AnalysisEntryRecord(Base):
    __tablename__ = "analysis_entries"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=_uuid)
    plan_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("analysis_plans.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    analysis_name: Mapped[str] = mapped_column(sa.String(256))
    analysis_class: Mapped[str] = mapped_column(sa.String(64))  # primary, secondary, sensitivity, exploratory
    current_version_id: Mapped[str | None] = mapped_column(sa.String(36))
    prespecified_status: Mapped[str] = mapped_column(sa.String(64), default="unclear")
    causal_language_flag: Mapped[str] = mapped_column(sa.String(64), default="association")
    multiple_testing_strategy: Mapped[str] = mapped_column(sa.String(64), default="none_required")
    decision_trace: Mapped[str | None] = mapped_column(sa.Text)  # JSON
    power_assessment: Mapped[str | None] = mapped_column(sa.Text)  # JSON


class AnalysisVersionRecord(Base):
    __tablename__ = "analysis_versions"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("analysis_entries.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    superseded_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    version_number: Mapped[str] = mapped_column(sa.String(32))
    specification: Mapped[str] = mapped_column(sa.Text)  # JSON StatisticalAnalysisSpecification
    validation_status: Mapped[str] = mapped_column(sa.String(64), default="pending")
    validation_warnings: Mapped[str | None] = mapped_column(sa.Text)  # JSON list
    validation_blockers: Mapped[str | None] = mapped_column(sa.Text)  # JSON list
    approval_status: Mapped[str] = mapped_column(sa.String(64), default="proposed")
    lifecycle: Mapped[str] = mapped_column(sa.String(64), default="draft")
    modified_by: Mapped[str | None] = mapped_column(sa.String(128))
    modification_reason: Mapped[str | None] = mapped_column(sa.Text)
    # Dataset identity binding
    dataset_id: Mapped[str | None] = mapped_column(sa.String(36))
    dataset_version: Mapped[int | None] = mapped_column(sa.Integer)
    dataset_content_hash: Mapped[str | None] = mapped_column(sa.String(128))
    dataset_schema_hash: Mapped[str | None] = mapped_column(sa.String(128))


class DatasetIdentityRecord(Base):
    __tablename__ = "dataset_identities"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("datasets.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    version: Mapped[int] = mapped_column(sa.Integer, default=1)
    content_hash: Mapped[str] = mapped_column(sa.String(128))
    schema_hash: Mapped[str] = mapped_column(sa.String(128))
    row_count: Mapped[int] = mapped_column(sa.Integer)
    column_count: Mapped[int] = mapped_column(sa.Integer)
    column_names: Mapped[str | None] = mapped_column(sa.Text)  # JSON sorted list
    parent_identity_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)


class TransformationRecord(Base):
    __tablename__ = "transformation_records"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    operation: Mapped[str] = mapped_column(sa.String(128))
    input_dataset_id: Mapped[str] = mapped_column(sa.String(36), index=True)
    input_dataset_version: Mapped[int] = mapped_column(sa.Integer)
    output_dataset_id: Mapped[str] = mapped_column(sa.String(36), index=True)
    output_dataset_version: Mapped[int] = mapped_column(sa.Integer)
    parameters: Mapped[str | None] = mapped_column(sa.Text)  # JSON
    agent: Mapped[str | None] = mapped_column(sa.String(128))
    researcher_approved: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    output_content_hash: Mapped[str | None] = mapped_column(sa.String(128))
    output_schema_hash: Mapped[str | None] = mapped_column(sa.String(128))


# ---------- Data Quality Profiles (Phase 2D.1 persistence) ----------

class QualityProfileRecord(Base):
    __tablename__ = "quality_profiles"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    dataset_id: Mapped[str] = mapped_column(sa.String(36), index=True)
    dataset_content_hash: Mapped[str | None] = mapped_column(sa.String(128))
    dataset_schema_hash: Mapped[str | None] = mapped_column(sa.String(128))
    profile_json: Mapped[str] = mapped_column(sa.Text)  # full DataQualityProfile serialized


# ---------- Transformation Proposals (Phase 2D.1 lifecycle) ----------

class TransformationProposalRecord(Base):
    __tablename__ = "transformation_proposals"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    # Dataset binding
    dataset_id: Mapped[str] = mapped_column(sa.String(36), index=True)
    source_content_hash: Mapped[str] = mapped_column(sa.String(128))
    source_schema_hash: Mapped[str] = mapped_column(sa.String(128))

    # Proposal content
    operation: Mapped[str] = mapped_column(sa.String(128))
    variable: Mapped[str] = mapped_column(sa.String(256))
    parameters: Mapped[str | None] = mapped_column(sa.Text)  # JSON
    strategy_json: Mapped[str | None] = mapped_column(sa.Text)  # JSON HandlingStrategy
    candidate_json: Mapped[str | None] = mapped_column(sa.Text)  # JSON TransformationCandidate
    reason: Mapped[str | None] = mapped_column(sa.Text)

    # Lifecycle state machine
    # Allowed transitions:
    #   DETECTED → PROPOSED → REVIEW_REQUIRED → APPROVED / REJECTED
    #   APPROVED → APPLIED / APPLICATION_FAILED / STALE
    #   STALE → PROPOSED  (re-proposed after dataset change)
    status: Mapped[str] = mapped_column(sa.String(64), default="detected")

    # Review fields
    reviewed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    reviewer: Mapped[str | None] = mapped_column(sa.String(128))
    rejection_reason: Mapped[str | None] = mapped_column(sa.Text)

    # Application fields
    applied_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    derived_dataset_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)
    derived_content_hash: Mapped[str | None] = mapped_column(sa.String(128))
    derived_schema_hash: Mapped[str | None] = mapped_column(sa.String(128))
    impact_summary: Mapped[str | None] = mapped_column(sa.Text)  # JSON
    application_error: Mapped[str | None] = mapped_column(sa.Text)

    # Idempotency: deterministic fingerprint of (dataset_id, operation, variable, parameters)
    application_fingerprint: Mapped[str | None] = mapped_column(sa.String(128), index=True)

    # Anti-p-hacking: store verdict alongside proposal for audit trail
    anti_p_hacking_verdict: Mapped[str | None] = mapped_column(sa.Text)  # JSON

    # Audit
    audit_events: Mapped[str | None] = mapped_column(sa.Text)  # JSON list of state transitions


# ---------- Analysis Runs (Phase 2E reproducibility) ----------

class AnalysisRunRecord(Base):
    __tablename__ = "analysis_runs"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))

    # Project and session
    project_id: Mapped[str] = mapped_column(sa.String(36), sa.ForeignKey("projects.id"), index=True)
    session_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)

    # What was run
    analysis_type: Mapped[str] = mapped_column(sa.String(128))
    specification_json: Mapped[str] = mapped_column(sa.Text)  # full StatisticalAnalysisSpecification

    # Dataset binding
    dataset_id: Mapped[str | None] = mapped_column(sa.String(36))
    dataset_content_hash: Mapped[str] = mapped_column(sa.String(128))
    dataset_schema_hash: Mapped[str] = mapped_column(sa.String(128))
    file_path: Mapped[str | None] = mapped_column(sa.String(1024))

    # Input fingerprint (deterministic hash of all inputs for reproducibility verification)
    input_fingerprint: Mapped[str] = mapped_column(sa.String(128), index=True)

    # Result
    result_json: Mapped[str | None] = mapped_column(sa.Text)  # full structured result
    result_fingerprint: Mapped[str | None] = mapped_column(sa.String(128))  # hash of result content
    success: Mapped[bool] = mapped_column(sa.Boolean, default=True)
    error_message: Mapped[str | None] = mapped_column(sa.Text)

    # Guardrails and validation
    validation_status: Mapped[str | None] = mapped_column(sa.String(64))
    approval_status: Mapped[str | None] = mapped_column(sa.String(64))
    lifecycle: Mapped[str | None] = mapped_column(sa.String(64))
    guardrails_passed: Mapped[bool] = mapped_column(sa.Boolean, default=True)
    warnings_json: Mapped[str | None] = mapped_column(sa.Text)  # JSON list

    # Decision trace: captures every decision point in the pipeline
    decision_trace_json: Mapped[str | None] = mapped_column(sa.Text)  # JSON

    # Reproducibility
    engine_version: Mapped[str | None] = mapped_column(sa.String(32))
    python_version: Mapped[str | None] = mapped_column(sa.String(32))
    dependencies_hash: Mapped[str | None] = mapped_column(sa.String(128))


# ---------------------------------------------------------------------------
# Variable Role Resolution (Phase 2D.2)
# ---------------------------------------------------------------------------

class VariableRoleResolutionRecord(Base):
    """Persistent record of variable role resolution for a dataset."""
    __tablename__ = "variable_role_resolutions"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(sa.String(36), index=True)
    dataset_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)

    # Variable info
    variable_name: Mapped[str] = mapped_column(sa.String(256))
    variable_dtype: Mapped[str | None] = mapped_column(sa.String(64))
    n_unique: Mapped[int | None] = mapped_column(sa.Integer)
    n_total: Mapped[int | None] = mapped_column(sa.Integer)

    # Role resolution
    proposed_role: Mapped[str] = mapped_column(sa.String(64))
    role_source: Mapped[str] = mapped_column(sa.String(64))
    confidence: Mapped[str] = mapped_column(sa.String(32), default="moderate")
    linked_research_question_term: Mapped[str | None] = mapped_column(sa.String(512))
    candidate_roles_json: Mapped[str | None] = mapped_column(sa.Text)  # JSON list

    # Status
    status: Mapped[str] = mapped_column(sa.String(32), default="resolved")
    requires_researcher_review: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    conflicts_json: Mapped[str | None] = mapped_column(sa.Text)  # JSON list
    reasoning_summary: Mapped[str | None] = mapped_column(sa.Text)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


# ---------------------------------------------------------------------------
# Publication Outputs (Phase 2G)
# ---------------------------------------------------------------------------

class FigureRecord(Base):
    """Persistent record of a publication figure."""
    __tablename__ = "figure_records"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    # Project and dataset binding
    project_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)
    dataset_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)
    dataset_content_hash: Mapped[str | None] = mapped_column(sa.String(128))
    dataset_schema_hash: Mapped[str | None] = mapped_column(sa.String(128))

    # Analysis binding
    analysis_version_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)
    analysis_run_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)
    result_fingerprint: Mapped[str | None] = mapped_column(sa.String(128))

    # Specification
    figure_type: Mapped[str] = mapped_column(sa.String(64))
    specification_json: Mapped[str] = mapped_column(sa.Text)
    specification_hash: Mapped[str] = mapped_column(sa.String(128))

    # Artifact
    artifact_path: Mapped[str | None] = mapped_column(sa.String(1024))
    artifact_hash: Mapped[str | None] = mapped_column(sa.String(128))
    artifact_format: Mapped[str | None] = mapped_column(sa.String(32))

    # Lifecycle
    lifecycle: Mapped[str] = mapped_column(sa.String(32), default="draft")
    validation_errors_json: Mapped[str | None] = mapped_column(sa.Text)
    approved_by: Mapped[str | None] = mapped_column(sa.String(128))
    approved_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))

    # Classification
    analytical_classification: Mapped[str] = mapped_column(sa.String(32), default="exploratory")
    prespecified_status: Mapped[str] = mapped_column(sa.String(32), default="unclear")

    # Provenance
    provenance_json: Mapped[str | None] = mapped_column(sa.Text)


class TableRecord(Base):
    """Persistent record of a publication table."""
    __tablename__ = "table_records"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    # Project and dataset binding
    project_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)
    dataset_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)
    dataset_content_hash: Mapped[str | None] = mapped_column(sa.String(128))
    dataset_schema_hash: Mapped[str | None] = mapped_column(sa.String(128))

    # Analysis binding
    analysis_version_ids_json: Mapped[str | None] = mapped_column(sa.Text)
    analysis_run_ids_json: Mapped[str | None] = mapped_column(sa.Text)
    result_fingerprints_json: Mapped[str | None] = mapped_column(sa.Text)

    # Specification
    table_type: Mapped[str] = mapped_column(sa.String(64))
    specification_json: Mapped[str] = mapped_column(sa.Text)
    specification_hash: Mapped[str] = mapped_column(sa.String(128))

    # Artifact
    artifact_path: Mapped[str | None] = mapped_column(sa.String(1024))
    artifact_hash: Mapped[str | None] = mapped_column(sa.String(128))
    artifact_format: Mapped[str | None] = mapped_column(sa.String(32))

    # Lifecycle
    lifecycle: Mapped[str] = mapped_column(sa.String(32), default="draft")
    validation_errors_json: Mapped[str | None] = mapped_column(sa.Text)
    approved_by: Mapped[str | None] = mapped_column(sa.String(128))
    approved_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))

    # Classification
    analytical_classification: Mapped[str] = mapped_column(sa.String(32), default="exploratory")
    prespecified_status: Mapped[str] = mapped_column(sa.String(32), default="unclear")

    # Provenance
    provenance_json: Mapped[str | None] = mapped_column(sa.Text)


class ReportRecord(Base):
    """Persistent record of a publication report."""
    __tablename__ = "report_records"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    # Project and dataset binding
    project_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)
    dataset_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)
    dataset_content_hash: Mapped[str | None] = mapped_column(sa.String(128))
    dataset_schema_hash: Mapped[str | None] = mapped_column(sa.String(128))

    # Analysis binding
    analysis_version_ids_json: Mapped[str | None] = mapped_column(sa.Text)
    analysis_run_ids_json: Mapped[str | None] = mapped_column(sa.Text)

    # Specification
    title: Mapped[str] = mapped_column(sa.String(512))
    specification_json: Mapped[str] = mapped_column(sa.Text)
    specification_hash: Mapped[str] = mapped_column(sa.String(128))

    # Artifact
    artifact_path: Mapped[str | None] = mapped_column(sa.String(1024))
    artifact_hash: Mapped[str | None] = mapped_column(sa.String(128))
    artifact_format: Mapped[str | None] = mapped_column(sa.String(32))

    # Lifecycle
    lifecycle: Mapped[str] = mapped_column(sa.String(32), default="draft")
    validation_errors_json: Mapped[str | None] = mapped_column(sa.Text)
    approved_by: Mapped[str | None] = mapped_column(sa.String(128))
    approved_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))

    # Classification
    analytical_classification: Mapped[str] = mapped_column(sa.String(32), default="exploratory")
    prespecified_status: Mapped[str] = mapped_column(sa.String(32), default="unclear")

    # Provenance
    provenance_json: Mapped[str | None] = mapped_column(sa.Text)


class OutputArtifactRecord(Base):
    """Persistent record of a generated output artifact."""
    __tablename__ = "output_artifacts"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)

    # Output reference
    output_type: Mapped[str] = mapped_column(sa.String(32))
    output_id: Mapped[str] = mapped_column(sa.String(36), index=True)

    # Artifact
    artifact_path: Mapped[str] = mapped_column(sa.String(1024))
    artifact_hash: Mapped[str] = mapped_column(sa.String(128), index=True)
    artifact_format: Mapped[str] = mapped_column(sa.String(32))
    artifact_size_bytes: Mapped[int | None] = mapped_column(sa.Integer)

    # Source binding
    dataset_content_hash: Mapped[str | None] = mapped_column(sa.String(128))
    result_fingerprint: Mapped[str | None] = mapped_column(sa.String(128))
    specification_hash: Mapped[str | None] = mapped_column(sa.String(128))

    # Integrity
    verified: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    verified_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))


class ReproducibilityPackageRecord(Base):
    """Persistent record of a generated reproducibility package."""
    __tablename__ = "reproducibility_packages"

    id: Mapped[str] = mapped_column(sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    # Package
    package_id: Mapped[str] = mapped_column(sa.String(64), unique=True, index=True)
    package_path: Mapped[str] = mapped_column(sa.String(1024))

    # Binding
    project_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)
    dataset_id: Mapped[str | None] = mapped_column(sa.String(36), index=True)
    dataset_content_hash: Mapped[str | None] = mapped_column(sa.String(128))
    dataset_schema_hash: Mapped[str | None] = mapped_column(sa.String(128))

    # Manifest
    manifest_json: Mapped[str] = mapped_column(sa.Text)
    manifest_hash: Mapped[str] = mapped_column(sa.String(128))

    # Verification
    verified: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    verified_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    verification_results_json: Mapped[str | None] = mapped_column(sa.Text)

    # Contents
    analysis_run_ids_json: Mapped[str | None] = mapped_column(sa.Text)
    output_ids_json: Mapped[str | None] = mapped_column(sa.Text)
    includes_raw_data: Mapped[bool] = mapped_column(sa.Boolean, default=False)
