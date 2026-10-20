"""Analysis Plan Persistence — CRUD operations for analysis plans, entries, and versions.

Uses async SQLAlchemy sessions following the existing STATIS patterns.
All data is serialized to JSON for flexible schema evolution.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import sqlalchemy as sa

from app.core.database import get_session
from app.models import (
    AnalysisPlanRecord,
    AnalysisEntryRecord,
    AnalysisVersionRecord,
)
from app.statistics.plan import (
    AnalysisPlan,
    AnalysisEntry,
    AnalysisVersion,
    VersionHistory,
    AnalysisLifecycle,
)
from app.statistics.specification import (
    StatisticalAnalysisSpecification,
    AnalysisDecisionTrace,
    ValidationStatus,
    ApprovalStatus,
)


# ---------------------------------------------------------------------------
# Plan CRUD
# ---------------------------------------------------------------------------

async def save_plan(plan: AnalysisPlan) -> str:
    """Persist an AnalysisPlan and all its entries/versions. Returns plan_id."""
    async with get_session() as session:
        # Upsert plan record
        existing = await session.get(AnalysisPlanRecord, plan.plan_id)
        if existing:
            existing.version = plan.version
            existing.research_question = plan.research_question
            existing.study_design = plan.study_design
            existing.hypotheses = json.dumps(plan.hypotheses)
            existing.warnings = json.dumps(plan.warnings)
            existing.audit_events = json.dumps(plan.audit_events, default=str)
            existing.updated_at = datetime.now(timezone.utc)
        else:
            record = AnalysisPlanRecord(
                id=plan.plan_id,
                project_id=plan.project_id,
                version=plan.version,
                research_question=plan.research_question,
                study_design=plan.study_design,
                hypotheses=json.dumps(plan.hypotheses),
                warnings=json.dumps(plan.warnings),
                audit_events=json.dumps(plan.audit_events, default=str),
            )
            session.add(record)

        # Upsert each entry
        for entry in plan.analyses:
            await _save_entry(session, plan.plan_id, entry)

        await session.commit()
    return plan.plan_id


async def load_plan(plan_id: str) -> AnalysisPlan | None:
    """Load an AnalysisPlan from the database."""
    async with get_session() as session:
        record = await session.get(AnalysisPlanRecord, plan_id)
        if not record:
            return None

        plan = AnalysisPlan(
            plan_id=record.id,
            project_id=record.project_id,
            version=record.version,
            research_question=record.research_question,
            study_design=record.study_design,
            hypotheses=json.loads(record.hypotheses) if record.hypotheses else [],
            warnings=json.loads(record.warnings) if record.warnings else [],
            audit_events=json.loads(record.audit_events) if record.audit_events else [],
            created_at=record.created_at.isoformat() if record.created_at else "",
            updated_at=record.updated_at.isoformat() if record.updated_at else "",
        )

        # Load entries
        stmt = sa.select(AnalysisEntryRecord).where(AnalysisEntryRecord.plan_id == plan_id)
        result = await session.execute(stmt)
        entry_records = result.scalars().all()

        for entry_rec in entry_records:
            entry = await _load_entry(session, entry_rec)
            plan.analyses.append(entry)

    return plan


async def list_plans(project_id: str | None = None) -> list[dict]:
    """List analysis plans, optionally filtered by project."""
    async with get_session() as session:
        stmt = sa.select(AnalysisPlanRecord).order_by(AnalysisPlanRecord.created_at.desc())
        if project_id:
            stmt = stmt.where(AnalysisPlanRecord.project_id == project_id)
        result = await session.execute(stmt)
        records = result.scalars().all()
        return [
            {
                "plan_id": r.id,
                "project_id": r.project_id,
                "version": r.version,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            }
            for r in records
        ]


async def delete_plan(plan_id: str) -> bool:
    """Delete a plan and all its entries/versions."""
    async with get_session() as session:
        # Delete versions
        stmt_v = sa.delete(AnalysisVersionRecord).where(
            AnalysisVersionRecord.analysis_id.in_(
                sa.select(AnalysisEntryRecord.id).where(AnalysisEntryRecord.plan_id == plan_id)
            )
        )
        await session.execute(stmt_v)

        # Delete entries
        stmt_e = sa.delete(AnalysisEntryRecord).where(AnalysisEntryRecord.plan_id == plan_id)
        await session.execute(stmt_e)

        # Delete plan
        stmt_p = sa.delete(AnalysisPlanRecord).where(AnalysisPlanRecord.id == plan_id)
        result = await session.execute(stmt_p)

        await session.commit()
        return result.rowcount > 0


# ---------------------------------------------------------------------------
# Entry CRUD
# ---------------------------------------------------------------------------

async def _save_entry(session, plan_id: str, entry: AnalysisEntry):
    """Persist a single AnalysisEntry and its versions."""
    existing = await session.get(AnalysisEntryRecord, entry.analysis_id)
    if existing:
        existing.analysis_name = entry.analysis_name
        existing.analysis_class = entry.analysis_class
        existing.prespecified_status = entry.prespecified_status
        existing.causal_language_flag = entry.causal_language_flag
        existing.multiple_testing_strategy = entry.multiple_testing_strategy
        existing.decision_trace = entry.decision_trace.model_dump_json() if entry.decision_trace else None
        existing.power_assessment = json.dumps(entry.power_assessment) if entry.power_assessment else None
        existing.updated_at = datetime.now(timezone.utc)
    else:
        record = AnalysisEntryRecord(
            id=entry.analysis_id,
            plan_id=plan_id,
            analysis_name=entry.analysis_name,
            analysis_class=entry.analysis_class,
            current_version_id=entry.version_history.current_version_id,
            prespecified_status=entry.prespecified_status,
            causal_language_flag=entry.causal_language_flag,
            multiple_testing_strategy=entry.multiple_testing_strategy,
            decision_trace=entry.decision_trace.model_dump_json() if entry.decision_trace else None,
            power_assessment=json.dumps(entry.power_assessment) if entry.power_assessment else None,
        )
        session.add(record)

    # Save all versions
    for version in entry.version_history.versions:
        await _save_version(session, entry.analysis_id, version)


async def _load_entry(session, record: AnalysisEntryRecord) -> AnalysisEntry:
    """Load an AnalysisEntry from a database record."""
    decision_trace = None
    if record.decision_trace:
        try:
            decision_trace = AnalysisDecisionTrace.model_validate_json(record.decision_trace)
        except Exception:
            pass

    entry = AnalysisEntry(
        analysis_id=record.id,
        analysis_name=record.analysis_name,
        analysis_class=record.analysis_class,
        prespecified_status=record.prespecified_status,
        causal_language_flag=record.causal_language_flag,
        multiple_testing_strategy=record.multiple_testing_strategy,
        decision_trace=decision_trace,
        power_assessment=json.loads(record.power_assessment) if record.power_assessment else {},
        created_at=record.created_at.isoformat() if record.created_at else "",
        updated_at=record.updated_at.isoformat() if record.updated_at else "",
    )

    # Override the default version that model_post_init created
    entry.version_history = VersionHistory()

    # Load versions
    stmt = sa.select(AnalysisVersionRecord).where(
        AnalysisVersionRecord.analysis_id == record.id
    ).order_by(AnalysisVersionRecord.created_at)
    result = await session.execute(stmt)
    version_records = result.scalars().all()

    for v_rec in version_records:
        version = _load_version(v_rec)
        entry.version_history.versions.append(version)

    entry.version_history.current_version_id = record.current_version_id

    return entry


# ---------------------------------------------------------------------------
# Version CRUD
# ---------------------------------------------------------------------------

async def _save_version(session, analysis_id: str, version: AnalysisVersion):
    """Persist a single AnalysisVersion."""
    existing = await session.get(AnalysisVersionRecord, version.version_id)
    spec_json = version.specification.model_dump_json() if version.specification else "{}"

    if existing:
        existing.version_number = version.version_number
        existing.specification = spec_json
        existing.validation_status = version.validation_status.value
        existing.validation_warnings = json.dumps(version.validation_warnings)
        existing.validation_blockers = json.dumps(version.validation_blockers)
        existing.approval_status = version.approval_status.value
        existing.lifecycle = version.lifecycle.value
        existing.modified_by = version.modified_by
        existing.modification_reason = version.modification_reason
        existing.dataset_id = version.dataset_id
        existing.dataset_version = version.dataset_version
        existing.dataset_content_hash = version.dataset_hash
    else:
        record = AnalysisVersionRecord(
            id=version.version_id,
            analysis_id=analysis_id,
            version_number=version.version_number,
            specification=spec_json,
            validation_status=version.validation_status.value,
            validation_warnings=json.dumps(version.validation_warnings),
            validation_blockers=json.dumps(version.validation_blockers),
            approval_status=version.approval_status.value,
            lifecycle=version.lifecycle.value,
            modified_by=version.modified_by,
            modification_reason=version.modification_reason,
            dataset_id=version.dataset_id,
            dataset_version=version.dataset_version,
            dataset_content_hash=version.dataset_hash,
        )
        session.add(record)


def _load_version(record: AnalysisVersionRecord) -> AnalysisVersion:
    """Load an AnalysisVersion from a database record."""
    spec = None
    if record.specification:
        try:
            spec = StatisticalAnalysisSpecification.model_validate_json(record.specification)
        except Exception:
            spec = StatisticalAnalysisSpecification(
                analysis_method="unknown",
                primary_outcome="unknown",
            )

    return AnalysisVersion(
        version_id=record.id,
        version_number=record.version_number,
        specification=spec,
        validation_status=ValidationStatus(record.validation_status),
        validation_warnings=json.loads(record.validation_warnings) if record.validation_warnings else [],
        validation_blockers=json.loads(record.validation_blockers) if record.validation_blockers else [],
        approval_status=ApprovalStatus(record.approval_status),
        lifecycle=AnalysisLifecycle(record.lifecycle),
        created_at=record.created_at.isoformat() if record.created_at else "",
        modified_by=record.modified_by,
        modification_reason=record.modification_reason,
        dataset_id=record.dataset_id,
        dataset_version=record.dataset_version,
        dataset_hash=record.dataset_content_hash,
    )


# ---------------------------------------------------------------------------
# Version supersedence
# ---------------------------------------------------------------------------

async def supersede_version(analysis_id: str, old_version_id: str):
    """Mark a version as superseded and create a new version entry."""
    async with get_session() as session:
        record = await session.get(AnalysisVersionRecord, old_version_id)
        if record:
            record.superseded_at = datetime.now(timezone.utc)
            await session.commit()
