"""Data Quality Store — persistent, auditable, dataset-bound storage for data quality
profiles, transformation proposals, and derived dataset lineage.

Phase 2D.1 hardening:
- All state lives in SQLite (no in-memory dicts)
- Proposals are bound to a specific dataset identity (content_hash + schema_hash)
- Proposal lifecycle is an explicit state machine with allowed transitions
- Transformation application is transactional with rollback on failure
- Repeated application is idempotent (deterministic fingerprint)
- Anti-p-hacking verdict is stored alongside each proposal
- Lineage is always complete (every apply creates an immutable audit trail)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import sqlalchemy as sa

from app.core.database import get_session
from app.models import (
    TransformationRecord,
    TransformationProposalRecord,
    QualityProfileRecord,
)
from app.data.dataset_identity import compute_dataset_identity
from app.statistics.data_quality import (
    DataQualityProfile,
    HandlingStrategy,
    TransformationCandidate,
)


# ---------------------------------------------------------------------------
# Proposal Lifecycle State Machine
# ---------------------------------------------------------------------------

_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "detected": {"proposed"},
    "proposed": {"review_required", "rejected"},
    "review_required": {"approved", "rejected"},
    "approved": {"applied", "application_failed", "stale"},
    "rejected": {"proposed"},
    "stale": {"proposed"},
    "applied": set(),           # terminal
    "application_failed": {"approved", "proposed"},  # can retry
}


def _validate_transition(current: str, target: str) -> None:
    """Raise if the transition is not allowed."""
    allowed = _ALLOWED_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise ValueError(
            f"Invalid state transition: {current!r} → {target!r}. "
            f"Allowed: {sorted(allowed) or '(terminal)'}"
        )


def _audit_event(from_status: str, to_status: str, reason: str | None = None) -> dict:
    """Create an audit event entry for a state transition."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "from": from_status,
        "to": to_status,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Idempotency Fingerprint
# ---------------------------------------------------------------------------

def _compute_fingerprint(dataset_id: str, operation: str, variable: str, parameters: dict) -> str:
    """Deterministic fingerprint for a transformation proposal.

    Ensures repeated identical proposals are detected and deduplicated.
    """
    canonical = json.dumps(
        {"dataset_id": dataset_id, "operation": operation, "variable": variable, "parameters": parameters},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Data Quality Profile Persistence
# ---------------------------------------------------------------------------

async def save_quality_profile(
    profile: DataQualityProfile,
    dataset_id: str,
    dataset_content_hash: str | None = None,
    dataset_schema_hash: str | None = None,
) -> str:
    """Persist a data quality profile to DB. Returns profile_id."""
    record = QualityProfileRecord(
        dataset_id=dataset_id,
        dataset_content_hash=dataset_content_hash,
        dataset_schema_hash=dataset_schema_hash,
        profile_json=profile.model_dump_json(),
    )
    async with get_session() as session:
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record.id


async def get_quality_profile(profile_id: str) -> dict | None:
    """Retrieve a data quality profile by its ID."""
    async with get_session() as session:
        stmt = sa.select(QualityProfileRecord).where(QualityProfileRecord.id == profile_id)
        result = await session.execute(stmt)
        record = result.scalar_one_or_none()
        if not record:
            return None
        return {
            "profile_id": record.id,
            "dataset_id": record.dataset_id,
            "dataset_content_hash": record.dataset_content_hash,
            "dataset_schema_hash": record.dataset_schema_hash,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "profile": json.loads(record.profile_json),
        }


async def get_quality_profile_by_dataset(dataset_id: str) -> dict | None:
    """Retrieve the most recent data quality profile for a dataset."""
    async with get_session() as session:
        stmt = (
            sa.select(QualityProfileRecord)
            .where(QualityProfileRecord.dataset_id == dataset_id)
            .order_by(QualityProfileRecord.created_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        record = result.scalar_one_or_none()
        if not record:
            return None
        return {
            "profile_id": record.id,
            "dataset_id": record.dataset_id,
            "dataset_content_hash": record.dataset_content_hash,
            "dataset_schema_hash": record.dataset_schema_hash,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "profile": json.loads(record.profile_json),
        }


# ---------------------------------------------------------------------------
# Transformation Proposal — Lifecycle Management
# ---------------------------------------------------------------------------

async def create_transformation_proposal(
    *,
    dataset_id: str,
    source_content_hash: str,
    source_schema_hash: str,
    operation: str,
    variable: str,
    strategy: HandlingStrategy,
    candidate: TransformationCandidate | None = None,
    parameters: dict[str, Any] | None = None,
    reason: str | None = None,
) -> str:
    """Create a transformation proposal in DETECTED status.

    Returns proposal_id. Enforces idempotency: if an identical proposal
    already exists (same fingerprint, non-terminal status), returns existing ID.
    """
    params = parameters or {}
    fingerprint = _compute_fingerprint(dataset_id, operation, variable, params)

    # Idempotency: check for existing active proposal with same fingerprint
    async with get_session() as session:
        stmt = (
            sa.select(TransformationProposalRecord)
            .where(TransformationProposalRecord.application_fingerprint == fingerprint)
            .where(
                TransformationProposalRecord.status.notin_(
                    ["rejected", "stale", "applied", "approved"]
                )
            )
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing:
            return existing.id

    record = TransformationProposalRecord(
        dataset_id=dataset_id,
        source_content_hash=source_content_hash,
        source_schema_hash=source_schema_hash,
        operation=operation,
        variable=variable,
        parameters=json.dumps(params),
        strategy_json=strategy.model_dump_json(),
        candidate_json=candidate.model_dump_json() if candidate else None,
        reason=reason,
        status="detected",
        application_fingerprint=fingerprint,
        audit_events=json.dumps([_audit_event("none", "detected", reason)]),
    )
    async with get_session() as session:
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record.id


async def transition_proposal(
    proposal_id: str,
    target_status: str,
    *,
    reason: str | None = None,
    reviewer: str | None = None,
) -> dict:
    """Transition a proposal to a new status.

    Returns the updated proposal dict or raises ValueError on invalid transition.
    """
    async with get_session() as session:
        stmt = sa.select(TransformationProposalRecord).where(
            TransformationProposalRecord.id == proposal_id
        )
        result = await session.execute(stmt)
        record = result.scalar_one_or_none()
        if not record:
            raise ValueError(f"Proposal {proposal_id} not found")

        current = record.status
        _validate_transition(current, target_status)

        # Apply transition
        record.status = target_status
        record.updated_at = datetime.now(timezone.utc)

        # Side effects per transition
        if target_status == "approved":
            record.reviewed_at = datetime.now(timezone.utc)
            record.reviewer = reviewer
        elif target_status == "rejected":
            record.reviewed_at = datetime.now(timezone.utc)
            record.reviewer = reviewer
            record.rejection_reason = reason
        elif target_status == "stale":
            record.rejection_reason = reason or "Dataset changed since proposal creation"

        # Append audit event
        events = json.loads(record.audit_events or "[]")
        events.append(_audit_event(current, target_status, reason))
        record.audit_events = json.dumps(events)

        await session.commit()
        await session.refresh(record)
        return _record_to_dict(record)


async def approve_transformation(proposal_id: str, reviewer: str = "researcher") -> dict | None:
    """Approve a transformation proposal (PROPOSED/REVIEW_REQUIRED → APPROVED)."""
    try:
        return await transition_proposal(proposal_id, "approved", reviewer=reviewer)
    except (ValueError, KeyError):
        return None


async def reject_transformation(proposal_id: str, reason: str = "", reviewer: str = "researcher") -> dict | None:
    """Reject a transformation proposal."""
    try:
        return await transition_proposal(proposal_id, "rejected", reason=reason, reviewer=reviewer)
    except (ValueError, KeyError):
        return None


async def mark_proposal_review_required(proposal_id: str) -> dict | None:
    """Move proposal to REVIEW_REQUIRED (proposed → review_required)."""
    try:
        return await transition_proposal(proposal_id, "review_required")
    except (ValueError, KeyError):
        return None


async def mark_proposal_stale(proposal_id: str, reason: str | None = None) -> dict | None:
    """Mark proposal as STALE (approved → stale) when dataset changes."""
    try:
        return await transition_proposal(proposal_id, "stale", reason=reason)
    except (ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Dataset-Binding Integrity
# ---------------------------------------------------------------------------

async def verify_proposal_integrity(proposal_id: str, current_file_path: str) -> tuple[bool, str]:
    """Verify that a proposal still matches the dataset it was created for.

    Returns (is_valid, reason).
    """
    async with get_session() as session:
        stmt = sa.select(TransformationProposalRecord).where(
            TransformationProposalRecord.id == proposal_id
        )
        result = await session.execute(stmt)
        record = result.scalar_one_or_none()
        if not record:
            return False, f"Proposal {proposal_id} not found"

        current_identity = compute_dataset_identity(current_file_path)

        if record.source_content_hash != current_identity["content_hash"]:
            return False, (
                f"Dataset content has changed since proposal creation. "
                f"Expected {record.source_content_hash[:16]}, "
                f"got {current_identity['content_hash'][:16]}"
            )
        if record.source_schema_hash != current_identity["schema_hash"]:
            return False, (
                f"Dataset schema has changed since proposal creation. "
                f"Expected {record.source_schema_hash[:16]}, "
                f"got {current_identity['schema_hash'][:16]}"
            )

        return True, "Dataset identity matches proposal"


# ---------------------------------------------------------------------------
# Transformation Application — Transactional
# ---------------------------------------------------------------------------

async def apply_transformation_proposal(
    proposal_id: str,
    input_file_path: str,
) -> dict | None:
    """Apply an approved transformation proposal to create a derived dataset.

    Transactional: all guardrails must pass before any write occurs.
    On failure, no derived dataset is created and status → application_failed.

    Idempotent: if this proposal was already applied successfully, returns
    the existing result without re-applying.
    """
    async with get_session() as session:
        stmt = sa.select(TransformationProposalRecord).where(
            TransformationProposalRecord.id == proposal_id
        )
        result = await session.execute(stmt)
        record = result.scalar_one_or_none()
        if not record:
            return None

        # Idempotency: already applied
        if record.status == "applied" and record.derived_dataset_id:
            return {
                "success": True,
                "proposal_id": proposal_id,
                "derived_dataset_id": record.derived_dataset_id,
                "derived_content_hash": record.derived_content_hash,
                "derived_schema_hash": record.derived_schema_hash,
                "impact_summary": json.loads(record.impact_summary or "{}"),
                "idempotent": True,
            }

        if record.status != "approved":
            return {"error": f"Cannot apply proposal in '{record.status}' status"}

    # --- Step 1: Verify dataset identity (outside transaction) ---
    is_valid, reason = await verify_proposal_integrity(proposal_id, input_file_path)
    if not is_valid:
        await transition_proposal(proposal_id, "application_failed", reason=reason)
        return {"error": f"Dataset integrity check failed: {reason}"}

    # --- Step 2: Load source dataset ---
    from pathlib import Path
    try:
        df = pd.read_csv(input_file_path)
    except Exception as e:
        await transition_proposal(proposal_id, "application_failed", reason=f"Load failed: {e}")
        return {"error": f"Failed to load dataset: {e}"}

    # --- Step 3: Re-read proposal for current parameters ---
    async with get_session() as session:
        stmt = sa.select(TransformationProposalRecord).where(
            TransformationProposalRecord.id == proposal_id
        )
        result = await session.execute(stmt)
        record = result.scalar_one_or_none()

    operation = record.operation
    variable = record.variable
    parameters = json.loads(record.parameters or "{}")

    # --- Step 4: Execute transformation (deterministic engine) ---
    from app.statistics.transformations import (
        apply_transformation,
        apply_imputation,
        apply_exclusion,
        apply_winsorize,
    )

    try:
        if operation in ("log", "log1p", "sqrt", "standardize", "min_max", "yeo_johnson", "box_cox"):
            from app.statistics.data_quality import TransformationType
            t_type = TransformationType(operation)
            derived_df, fitted_params = apply_transformation(df, variable, t_type, parameters)
        elif operation in ("mean_impute", "median_impute", "mode_impute"):
            method = operation.replace("_impute", "")
            derived_df, fitted_params = apply_imputation(df, variable, method=method)
        elif operation == "exclude_observations":
            indices = parameters.get("indices", [])
            derived_df, fitted_params = apply_exclusion(df, indices)
        elif operation == "winsorize":
            derived_df, fitted_params = apply_winsorize(
                df, variable,
                lower_pct=parameters.get("lower_pct", 0.05),
                upper_pct=parameters.get("upper_pct", 0.95),
            )
        else:
            return {"error": f"Unknown operation: {operation}"}
    except Exception as e:
        await transition_proposal(proposal_id, "application_failed", reason=f"Engine failed: {e}")
        return {"error": f"Transformation failed: {e}"}

    # --- Step 5: Anti-p-hacking check (before writing) ---
    from app.statistics.anti_phacking import AntiPHackingGuard
    guard = AntiPHackingGuard()
    ap_checks = guard.check_transformation_selection(
        transformations=[{"operation": operation, "variable": variable}],
        selected={"operation": operation, "variable": variable},
    )
    ap_report = guard.generate_report(ap_checks)

    if ap_report["overall_verdict"] == "blocked":
        await transition_proposal(proposal_id, "application_failed", reason="Anti-p-hacking BLOCKED")
        return {"error": "Transformation blocked by anti-p-hacking guard", "verdict": ap_report}

    # --- Step 6: Write derived dataset + compute identity (atomic) ---
    derived_path = Path(input_file_path).parent / f"derived_{proposal_id}.csv"
    try:
        derived_df.to_csv(derived_path, index=False)
    except Exception as e:
        await transition_proposal(proposal_id, "application_failed", reason=f"Write failed: {e}")
        return {"error": f"Failed to write derived dataset: {e}"}

    derived_identity = compute_dataset_identity(derived_path)

    # --- Step 7: Persist transformation record + update proposal (single transaction) ---
    impact = {
        "n_rows_before": len(df),
        "n_rows_after": len(derived_df),
        "n_cols_before": len(df.columns),
        "n_cols_after": len(derived_df.columns),
    }

    async with get_session() as session:
        # Re-read proposal in this session for proper attachment
        stmt = sa.select(TransformationProposalRecord).where(
            TransformationProposalRecord.id == proposal_id
        )
        result = await session.execute(stmt)
        record = result.scalar_one_or_none()

        # Persist transformation record
        t_record = TransformationRecord(
            operation=operation,
            input_dataset_id=record.source_content_hash[:16],
            input_dataset_version=1,
            output_dataset_id=derived_identity["content_hash"][:16],
            output_dataset_version=1,
            parameters=json.dumps({"variable": variable, **fitted_params, "proposal_id": proposal_id}),
            agent="data_quality_engine",
            researcher_approved=True,
            output_content_hash=derived_identity["content_hash"],
            output_schema_hash=derived_identity["schema_hash"],
        )
        session.add(t_record)

        # Update proposal record
        record.status = "applied"
        record.applied_at = datetime.now(timezone.utc)
        record.derived_dataset_id = derived_identity["content_hash"][:16]
        record.derived_content_hash = derived_identity["content_hash"]
        record.derived_schema_hash = derived_identity["schema_hash"]
        record.impact_summary = json.dumps(impact)
        record.anti_p_hacking_verdict = json.dumps(ap_report)

        events = json.loads(record.audit_events or "[]")
        events.append(_audit_event("approved", "applied", "Transactional apply succeeded"))
        record.audit_events = json.dumps(events)

        await session.commit()

    return {
        "success": True,
        "proposal_id": proposal_id,
        "derived_path": str(derived_path),
        "derived_identity": derived_identity,
        "fitted_parameters": fitted_params,
        "impact_summary": impact,
        "anti_p_hacking_verdict": ap_report,
        "idempotent": False,
    }


# ---------------------------------------------------------------------------
# Retrieval Helpers
# ---------------------------------------------------------------------------

async def get_transformation_proposal(proposal_id: str) -> dict | None:
    """Retrieve a transformation proposal by ID."""
    async with get_session() as session:
        stmt = sa.select(TransformationProposalRecord).where(
            TransformationProposalRecord.id == proposal_id
        )
        result = await session.execute(stmt)
        record = result.scalar_one_or_none()
        if not record:
            return None
        return _record_to_dict(record)


async def list_transformation_proposals(dataset_id: str | None = None) -> list[dict]:
    """List transformation proposals, optionally filtered by dataset."""
    async with get_session() as session:
        stmt = sa.select(TransformationProposalRecord).order_by(
            TransformationProposalRecord.created_at.desc()
        )
        if dataset_id:
            stmt = stmt.where(TransformationProposalRecord.dataset_id == dataset_id)
        result = await session.execute(stmt)
        records = result.scalars().all()
        return [_record_to_dict(r) for r in records]


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------

async def get_dataset_lineage(dataset_id: str) -> list[dict]:
    """Get transformation lineage for a dataset.

    Returns a chain of transformation records showing parent → child relationships.
    """
    async with get_session() as session:
        stmt = (
            sa.select(TransformationRecord)
            .where(
                (TransformationRecord.input_dataset_id == dataset_id)
                | (TransformationRecord.output_dataset_id == dataset_id)
            )
            .order_by(TransformationRecord.created_at)
        )
        result = await session.execute(stmt)
        records = result.scalars().all()

        return [
            {
                "id": r.id,
                "operation": r.operation,
                "input_dataset_id": r.input_dataset_id,
                "input_dataset_version": r.input_dataset_version,
                "output_dataset_id": r.output_dataset_id,
                "output_dataset_version": r.output_dataset_version,
                "parameters": json.loads(r.parameters) if r.parameters else {},
                "researcher_approved": r.researcher_approved,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _record_to_dict(record: TransformationProposalRecord) -> dict:
    """Convert a DB record to a plain dict for API responses."""
    return {
        "proposal_id": record.id,
        "dataset_id": record.dataset_id,
        "source_content_hash": record.source_content_hash,
        "source_schema_hash": record.source_schema_hash,
        "operation": record.operation,
        "variable": record.variable,
        "status": record.status,
        "parameters": json.loads(record.parameters or "{}"),
        "strategy": json.loads(record.strategy_json) if record.strategy_json else None,
        "candidate": json.loads(record.candidate_json) if record.candidate_json else None,
        "reason": record.reason,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        "reviewed_at": record.reviewed_at.isoformat() if record.reviewed_at else None,
        "reviewer": record.reviewer,
        "rejection_reason": record.rejection_reason,
        "applied_at": record.applied_at.isoformat() if record.applied_at else None,
        "derived_dataset_id": record.derived_dataset_id,
        "derived_content_hash": record.derived_content_hash,
        "derived_schema_hash": record.derived_schema_hash,
        "impact_summary": json.loads(record.impact_summary) if record.impact_summary else None,
        "application_error": record.application_error,
        "application_fingerprint": record.application_fingerprint,
        "anti_p_hacking_verdict": json.loads(record.anti_p_hacking_verdict) if record.anti_p_hacking_verdict else None,
        "audit_events": json.loads(record.audit_events or "[]"),
    }
