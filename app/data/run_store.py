"""Run Store — persistence layer for analysis runs, reproducibility fingerprints, and decision traces.

Phase 2E: Every analysis execution is captured as an immutable AnalysisRun record
with deterministic fingerprints that enable exact reproduction and comparison.
"""

from __future__ import annotations

import hashlib
import json
import sys
import platform
from datetime import datetime, timezone
from typing import Any

import sqlalchemy as sa

from app.core.database import get_session
from app.models import AnalysisRunRecord


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

def compute_input_fingerprint(
    specification: dict,
    dataset_content_hash: str,
    dataset_schema_hash: str,
    extra_context: dict | None = None,
) -> str:
    """Compute a deterministic SHA-256 fingerprint of all inputs to an analysis run.

    This fingerprint is invariant to execution order and captures everything
    needed to reproduce the result: specification, dataset identity, and
    any additional context that affects the outcome.
    """
    canonical = {
        "specification": _canonicalize(specification),
        "dataset_content_hash": dataset_content_hash,
        "dataset_schema_hash": dataset_schema_hash,
    }
    if extra_context:
        canonical["extra_context"] = _canonicalize(extra_context)

    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def compute_result_fingerprint(result: dict) -> str:
    """Compute a deterministic fingerprint of an analysis result.

    Two identical analyses on identical data must produce the same result fingerprint.
    """
    canonical = _canonicalize(result)
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def _json_safe(obj: Any) -> Any:
    """Convert to JSON-serializable form WITHOUT precision loss.

    Handles numpy scalars (np.int64/np.float64/np.bool_ from pandas/statsmodels),
    tuples, and non-finite floats (NaN/Inf → null — not JSON-compliant).
    Used when storing results; rounding is reserved for fingerprints.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_json_safe(item) for item in obj]
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    elif isinstance(obj, int):
        return obj
    elif hasattr(obj, "item") and callable(obj.item):
        try:
            return _json_safe(obj.item())
        except (ValueError, TypeError):
            return obj
    return obj


def _canonicalize(obj: Any) -> Any:
    """Fingerprint canonicalization: sorted keys + rounded floats + JSON-safe.

    Rounding to 10 decimal places gives cross-platform fingerprint stability;
    this is ONLY for fingerprints, never for stored result payloads.
    """
    if isinstance(obj, dict):
        return {k: _canonicalize(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, (list, tuple)):
        return [_canonicalize(item) for item in obj]
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, float):
        # NaN/Inf are not JSON-compliant; undefined stats → null in fingerprints
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        # Round floats to 10 decimal places for cross-platform stability
        return round(obj, 10)
    elif isinstance(obj, int):
        return obj
    elif hasattr(obj, "item") and callable(obj.item):
        # numpy scalar → Python native, then re-canonicalize
        try:
            return _canonicalize(obj.item())
        except (ValueError, TypeError):
            return obj
    return obj


# ---------------------------------------------------------------------------
# Environment Capture
# ---------------------------------------------------------------------------

def capture_environment() -> dict:
    """Capture the current execution environment for reproducibility metadata."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "engine_version": "2e.1",
    }


# ---------------------------------------------------------------------------
# CRUD Operations
# ---------------------------------------------------------------------------

async def create_analysis_run(
    *,
    project_id: str,
    analysis_type: str,
    specification: dict,
    dataset_id: str | None,
    dataset_content_hash: str,
    dataset_schema_hash: str,
    file_path: str | None = None,
    session_id: str | None = None,
    extra_context: dict | None = None,
) -> str:
    """Create an analysis run record. Returns run_id.

    Automatically computes input fingerprint and captures environment.
    """
    input_fp = compute_input_fingerprint(
        specification, dataset_content_hash, dataset_schema_hash, extra_context
    )
    env = capture_environment()

    record = AnalysisRunRecord(
        project_id=project_id,
        session_id=session_id,
        analysis_type=analysis_type,
        specification_json=json.dumps(_json_safe(specification)),
        dataset_id=dataset_id,
        dataset_content_hash=dataset_content_hash,
        dataset_schema_hash=dataset_schema_hash,
        file_path=file_path,
        input_fingerprint=input_fp,
        engine_version=env["engine_version"],
        python_version=env["python_version"],
    )

    async with get_session() as session:
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record.id


async def complete_analysis_run(
    run_id: str,
    *,
    result: dict,
    success: bool = True,
    error_message: str | None = None,
    validation_status: str | None = None,
    approval_status: str | None = None,
    lifecycle: str | None = None,
    guardrails_passed: bool = True,
    warnings: list[str] | None = None,
    decision_trace: list[dict] | None = None,
) -> dict | None:
    """Mark an analysis run as completed with its result.

    Computes result fingerprint and stores the full result.
    """
    result_fp = compute_result_fingerprint(result) if success else None

    async with get_session() as session:
        stmt = sa.select(AnalysisRunRecord).where(AnalysisRunRecord.id == run_id)
        result_q = await session.execute(stmt)
        record = result_q.scalar_one_or_none()
        if not record:
            return None

        record.completed_at = datetime.now(timezone.utc)
        record.result_json = json.dumps(_json_safe(result))
        record.result_fingerprint = result_fp
        record.success = success
        record.error_message = error_message
        record.validation_status = validation_status
        record.approval_status = approval_status
        record.lifecycle = lifecycle
        record.guardrails_passed = guardrails_passed
        record.warnings_json = json.dumps(warnings) if warnings else None
        record.decision_trace_json = json.dumps(_json_safe(decision_trace)) if decision_trace else None

        await session.commit()
        await session.refresh(record)
        return _record_to_dict(record)


async def get_analysis_run(run_id: str) -> dict | None:
    """Retrieve an analysis run by ID."""
    async with get_session() as session:
        stmt = sa.select(AnalysisRunRecord).where(AnalysisRunRecord.id == run_id)
        result = await session.execute(stmt)
        record = result.scalar_one_or_none()
        if not record:
            return None
        return _record_to_dict(record)


async def list_analysis_runs(
    project_id: str | None = None,
    analysis_type: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """List analysis runs with optional filters."""
    async with get_session() as session:
        stmt = sa.select(AnalysisRunRecord).order_by(AnalysisRunRecord.created_at.desc())
        if project_id:
            stmt = stmt.where(AnalysisRunRecord.project_id == project_id)
        if analysis_type:
            stmt = stmt.where(AnalysisRunRecord.analysis_type == analysis_type)
        stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        records = result.scalars().all()
        return [_record_to_dict(r) for r in records]


async def find_runs_by_fingerprint(input_fingerprint: str) -> list[dict]:
    """Find all runs with the same input fingerprint (reproducibility check)."""
    async with get_session() as session:
        stmt = (
            sa.select(AnalysisRunRecord)
            .where(AnalysisRunRecord.input_fingerprint == input_fingerprint)
            .order_by(AnalysisRunRecord.created_at.desc())
        )
        result = await session.execute(stmt)
        records = result.scalars().all()
        return [_record_to_dict(r) for r in records]


async def find_duplicate_runs(
    specification: dict,
    dataset_content_hash: str,
    dataset_schema_hash: str,
) -> list[dict]:
    """Find runs that used the exact same inputs (for deduplication check)."""
    fp = compute_input_fingerprint(specification, dataset_content_hash, dataset_schema_hash)
    return await find_runs_by_fingerprint(fp)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _record_to_dict(record: AnalysisRunRecord) -> dict:
    """Convert a DB record to a plain dict for API responses."""
    return {
        "run_id": record.id,
        "project_id": record.project_id,
        "session_id": record.session_id,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "completed_at": record.completed_at.isoformat() if record.completed_at else None,
        "analysis_type": record.analysis_type,
        "specification": json.loads(record.specification_json) if record.specification_json else None,
        "dataset_id": record.dataset_id,
        "dataset_content_hash": record.dataset_content_hash,
        "dataset_schema_hash": record.dataset_schema_hash,
        "file_path": record.file_path,
        "input_fingerprint": record.input_fingerprint,
        "result": json.loads(record.result_json) if record.result_json else None,
        "result_fingerprint": record.result_fingerprint,
        "success": record.success,
        "error_message": record.error_message,
        "validation_status": record.validation_status,
        "approval_status": record.approval_status,
        "lifecycle": record.lifecycle,
        "guardrails_passed": record.guardrails_passed,
        "warnings": json.loads(record.warnings_json) if record.warnings_json else None,
        "decision_trace": json.loads(record.decision_trace_json) if record.decision_trace_json else None,
        "engine_version": record.engine_version,
        "python_version": record.python_version,
    }
