"""Audit Ledger — immutable-style record of every analytical decision."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import sqlalchemy as sa

from app.core.database import get_session
from app.models import AuditEntry


async def log_audit(
    *,
    action: str,
    agent: str,
    decision: str,
    input_summary: str | None = None,
    rationale: str | None = None,
    output_summary: str | None = None,
    confidence: str | None = None,
    warnings: str | list[str] | None = None,
    session_id: str | None = None,
    project_id: str | None = None,
    researcher_decision: str | None = None,
) -> str:
    """Insert an audit entry and return its id."""
    if isinstance(warnings, list):
        warnings = json.dumps(warnings) if warnings else None

    entry = AuditEntry(
        session_id=session_id,
        project_id=project_id,
        action=action,
        agent=agent,
        input_summary=input_summary,
        decision=decision,
        rationale=rationale,
        output_summary=output_summary,
        confidence=confidence,
        warnings=warnings,
        researcher_decision=researcher_decision,
    )

    async with get_session() as session:
        session.add(entry)
        await session.commit()
        await session.refresh(entry)
        return entry.id


async def get_audit_log(
    project_id: str | None = None,
    session_id: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Retrieve audit entries, most recent first."""
    async with get_session() as session:
        stmt = sa.select(AuditEntry).order_by(AuditEntry.timestamp.desc())
        if project_id:
            stmt = stmt.where(AuditEntry.project_id == project_id)
        if session_id:
            stmt = stmt.where(AuditEntry.session_id == session_id)
        stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "action": r.action,
                "agent": r.agent,
                "decision": r.decision,
                "rationale": r.rationale,
                "confidence": r.confidence,
                "warnings": r.warnings,
                "researcher_decision": r.researcher_decision,
            }
            for r in rows
        ]
