"""Analysis plan persistence — flat-dict DB store for API endpoints.

Phase 2J: Replaces the in-memory `_plan_store` dict with database-backed
persistence so plans survive application restart.

The flat-dict structure matches what the API endpoints and frontend expect:
    {
        "plan_id": str,
        "project_id": str,
        "dataset_id": str | None,
        "research_question": str,
        "study_design": str,
        "analyses": [ { "analysis_id": str, "method": str, ... } ],
        "warnings": list[str],
        "rejected_methods": list[str],
    }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import sqlalchemy as sa

from app.core.database import get_session
from app.models import AnalysisPlanRecord


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

async def save_plan(plan: dict[str, Any]) -> str:
    """Persist a plan dict to the database. Returns plan_id."""
    plan_id = plan["plan_id"]
    now = datetime.now(timezone.utc)

    async with get_session() as session:
        record = await session.get(AnalysisPlanRecord, plan_id)
        plan_json = json.dumps(plan, default=str)

        if record:
            record.plan_json = plan_json
            record.research_question = plan.get("research_question")
            record.study_design = plan.get("study_design")
            record.warnings = json.dumps(plan.get("warnings", []))
            record.updated_at = now
        else:
            record = AnalysisPlanRecord(
                id=plan_id,
                project_id=plan.get("project_id", ""),
                research_question=plan.get("research_question"),
                study_design=plan.get("study_design"),
                warnings=json.dumps(plan.get("warnings", [])),
                plan_json=plan_json,
                created_at=now,
                updated_at=now,
            )
            session.add(record)

        await session.commit()
        return plan_id


async def load_plan(plan_id: str) -> dict[str, Any] | None:
    """Load a plan dict from the database. Returns None if not found."""
    async with get_session() as session:
        record = await session.get(AnalysisPlanRecord, plan_id)
        if not record or not record.plan_json:
            return None
        try:
            return json.loads(record.plan_json)
        except (json.JSONDecodeError, TypeError):
            return None


async def update_plan(plan_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    """Update fields on a plan dict and persist. Returns updated plan or None."""
    plan = await load_plan(plan_id)
    if plan is None:
        return None

    plan.update(updates)
    plan["updated_at"] = datetime.now(timezone.utc).isoformat()
    await save_plan(plan)
    return plan


async def update_analysis(
    plan_id: str,
    analysis_id: str,
    updates: dict[str, Any],
) -> dict[str, Any] | None:
    """Update fields on a single analysis within a plan. Returns updated plan or None."""
    plan = await load_plan(plan_id)
    if plan is None:
        return None

    for a in plan.get("analyses", []):
        if a.get("analysis_id") == analysis_id:
            a.update(updates)
            break
    else:
        return None  # analysis not found

    plan["updated_at"] = datetime.now(timezone.utc).isoformat()
    await save_plan(plan)
    return plan


async def find_analysis(analysis_id: str) -> tuple[dict, dict] | None:
    """Search all plans for an analysis. Returns (plan, analysis) or None."""
    async with get_session() as session:
        result = await session.execute(
            sa.select(AnalysisPlanRecord).where(AnalysisPlanRecord.plan_json.isnot(None))
        )
        records = result.scalars().all()

    for record in records:
        try:
            plan = json.loads(record.plan_json)
        except (json.JSONDecodeError, TypeError):
            continue
        for a in plan.get("analyses", []):
            if a.get("analysis_id") == analysis_id:
                return plan, a
    return None


async def delete_plan(plan_id: str) -> bool:
    """Delete a plan from the database. Returns True if deleted."""
    async with get_session() as session:
        record = await session.get(AnalysisPlanRecord, plan_id)
        if not record:
            return False
        await session.delete(record)
        await session.commit()
        return True
