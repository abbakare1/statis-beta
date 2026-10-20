"""Persistent memory system — project, system, and session memory."""

from __future__ import annotations

import json

import sqlalchemy as sa

from app.core.database import get_session
from app.models import MemoryEntry


async def store_memory(
    *,
    memory_type: str,
    category: str,
    key: str,
    value: str | dict | list,
    source: str = "ai_interpretation",
    confidence: str | None = None,
    project_id: str | None = None,
) -> str:
    """Store a memory entry. Returns the entry id."""
    if not isinstance(value, str):
        value = json.dumps(value)

    entry = MemoryEntry(
        project_id=project_id,
        memory_type=memory_type,
        category=category,
        key=key,
        value=value,
        source=source,
        confidence=confidence,
    )

    async with get_session() as session:
        session.add(entry)
        await session.commit()
        await session.refresh(entry)
        return entry.id


async def retrieve_memory(
    *,
    memory_type: str | None = None,
    category: str | None = None,
    key: str | None = None,
    project_id: str | None = None,
    limit: int = 500,
) -> list[dict]:
    """Query memory entries."""
    async with get_session() as session:
        stmt = sa.select(MemoryEntry)
        if memory_type:
            stmt = stmt.where(MemoryEntry.memory_type == memory_type)
        if category:
            stmt = stmt.where(MemoryEntry.category == category)
        if key:
            stmt = stmt.where(MemoryEntry.key == key)
        if project_id:
            stmt = stmt.where(MemoryEntry.project_id == project_id)
        stmt = stmt.order_by(MemoryEntry.created_at.desc()).limit(limit)
        result = await session.execute(stmt)
        rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "memory_type": r.memory_type,
                "category": r.category,
                "key": r.key,
                "value": r.value,
                "source": r.source,
                "confidence": r.confidence,
                "project_id": r.project_id,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


async def update_memory(memory_id: str, **fields) -> bool:
    """Update specific fields of a memory entry."""
    async with get_session() as session:
        stmt = sa.select(MemoryEntry).where(MemoryEntry.id == memory_id)
        result = await session.execute(stmt)
        entry = result.scalar_one_or_none()
        if entry is None:
            return False
        for k, v in fields.items():
            if k in ("value",) and not isinstance(v, str):
                v = json.dumps(v)
            setattr(entry, k, v)
        await session.commit()
        return True


async def delete_memory(memory_id: str) -> bool:
    """Delete a memory entry."""
    async with get_session() as session:
        stmt = sa.delete(MemoryEntry).where(MemoryEntry.id == memory_id)
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount > 0
