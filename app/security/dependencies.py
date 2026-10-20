"""FastAPI authentication dependencies."""

import sqlalchemy as sa
from datetime import datetime, timezone
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.security.auth import decode_access_token, get_user_by_id
from app.models import Project, User

security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_db_session),
) -> User:
    """Get the current authenticated user from the JWT token."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = payload.get("user_id")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = await get_user_by_id(session, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=401, detail="User is inactive")
    # Check beta access expiration
    if user.access_expires_at and user.access_expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Beta access has expired")
    
    return user


async def get_current_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Get the current authenticated admin user."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


async def validate_project_ownership(project_id: str, current_user: User) -> Project:
    """Validate that the current user owns the specified project."""
    from app.core.database import get_session
    async with get_session() as session:
        result = await session.execute(
            sa.select(Project).where(Project.id == project_id, Project.owner_id == current_user.id)
        )
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project


async def get_project_owner(
    project_id: str,
    current_user: User = Depends(get_current_user),
) -> Project:
    """FastAPI dependency: validate that the current user owns the specified project."""
    return await validate_project_ownership(project_id, current_user)
