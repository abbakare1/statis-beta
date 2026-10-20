"""JWT-based authentication for STATIS."""

import os
import uuid
from datetime import datetime, timedelta, timezone

import jwt
import sqlalchemy as sa
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.models import Invitation, User

# JWT configuration
JWT_SECRET = os.environ.get("STATIS_JWT_SECRET", "statis-beta-jwt-secret-2026")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24

# Password hashing
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(password, hashed)


def create_access_token(user_id: str, role: str) -> str:
    """Create a JWT access token."""
    payload = {
        "user_id": user_id,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat": datetime.now(timezone.utc),
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode and validate a JWT access token."""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


async def get_user_by_id(session: AsyncSession, user_id: str) -> User | None:
    """Get a user by ID."""
    result = await session.execute(sa.select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def get_user_by_email(session: AsyncSession, email: str) -> User | None:
    """Get a user by email."""
    result = await session.execute(sa.select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def create_user(session: AsyncSession, email: str, password: str, display_name: str = None, role: str = "researcher") -> User:
    """Create a new user."""
    user = User(
        email=email,
        password_hash=hash_password(password),
        display_name=display_name,
        role=role,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def authenticate_user(session: AsyncSession, email: str, password: str) -> User | None:
    """Authenticate a user by email and password."""
    user = await get_user_by_email(session, email)
    if user and verify_password(password, user.password_hash) and user.is_active:
        return user
    return None


async def create_invitation(session: AsyncSession, email: str, role: str = "researcher", expires_hours: int = 72) -> Invitation:
    """Create an invitation for a new user."""
    invitation = Invitation(
        email=email,
        token=uuid.uuid4().hex,
        role=role,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=expires_hours),
    )
    session.add(invitation)
    await session.commit()
    await session.refresh(invitation)
    return invitation


async def get_invitation_by_token(session: AsyncSession, token: str) -> Invitation | None:
    """Get an invitation by token."""
    result = await session.execute(sa.select(Invitation).where(Invitation.token == token))
    return result.scalar_one_or_none()


async def redeem_invitation(session: AsyncSession, token: str, password: str, display_name: str = None) -> User | None:
    """Redeem an invitation and create a user."""
    invitation = await get_invitation_by_token(session, token)
    if not invitation:
        return None
    if invitation.used_at:
        return None
    if invitation.expires_at < datetime.now(timezone.utc):
        return None
    
    # Create user
    user = await create_user(
        session,
        email=invitation.email,
        password=password,
        display_name=display_name,
        role=invitation.role,
    )
    
    # Mark invitation as used
    invitation.used_at = datetime.now(timezone.utc)
    await session.commit()
    
    return user


async def list_users(session: AsyncSession) -> list[User]:
    """List all users."""
    result = await session.execute(sa.select(User))
    return result.scalars().all()


async def deactivate_user(session: AsyncSession, user_id: str) -> bool:
    """Deactivate a user."""
    user = await get_user_by_id(session, user_id)
    if user:
        user.is_active = False
        await session.commit()
        return True
    return False


async def activate_user(session: AsyncSession, user_id: str) -> bool:
    """Activate a user."""
    user = await get_user_by_id(session, user_id)
    if user:
        user.is_active = True
        await session.commit()
        return True
    return False
