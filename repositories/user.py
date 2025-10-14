from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import User
from schemas.user import UserCreate, UserUpdate


async def create_user(session: AsyncSession, payload: UserCreate) -> User:
    user = User(**payload.dict())
    session.add(user)
    await session.flush()
    await session.refresh(user)
    return user


async def get_user(session: AsyncSession, user_id: UUID) -> Optional[User]:
    result = await session.execute(select(User).where(User.user_id == user_id))
    return result.scalar_one_or_none()


async def get_user_by_email(session: AsyncSession, email: str) -> Optional[User]:
    result = await session.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def list_users(session: AsyncSession, limit: int = 100, offset: int = 0) -> List[User]:
    stmt = (
        select(User)
        .order_by(User.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def update_user(session: AsyncSession, user: User, payload: UserUpdate) -> User:
    for field, value in payload.dict(exclude_unset=True).items():
        setattr(user, field, value)
    await session.flush()
    await session.refresh(user)
    return user


async def delete_user(session: AsyncSession, user: User) -> None:
    await session.delete(user)
    await session.flush()