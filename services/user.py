from __future__ import annotations

from typing import List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from repositories import (
    create_user,
    delete_user,
    get_user,
    get_user_by_email,
    list_users,
    update_user,
)
from schemas.user import UserCreate, UserRead, UserUpdate
from .base import unwrap_or_raise
from .exceptions import ValidationError


class UserService:
    @staticmethod
    async def create(session: AsyncSession, payload: UserCreate) -> UserRead:
        existing = await get_user_by_email(session, payload.email)
        if existing:
            raise ValidationError("A user with this email already exists.")
        user = await create_user(session, payload)
        return UserRead.from_orm(user)

    @staticmethod
    async def get(session: AsyncSession, user_id: UUID) -> UserRead:
        user = unwrap_or_raise(await get_user(session, user_id), resource_name="User")
        return UserRead.from_orm(user)

    @staticmethod
    async def list(session: AsyncSession, limit: int = 100, offset: int = 0) -> List[UserRead]:
        users = await list_users(session, limit=limit, offset=offset)
        return [UserRead.from_orm(u) for u in users]

    @staticmethod
    async def update(session: AsyncSession, user_id: UUID, payload: UserUpdate) -> UserRead:
        user = unwrap_or_raise(await get_user(session, user_id), resource_name="User")
        user = await update_user(session, user, payload)
        return UserRead.from_orm(user)

    @staticmethod
    async def delete(session: AsyncSession, user_id: UUID) -> None:
        user = unwrap_or_raise(await get_user(session, user_id), resource_name="User")
        await delete_user(session, user)