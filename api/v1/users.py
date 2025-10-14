from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import guard_service, get_session
from schemas.user import UserCreate, UserRead, UserUpdate
from services import UserService

router = APIRouter()


@router.post("/", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    session: AsyncSession = Depends(get_session),
) -> UserRead:
    return await guard_service(UserService.create(session, payload))


@router.get("/", response_model=List[UserRead])
async def list_users(
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
) -> List[UserRead]:
    return await guard_service(UserService.list(session, limit=limit, offset=offset))


@router.get("/{user_id}", response_model=UserRead)
async def get_user(
    user_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> UserRead:
    return await guard_service(UserService.get(session, user_id))


@router.patch("/{user_id}", response_model=UserRead)
async def update_user(
    user_id: UUID,
    payload: UserUpdate,
    session: AsyncSession = Depends(get_session),
) -> UserRead:
    return await guard_service(UserService.update(session, user_id, payload))


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> Response:
    await guard_service(UserService.delete(session, user_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)