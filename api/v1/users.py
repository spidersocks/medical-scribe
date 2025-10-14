from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Response, status

from api.deps import guard_service
from schemas.user import UserCreate, UserRead, UserUpdate
from services import user_service

router = APIRouter()


@router.post("/", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(payload: UserCreate) -> UserRead:
    return await guard_service(user_service.create(payload))


@router.get("/", response_model=List[UserRead])
async def list_users(limit: int = 100, offset: int = 0) -> List[UserRead]:
    return await guard_service(user_service.list(limit=limit, offset=offset))


@router.get("/{user_id}", response_model=UserRead)
async def get_user(user_id: UUID) -> UserRead:
    return await guard_service(user_service.get(str(user_id)))


@router.patch("/{user_id}", response_model=UserRead)
async def update_user(user_id: UUID, payload: UserUpdate) -> UserRead:
    return await guard_service(user_service.update(str(user_id), payload))


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: UUID) -> Response:
    await guard_service(user_service.delete(str(user_id)))
    return Response(status_code=status.HTTP_204_NO_CONTENT)