from __future__ import annotations

from typing import AsyncIterator, Awaitable, TypeVar

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from database import AsyncSessionFactory
from services import NotFoundError, ServiceError, ValidationError

T = TypeVar("T")


async def get_session() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def guard_service(call: Awaitable[T]) -> T:
    try:
        return await call
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc))