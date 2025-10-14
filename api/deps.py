from __future__ import annotations

from typing import Awaitable, TypeVar

from fastapi import HTTPException

from services import NotFoundError, ServiceError, ValidationError

T = TypeVar("T")


async def guard_service(call: Awaitable[T]) -> T:
    try:
        return await call
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc))