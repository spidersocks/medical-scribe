from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Coroutine, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from .exceptions import ServiceError

T = TypeVar("T")


@asynccontextmanager
async def transactional_session(session: AsyncSession) -> AsyncIterator[AsyncSession]:
    """
    Helper context manager for manual transaction boundaries.
    Usage:
        async with transactional_session(session) as tx:
            ...
    """
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise


async def run_in_transaction(
    session: AsyncSession,
    func: Callable[[AsyncSession], Coroutine[None, None, T]],
) -> T:
    """
    Execute a coroutine with automatic commit/rollback.
    """
    async with transactional_session(session) as tx:
        return await func(tx)


def unwrap_or_raise(obj, *, resource_name: str):
    """
    Convenience helper to convert None into a NotFoundError.
    """
    if obj is None:
        from .exceptions import NotFoundError

        raise NotFoundError(f"{resource_name} not found.")
    return obj


def ensure_condition(condition: bool, message: str):
    """
    Raise ValidationError if condition is False.
    """
    if not condition:
        from .exceptions import ValidationError

        raise ValidationError(message)