import os
import ssl
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker as _async_sessionmaker,
    create_async_engine,
)

load_dotenv()


def _build_urls(raw_url: str) -> Tuple[str, str, Dict[str, Any]]:
    url: URL = make_url(raw_url)

    query = dict(url.query)
    ssl_mode = os.getenv("DATABASE_SSLMODE", "require") or "require"
    query["sslmode"] = ssl_mode
    sync_url = url.set(query=query)

    async_query = dict(query)
    async_query.pop("sslmode", None)
    async_url = url.set(drivername="postgresql+asyncpg", query=async_query)

    ssl_cert_path = os.getenv("DATABASE_SSL_ROOT_CERT")
    connect_args: Dict[str, Any]

    if ssl_cert_path:
        ssl_context = ssl.create_default_context(cafile=ssl_cert_path)
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        ssl_context.check_hostname = True
        connect_args = {"ssl": ssl_context}
    elif ssl_mode.lower() == "disable":
        connect_args = {"ssl": False}
    else:
        connect_args = {"ssl": True}

    return (
        sync_url.render_as_string(hide_password=False),
        async_url.render_as_string(hide_password=False),
        connect_args,
    )


raw_database_url: Optional[str] = os.getenv("DATABASE_URL")
if not raw_database_url:
    raise RuntimeError("DATABASE_URL is missing")

SYNC_DATABASE_URL, ASYNC_DATABASE_URL, ASYNC_CONNECT_ARGS = _build_urls(raw_database_url)

POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", 5))
MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", 10))
POOL_TIMEOUT = int(os.getenv("DATABASE_POOL_TIMEOUT", 30))
POOL_RECYCLE = int(os.getenv("DATABASE_POOL_RECYCLE", 1800))
ECHO_SQL = os.getenv("DATABASE_ECHO", "false").lower() == "true"

async_engine: AsyncEngine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_timeout=POOL_TIMEOUT,
    pool_recycle=POOL_RECYCLE,
    pool_pre_ping=True,
    echo=ECHO_SQL,
    connect_args=ASYNC_CONNECT_ARGS,
)

AsyncSessionFactory = _async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Backward compatibility for previous imports
async_sessionmaker = AsyncSessionFactory