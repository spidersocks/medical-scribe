# migrations/env.py
import asyncio
import logging
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from database import ASYNC_CONNECT_ARGS, ASYNC_DATABASE_URL  # noqa: E402
from models import Base  # noqa: E402

# --------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

# --------------------------------------------------------------------
# Metadata & Alembic configuration
# --------------------------------------------------------------------
target_metadata = Base.metadata
config.set_main_option("sqlalchemy.url", ASYNC_DATABASE_URL.replace("%", "%%"))

# --------------------------------------------------------------------
# Migration helpers
# --------------------------------------------------------------------
def run_migrations_offline() -> None:
    """Run migrations without a live DB connection."""
    context.configure(
        url=ASYNC_DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations given an open connection."""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Create an async engine that enforces TLS and run migrations."""
    connectable: AsyncEngine = create_async_engine(
        ASYNC_DATABASE_URL,
        poolclass=pool.NullPool,
        connect_args=ASYNC_CONNECT_ARGS,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


# --------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())