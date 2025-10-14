import asyncio
import os

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing")

async def main():
    async_url = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(
        async_url,
        pool_size=int(os.getenv("DATABASE_POOL_SIZE", 5)),
        max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", 10)),
        echo=True,
    )
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))
        print("DB says:", result.scalar_one())
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())