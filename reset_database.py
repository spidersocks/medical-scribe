import asyncio
from sqlalchemy import text

from database import engine

async def reset():
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))

if __name__ == "__main__":
    asyncio.run(reset())