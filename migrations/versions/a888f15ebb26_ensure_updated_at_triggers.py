"""ensure updated_at triggers

Revision ID: a888f15ebb26
Revises: a8cf9d2e33bd
Create Date: 2025-10-14 10:08:56.238539

"""
from typing import Sequence, Union

from alembic import op

revision: str = "a888f15ebb26"
down_revision: Union[str, Sequence[str], None] = "a8cf9d2e33bd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLES_WITH_UPDATED_AT = ("users", "patients", "consultations")


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    for table_name in TABLES_WITH_UPDATED_AT:
        op.execute(
            f"""
            CREATE TRIGGER {table_name}_set_updated_at
            BEFORE UPDATE ON {table_name}
            FOR EACH ROW
            EXECUTE FUNCTION set_updated_at();
            """
        )


def downgrade() -> None:
    for table_name in TABLES_WITH_UPDATED_AT:
        op.execute(f"DROP TRIGGER IF EXISTS {table_name}_set_updated_at ON {table_name};")
    op.execute("DROP FUNCTION IF EXISTS set_updated_at();")