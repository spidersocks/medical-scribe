"""initial schema

Revision ID: a8cf9d2e33bd
Revises:
Create Date: 2025-10-14 09:02:03.252764
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a8cf9d2e33bd"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

sex_enum = postgresql.ENUM(
    "male",
    "female",
    "other",
    "unknown",
    name="sex_enum",
    create_type=False,
)


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
    conn.execute(sa.text("DROP TYPE IF EXISTS sex_enum CASCADE"))
    sex_enum.create(conn, checkfirst=True)

    op.create_table(
        "users",
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    op.create_table(
        "patients",
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.user_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("date_of_birth", sa.Date(), nullable=True),
        sa.Column("sex", sex_enum, nullable=True),
        sa.Column("hkid_number", sa.Text(), nullable=True),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("phone", sa.String(50), nullable=True),
        sa.Column("referring_physician", sa.String(255), nullable=True),
        sa.Column("additional_context", sa.Text(), nullable=True),
        sa.Column(
            "is_starred",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.UniqueConstraint("user_id", "email", name="patients_email_unique_per_user"),
    )
    op.create_index("ix_patients_user_id", "patients", ["user_id"])

    op.create_table(
        "consultations",
        sa.Column(
            "consultation_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.patient_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.user_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255)),
        sa.Column("language_code", sa.String(20)),
        sa.Column("note_type", sa.String(50)),
        sa.Column("session_state", sa.String(50)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index("ix_consultations_patient_id", "consultations", ["patient_id"])
    op.create_index("ix_consultations_user_id", "consultations", ["user_id"])

    op.create_table(
        "transcript_segments",
        sa.Column(
            "segment_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "consultation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("consultations.consultation_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("sequence_number", sa.Integer(), nullable=False),
        sa.Column("speaker_label", sa.String(50)),
        sa.Column("speaker_role", sa.String(50)),
        sa.Column("original_text", sa.Text(), nullable=False),
        sa.Column("translated_text", sa.Text()),
        sa.Column("detected_language", sa.String(20)),
        sa.Column("start_time_ms", sa.Integer()),
        sa.Column("end_time_ms", sa.Integer()),
        sa.Column("entities", postgresql.JSONB()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "uq_segments_consultation_sequence",
        "transcript_segments",
        ["consultation_id", "sequence_number"],
        unique=True,
    )
    op.create_index(
        "ix_segments_consultation",
        "transcript_segments",
        ["consultation_id"],
    )

    op.create_table(
        "clinical_notes",
        sa.Column(
            "note_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "consultation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("consultations.consultation_id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("note_type", sa.String(50), nullable=False),
        sa.Column("content", postgresql.JSONB(), nullable=False),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("last_edited_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    conn = op.get_bind()

    op.drop_table("clinical_notes")
    op.drop_index("ix_segments_consultation", table_name="transcript_segments")
    op.drop_index("uq_segments_consultation_sequence", table_name="transcript_segments")
    op.drop_table("transcript_segments")
    op.drop_index("ix_consultations_user_id", table_name="consultations")
    op.drop_index("ix_consultations_patient_id", table_name="consultations")
    op.drop_table("consultations")
    op.drop_index("ix_patients_user_id", table_name="patients")
    op.drop_table("patients")
    op.drop_table("users")

    sex_enum.drop(conn, checkfirst=True)
    conn.execute(sa.text("DROP EXTENSION IF EXISTS pgcrypto"))