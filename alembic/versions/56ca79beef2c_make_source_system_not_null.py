"""make source_system not null

Revision ID: 56ca79beef2c
Revises: 134df7431e02
Create Date: 2026-02-18 14:06:08.051529

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '56ca79beef2c'
down_revision: Union[str, Sequence[str], None] = '134df7431e02'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
