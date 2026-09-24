"""add is_draft to articles and article_plans

Live/Draft Article Workflow: replaces the plain UNIQUE(episode_id) on
both tables with a partial unique index that only applies to
is_draft=false rows, so an episode can have at most one LIVE row and at
most one DRAFT row per table at once, instead of exactly one row total.

server_default=false backfills every existing row (including any
currently-published article) as is_draft=false -- i.e. LIVE -- with no
data movement and no application code involved; existing published
articles are unaffected by this migration.

Revision ID: 1bbe09cd18c7
Revises: 1eaf48a74266
Create Date: 2026-09-24 17:54:15.446775

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1bbe09cd18c7'
down_revision: Union[str, Sequence[str], None] = '1eaf48a74266'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('article_plans', sa.Column('is_draft', sa.Boolean(), server_default=sa.text('false'), nullable=False))
    op.drop_constraint(op.f('article_plans_episode_id_key'), 'article_plans', type_='unique')
    op.create_index('uq_article_plans_episode_id_live', 'article_plans', ['episode_id'], unique=True, postgresql_where=sa.text('NOT is_draft'))
    op.add_column('articles', sa.Column('is_draft', sa.Boolean(), server_default=sa.text('false'), nullable=False))
    op.drop_constraint(op.f('articles_episode_id_key'), 'articles', type_='unique')
    op.create_index('uq_articles_episode_id_live', 'articles', ['episode_id'], unique=True, postgresql_where=sa.text('NOT is_draft'))


def downgrade() -> None:
    """Downgrade schema.

    Only safe if no draft rows exist at rollback time -- restoring the
    plain UNIQUE(episode_id) constraint fails if any episode currently has
    both a live and a draft row (two rows sharing one episode_id, which
    the partial index allows but a plain unique constraint does not).
    """
    op.drop_index('uq_articles_episode_id_live', table_name='articles', postgresql_where=sa.text('NOT is_draft'))
    op.create_unique_constraint(op.f('articles_episode_id_key'), 'articles', ['episode_id'])
    op.drop_column('articles', 'is_draft')
    op.drop_index('uq_article_plans_episode_id_live', table_name='article_plans', postgresql_where=sa.text('NOT is_draft'))
    op.create_unique_constraint(op.f('article_plans_episode_id_key'), 'article_plans', ['episode_id'])
    op.drop_column('article_plans', 'is_draft')
