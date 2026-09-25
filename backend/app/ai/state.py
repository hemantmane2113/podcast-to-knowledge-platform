"""Shared state passed between LangGraph nodes, and the dependency bundle
each node closes over. Deliberately NOT a fully-serializable/checkpointed
LangGraph state (no checkpointer is configured in graph.py) -- this is one
arq job's in-process run, and Postgres (via each node's own repository
writes) is already the durability layer; a crash mid-run is handled by
arq's existing retry policy (app/worker/tasks.py), the same as every other
job in this codebase, not by LangGraph's own persistence.
"""

import uuid
from dataclasses import dataclass, field
from typing import TypedDict

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.schemas import ArticleEditorialReview
from app.config.settings import Settings
from app.models.article import Article
from app.models.article_plan import ArticlePlan
from app.models.chunk import Chunk
from app.models.topic import Topic
from app.providers.llm.base import LLMProvider
from app.repositories.article_plan_repository import ArticlePlanRepository
from app.repositories.article_repository import ArticleRepository
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.topic_repository import TopicRepository
from app.repositories.validation_result_repository import ValidationResultRepository
from app.services.article_validation import ValidationReport


@dataclass
class PipelineDeps:
    """Built once per ARTICLE_GENERATION job run (app/worker/tasks.py) and
    threaded through every node -- the only place a node reaches the DB or
    the LLM provider."""

    session: AsyncSession
    llm_provider: LLMProvider
    settings: Settings

    def __post_init__(self) -> None:
        self.episodes = EpisodeRepository(self.session)
        self.topics = TopicRepository(self.session)
        self.article_plans = ArticlePlanRepository(self.session)
        self.articles = ArticleRepository(self.session)
        self.validation_results = ValidationResultRepository(self.session)


class ArticlePipelineState(TypedDict, total=False):
    episode_id: uuid.UUID
    transcript_id: uuid.UUID
    chunks: list[Chunk]
    transcript_word_count: int
    # Live/Draft Article Workflow: whether this pipeline run is writing
    # into the episode's DRAFT ArticlePlan/Article (True) or its LIVE one
    # (False) -- threaded into every ArticlePlanRepository/ArticleRepository
    # call the nodes make (app/ai/nodes/planning.py, section_generation.py,
    # revision.py), so a draft run can never accidentally retrieve or
    # overwrite the live row. app/worker/tasks.py::generate_article always
    # sets this to True -- every run reached through the regenerate/generate
    # API always targets a draft; only ArticleService.promote_draft ever
    # makes something live.
    is_draft: bool

    topics: list[Topic]
    plan: ArticlePlan
    article: Article
    validation_report: ValidationReport
    # None when Settings.enable_llm_validation is off (the default), or the
    # optional review call failed -- see app/ai/nodes/validation.py. Its
    # sections_needing_revision/overall_feedback are a second, independent
    # trigger _should_revise (app/ai/graph.py) checks alongside
    # validation_report.passed; its coherent/notes stay advisory-only.
    editorial_review: ArticleEditorialReview | None

    revision_count: int
    max_revision_attempts: int
