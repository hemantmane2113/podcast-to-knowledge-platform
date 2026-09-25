"""Phase H: validation. Deterministic checks (app/services/article_validation.py)
are the source of truth for pass/fail -- ValidationReport.passed stays
entirely deterministic, always.

Separately, an optional whole-article editorial review
(Settings.enable_llm_validation, off by default) reads the ASSEMBLED
article to catch what per-section generation structurally cannot see:
cross-section repetition, weak transitions, a soft-target overshoot that
calls for targeted tightening rather than uniform shortening, and so on
(app/ai/prompts.py::article_editorial_review_prompt has the exact
instructions; app/ai/schemas.py::ArticleEditorialReview docstring has the
full design). Its `coherent`/`notes` stay purely advisory, appended to the
persisted `checks` exactly as the prior LLMCoherenceReview was -- never
folded into ValidationReport.passed. Its `sections_needing_revision`/
`overall_feedback` are new and ACTIONABLE: returned separately as
`editorial_review` in pipeline state, read by app/ai/graph.py's
_should_revise and app/ai/nodes/revision.py to route into the SAME
existing per-section revision mechanism a failed deterministic check
already uses -- a second, independent trigger for revision, never a
change to what "passed" means.
"""

import logging

from app.ai.prompts import article_editorial_review_prompt
from app.ai.schemas import ArticleEditorialReview
from app.ai.state import ArticlePipelineState, PipelineDeps
from app.core.exceptions import LLMProviderError
from app.providers.llm.base import LLMMessage
from app.services.article_validation import run_validation

logger = logging.getLogger(__name__)


async def _editorial_review(deps: PipelineDeps, article) -> ArticleEditorialReview | None:
    if not deps.settings.enable_llm_validation:
        return None
    sections = [
        {"sequence_number": s.sequence_number, "heading": s.heading, "content": s.content}
        for s in article.sections
    ]
    article_word_count = sum(len(s.content.split()) for s in article.sections)
    system, user = article_editorial_review_prompt(
        article_title=article.title,
        sections=sections,
        article_word_count=article_word_count,
        target_word_count_min=deps.settings.article_target_word_count_min,
        target_word_count_max=deps.settings.article_target_word_count_max,
    )
    try:
        return await deps.llm_provider.generate_structured(
            messages=[LLMMessage(role="user", content=user)],
            system=system,
            response_model=ArticleEditorialReview,
        )
    except LLMProviderError as exc:
        # Optional in every sense -- never fails the pipeline on its own,
        # and a failed call simply means no editorial-review-driven
        # revision this round, same as if it were disabled.
        logger.warning("Optional article editorial review failed (non-fatal): %s", exc)
        return None


def build(deps: PipelineDeps):
    async def validation_node(state: ArticlePipelineState) -> dict:
        # Article-generation progress lives on ProcessingJob.status now,
        # never on Episode.status (Episode.status Decoupling / Live-Draft
        # Article Workflow) -- no episode fetch/status write needed here.
        article = state["article"]
        report = run_validation(
            sections=article.sections,
            plan=state["plan"],
            chunks=state["chunks"],
            topics=state["topics"],
            transcript_word_count=state["transcript_word_count"],
            max_length_ratio=deps.settings.article_max_length_ratio,
            min_sections=deps.settings.section_count_min,
            max_sections=deps.settings.section_count_max,
            target_word_count_min=deps.settings.article_target_word_count_min,
            target_word_count_max=deps.settings.article_target_word_count_max,
        )

        checks_json = report.to_json()
        editorial_review = await _editorial_review(deps, article)
        if editorial_review is not None:
            checks_json = [
                *checks_json,
                {
                    "name": "article_editorial_review",
                    "passed": editorial_review.coherent,
                    "details": editorial_review.notes,
                },
            ]

        deps.validation_results.create(article_id=article.id, passed=report.passed, checks=checks_json)
        await deps.session.commit()

        return {"validation_report": report, "editorial_review": editorial_review}

    return validation_node
