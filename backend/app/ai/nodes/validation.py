"""Phase H: validation. Deterministic checks (app/services/article_validation.py)
are the source of truth for pass/fail; an optional LLM coherence review
(Settings.enable_llm_validation, off by default) is appended to the
persisted `checks` list purely as supplementary information and never
affects ValidationReport.passed -- "never a replacement for the
deterministic checks."""

import logging

from app.ai.prompts import llm_coherence_review_prompt
from app.ai.schemas import LLMCoherenceReview
from app.ai.state import ArticlePipelineState, PipelineDeps
from app.core.exceptions import LLMProviderError
from app.models.episode import ProcessingStatus
from app.providers.llm.base import LLMMessage
from app.services.article_validation import run_validation

logger = logging.getLogger(__name__)


async def _optional_llm_review(deps: PipelineDeps, article) -> dict | None:
    if not deps.settings.enable_llm_validation:
        return None
    article_text = "\n\n".join(f"## {s.heading}\n\n{s.content}" for s in article.sections)
    system, user = llm_coherence_review_prompt(article_text)
    try:
        review: LLMCoherenceReview = await deps.llm_provider.generate_structured(
            messages=[LLMMessage(role="user", content=user)],
            system=system,
            response_model=LLMCoherenceReview,
        )
    except LLMProviderError as exc:
        # Supplementary only -- never fails the pipeline on its own.
        logger.warning("Optional LLM coherence review failed (non-fatal): %s", exc)
        return None
    return {"name": "llm_coherence_review", "passed": review.coherent, "details": review.notes}


def build(deps: PipelineDeps):
    async def validation_node(state: ArticlePipelineState) -> dict:
        episode = await deps.episodes.get_by_id(state["episode_id"])
        if episode is not None:
            deps.episodes.set_status(episode, ProcessingStatus.VERIFYING)
            await deps.session.commit()

        article = state["article"]
        report = run_validation(
            sections=article.sections,
            plan=state["plan"],
            chunks=state["chunks"],
            topics=state["topics"],
            transcript_word_count=state["transcript_word_count"],
            max_length_ratio=deps.settings.article_max_length_ratio,
        )

        checks_json = report.to_json()
        llm_check = await _optional_llm_review(deps, article)
        if llm_check is not None:
            checks_json = [*checks_json, llm_check]

        deps.validation_results.create(article_id=article.id, passed=report.passed, checks=checks_json)
        await deps.session.commit()

        return {"validation_report": report}

    return validation_node
