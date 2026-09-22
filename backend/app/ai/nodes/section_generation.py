"""Phase F: section-by-section generation -- never one call over the
whole transcript. Each section gets its own plan instructions, its own
supporting chunks' actual text, the relevant slice of topic analysis's
knowledge layer for the topic(s) it draws on, and a compact whole-article
outline for structural awareness -- plus the fidelity/attribution
constraints in app/ai/prompts.py. generate_section() is also used by
app/ai/nodes/revision.py to regenerate just the sections a failed
validation flagged.

Resumability (Batch 2B): each section is persisted (and committed)
immediately after its own LLM call succeeds, and a section already
persisted from a prior attempt is reused (no LLM call at all) rather than
regenerated -- see section_generation_node below. A worker crash after
section N therefore leaves sections 1..N durably available to the next
attempt; only the remaining, still-missing sections are ever regenerated.
"""

import logging
import uuid

from app.ai.prompts import section_generation_prompt
from app.ai.schemas import GeneratedSection
from app.ai.state import ArticlePipelineState, PipelineDeps
from app.models.chunk import Chunk
from app.models.episode import ProcessingStatus
from app.models.topic import Topic
from app.providers.llm.base import LLMMessage
from app.repositories.article_repository import ArticleSectionCandidate
from app.services.article_validation import is_section_content_valid

logger = logging.getLogger(__name__)


def section_headings_by_sequence(plan_sections: list[dict]) -> list[str]:
    """The whole article's ordered section headings -- used both to build
    each section-generation call's compact outline and to identify which
    position a given section occupies in it. Sorted by sequence_number
    rather than trusting storage order, since that's the one thing that
    actually defines "ordered" here (see ArticlePlanRepository.replace).
    """
    ordered = sorted(plan_sections, key=lambda s: s["sequence_number"])
    return [s["heading"] for s in ordered]


async def generate_section(
    deps: PipelineDeps,
    planned_section: dict,
    chunk_by_id: dict[uuid.UUID, Chunk],
    topic_by_id: dict[uuid.UUID, Topic],
    *,
    article_title: str,
    section_headings: list[str],
    revision_feedback: str | None = None,
) -> ArticleSectionCandidate:
    supporting_chunk_ids = [uuid.UUID(cid) for cid in planned_section.get("supporting_chunk_ids", [])]
    supporting_topic_ids = [uuid.UUID(tid) for tid in planned_section.get("supporting_topic_ids", [])]
    supporting_chunks = [chunk_by_id[cid] for cid in supporting_chunk_ids if cid in chunk_by_id]
    # Only THIS section's own topics -- never the full knowledge layer
    # (see section_generation_prompt's docstring).
    relevant_topics = [topic_by_id[tid] for tid in supporting_topic_ids if tid in topic_by_id]

    system, user = section_generation_prompt(
        heading=planned_section["heading"],
        key_ideas=planned_section.get("key_ideas", []),
        viewpoints=planned_section.get("viewpoints", []),
        attribution_notes=planned_section.get("attribution_notes", []),
        supporting_chunks=supporting_chunks,
        relevant_topics=relevant_topics,
        article_title=article_title,
        section_headings=section_headings,
        current_section_number=planned_section["sequence_number"],
        revision_feedback=revision_feedback,
    )
    result: GeneratedSection = await deps.llm_provider.generate_structured(
        messages=[LLMMessage(role="user", content=user)],
        system=system,
        response_model=GeneratedSection,
    )

    return ArticleSectionCandidate(
        sequence_number=planned_section["sequence_number"],
        heading=result.heading,
        content=result.content,
        supporting_chunk_ids=supporting_chunk_ids,
        supporting_topic_ids=supporting_topic_ids,
    )


def build(deps: PipelineDeps):
    async def section_generation_node(state: ArticlePipelineState) -> dict:
        episode = await deps.episodes.get_by_id(state["episode_id"])
        if episode is not None:
            deps.episodes.set_status(episode, ProcessingStatus.GENERATING)
            await deps.session.commit()

        plan = state["plan"]
        chunk_by_id = {c.id: c for c in state["chunks"]}
        topic_by_id = {t.id: t for t in state["topics"]}
        section_headings = section_headings_by_sequence(plan.sections)

        # get_or_create reuses the existing Article (sections and all) if
        # it already belongs to this plan -- see its docstring for the
        # identity guard against ever reusing one that doesn't.
        article = await deps.articles.get_or_create(
            episode_id=state["episode_id"],
            article_plan_id=plan.id,
            title=plan.title,
            revision_count=state.get("revision_count", 0),
        )
        await deps.session.commit()

        existing_by_sequence = {s.sequence_number: s for s in article.sections}

        for planned_section in plan.sections:
            seq = planned_section["sequence_number"]
            existing = existing_by_sequence.get(seq)
            if existing is not None and is_section_content_valid(existing.heading, existing.content):
                # Already generated (and valid) in a prior attempt -- no
                # LLM call, no write.
                logger.info("Section generation: reusing already-persisted section %d", seq)
                continue

            candidate = await generate_section(
                deps,
                planned_section,
                chunk_by_id,
                topic_by_id,
                article_title=plan.title,
                section_headings=section_headings,
            )
            # Persisted (and committed) immediately after this section's
            # own LLM call succeeds -- a failed/raising call never reaches
            # this line, so a failed section is never persisted as if it
            # were complete (see upsert_section's docstring for the
            # invalid-existing-row replace-in-place case).
            deps.articles.upsert_section(article, candidate)
            await deps.session.commit()

        return {"article": article}

    return section_generation_node
