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

from app.ai.prompts import EpisodeContext, section_generation_prompt
from app.ai.schemas import GeneratedSection
from app.ai.state import ArticlePipelineState, PipelineDeps
from app.config.settings import Settings
from app.models.chunk import Chunk
from app.models.episode import ProcessingStatus
from app.models.topic import Topic
from app.providers.llm.base import LLMMessage
from app.repositories.article_repository import ArticleSectionCandidate
from app.services.article_validation import is_section_content_valid

logger = logging.getLogger(__name__)

_PRECEDING_EXCERPT_MAX_CHARS = 500


def section_headings_by_sequence(plan_sections: list[dict]) -> list[str]:
    """The whole article's ordered section headings -- used both to build
    each section-generation call's compact outline and to identify which
    position a given section occupies in it. Sorted by sequence_number
    rather than trusting storage order, since that's the one thing that
    actually defines "ordered" here (see ArticlePlanRepository.replace).
    """
    ordered = sorted(plan_sections, key=lambda s: s["sequence_number"])
    return [s["heading"] for s in ordered]


def order_plan_sections(plan_sections: list[dict]) -> list[dict]:
    """Same ordering guarantee as section_headings_by_sequence, but
    returning the full section dicts -- shared by section_generation_node
    and revision_node so both build "already covered by earlier sections"
    context (see collect_preceding_key_ideas) against the same, consistent
    ordering."""
    return sorted(plan_sections, key=lambda s: s["sequence_number"])


def collect_preceding_key_ideas(ordered_sections: list[dict], sequence_number: int) -> list[tuple[str, list[str]]]:
    """Layer 1 of "already covered" context (Batch 3 editorial rework):
    every earlier section's own PLANNED key_ideas -- cheap (already in
    memory, no extra DB read) and available even for the very first
    section of a resumed/revised run. Deliberately not assumed to be a
    perfect record of what was actually written (planned intent can drift
    from generated prose) -- see excerpt_preceding_section below for the
    complementary, actually-generated-content layer. Shared by
    section_generation_node and revision_node (app/ai/nodes/revision.py)."""
    return [
        (s["heading"], s.get("key_ideas", []))
        for s in ordered_sections
        if s["sequence_number"] < sequence_number
    ]


def excerpt_preceding_section(content: str) -> str:
    """Layer 2 of "already covered" context: a short, PURELY DETERMINISTIC
    (never an extra LLM call) tail of the immediately preceding section's
    actual generated content -- how it ended, so the next section can pick
    up the thread and avoid an abrupt jump. Takes the last paragraph (most
    relevant for a transition), normalizes whitespace, and hard-truncates
    to a small char budget -- never the immediately preceding section's
    full prose, and never any section other than the immediately preceding
    one (see section_generation_prompt's docstring)."""
    paragraphs = [p for p in content.strip().split("\n\n") if p.strip()]
    tail = paragraphs[-1] if paragraphs else content.strip()
    tail = " ".join(tail.split())
    if len(tail) <= _PRECEDING_EXCERPT_MAX_CHARS:
        return tail
    return "..." + tail[-_PRECEDING_EXCERPT_MAX_CHARS:]


def next_section_narrative_purpose(ordered_sections: list[dict], sequence_number: int) -> str | None:
    """"Where appropriate, the direction of the next section" (Batch 4
    editorial rework): a single short forward-looking signal -- the
    immediately FOLLOWING section's own planned narrative_purpose. Never
    its heading (already visible in ARTICLE STRUCTURE) and never its
    transition_from_previous (that's written from the NEXT section's own
    perspective looking back at this one, which would be circular here).
    None when there is no next section, or the plan has no purpose
    recorded for it (an older/incomplete plan) -- "where appropriate" is
    handled by simply omitting the line, not by inventing one."""
    for s in ordered_sections:
        if s["sequence_number"] == sequence_number + 1:
            return s.get("narrative_purpose") or None
    return None


def section_word_target(settings: Settings, section_count: int) -> tuple[int | None, int | None]:
    """A rough, soft per-section word-count guideline derived from
    Settings.article_target_word_count_min/max -- an even split across the
    planned section count, presented to the model as approximate guidance
    (see section_generation_prompt), never a hard constraint. None/None
    when there's nothing to divide across (an empty plan)."""
    if section_count <= 0:
        return None, None
    return (
        settings.article_target_word_count_min // section_count,
        settings.article_target_word_count_max // section_count,
    )


async def generate_section(
    deps: PipelineDeps,
    planned_section: dict,
    chunk_by_id: dict[uuid.UUID, Chunk],
    topic_by_id: dict[uuid.UUID, Topic],
    *,
    article_title: str,
    section_headings: list[str],
    introduction_summary: str = "",
    conclusion_summary: str = "",
    preceding_sections_key_ideas: list[tuple[str, list[str]]] | None = None,
    preceding_section_excerpt: str | None = None,
    next_narrative_purpose: str | None = None,
    target_word_count_min: int | None = None,
    target_word_count_max: int | None = None,
    revision_feedback: str | None = None,
    episode_context: EpisodeContext | None = None,
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
        # .get(..., "") -- an ArticlePlan persisted before these two
        # fields existed (or a test fixture that omits them) has no such
        # keys; treated the same as "the planner didn't provide one"
        # rather than raising.
        narrative_purpose=planned_section.get("narrative_purpose", ""),
        transition_from_previous=planned_section.get("transition_from_previous", ""),
        introduction_summary=introduction_summary,
        conclusion_summary=conclusion_summary,
        preceding_sections_key_ideas=preceding_sections_key_ideas,
        preceding_section_excerpt=preceding_section_excerpt,
        next_narrative_purpose=next_narrative_purpose,
        target_word_count_min=target_word_count_min,
        target_word_count_max=target_word_count_max,
        revision_feedback=revision_feedback,
        episode_context=episode_context,
    )
    result: GeneratedSection = await deps.llm_provider.generate_structured(
        messages=[LLMMessage(role="user", content=user)],
        system=system,
        response_model=GeneratedSection,
    )

    return ArticleSectionCandidate(
        sequence_number=planned_section["sequence_number"],
        heading=result.heading,
        content="\n\n".join(result.paragraphs),
        supporting_chunk_ids=supporting_chunk_ids,
        supporting_topic_ids=supporting_topic_ids,
    )


def build(deps: PipelineDeps):
    async def section_generation_node(state: ArticlePipelineState) -> dict:
        episode = await deps.episodes.get_by_id(state["episode_id"])
        if episode is not None:
            deps.episodes.set_status(episode, ProcessingStatus.GENERATING)
            await deps.session.commit()

        episode_context = (
            EpisodeContext(title=episode.title, channel_name=episode.channel_name)
            if episode is not None
            else None
        )

        plan = state["plan"]
        chunk_by_id = {c.id: c for c in state["chunks"]}
        topic_by_id = {t.id: t for t in state["topics"]}
        ordered_sections = order_plan_sections(plan.sections)
        section_headings = [s["heading"] for s in ordered_sections]
        word_target_min, word_target_max = section_word_target(deps.settings, len(ordered_sections))

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
        # Tracks each section's ACTUAL generated content (reused or freshly
        # generated) as the loop proceeds, keyed by sequence_number -- the
        # source for the immediately-preceding-section excerpt (layer 2 of
        # "already covered" context). Built incrementally rather than read
        # once from existing_by_sequence, since a section generated earlier
        # in THIS SAME pass isn't in existing_by_sequence at all yet.
        generated_content_by_sequence: dict[int, str] = {}

        for planned_section in ordered_sections:
            seq = planned_section["sequence_number"]
            existing = existing_by_sequence.get(seq)
            if existing is not None and is_section_content_valid(existing.heading, existing.content):
                # Already generated (and valid) in a prior attempt -- no
                # LLM call, no write.
                logger.info("Section generation: reusing already-persisted section %d", seq)
                generated_content_by_sequence[seq] = existing.content
                continue

            preceding_content = generated_content_by_sequence.get(seq - 1)
            candidate = await generate_section(
                deps,
                planned_section,
                chunk_by_id,
                topic_by_id,
                article_title=plan.title,
                section_headings=section_headings,
                introduction_summary=plan.introduction_summary,
                conclusion_summary=plan.conclusion_summary,
                preceding_sections_key_ideas=collect_preceding_key_ideas(ordered_sections, seq),
                preceding_section_excerpt=(
                    excerpt_preceding_section(preceding_content) if preceding_content else None
                ),
                next_narrative_purpose=next_section_narrative_purpose(ordered_sections, seq),
                target_word_count_min=word_target_min,
                target_word_count_max=word_target_max,
                episode_context=episode_context,
            )
            # Persisted (and committed) immediately after this section's
            # own LLM call succeeds -- a failed/raising call never reaches
            # this line, so a failed section is never persisted as if it
            # were complete (see upsert_section's docstring for the
            # invalid-existing-row replace-in-place case).
            deps.articles.upsert_section(article, candidate)
            await deps.session.commit()
            generated_content_by_sequence[seq] = candidate.content

        return {"article": article}

    return section_generation_node
