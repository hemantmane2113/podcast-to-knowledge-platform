"""Phase C: topic/idea understanding over canonical chunks. Batches
chunks by an estimated token budget (Settings.topic_analysis_token_budget)
so a long-form podcast's full chunk set doesn't exceed the model's usable
context in one call, then reconciles per-batch results if there was more
than one batch. See app/ai/prompts.py's topic_analysis_prompt for why
per-batch chunk numbering doesn't need renumbering before reconciliation.

Reconciliation is boundary-scoped, not a full-list merge: most topics
cannot possibly be a batch-boundary artifact (only a topic whose
chunk_sequence_numbers actually touch the seam between two batches can
be), so only those candidates are sent to the LLM, and only for a
semantic decision (are these the same topic split in two?) -- chunk
numbers, claims, and subtopics for a merged topic are always
reconstructed in Python from the original topics, never taken from the
model's response. See _boundary_candidate_indices/_reconstruct_topics
below and TopicMergeGroup in app/ai/schemas.py.
"""

import logging
import uuid

from app.ai.prompts import topic_analysis_prompt, topic_boundary_merge_prompt
from app.ai.schemas import TopicAnalysisResult, TopicClaim, TopicItem, TopicMergeDecision, TopicMergeGroup
from app.ai.state import ArticlePipelineState, PipelineDeps
from app.models.chunk import Chunk
from app.models.episode import ProcessingStatus
from app.models.topic import Topic
from app.providers.llm.base import LLMMessage
from app.repositories.topic_repository import TopicCandidate
from app.services.chunking_service import estimate_tokens

logger = logging.getLogger(__name__)


def _topics_are_complete(topics: list[Topic]) -> bool:
    """Resumability's stage-completion signal for topic analysis (Batch
    2B): TopicRepository.replace_all is the only writer, called exactly
    once with the FULL final topic list after batching/reconciliation is
    entirely done -- so any row existing at all for this transcript means
    that call (and its commit) already succeeded, i.e. topic analysis
    previously ran to completion. The per-row sanity check below is a
    basic guard against trusting a corrupt/degenerate row as completed
    work, not a separate stage-completion marker -- there is no
    partial-write state to distinguish here, unlike sections (see
    app/ai/nodes/section_generation.py), since topics are only ever
    persisted all at once.
    """
    return bool(topics) and all(t.title.strip() and t.summary.strip() for t in topics)


def _batch_chunks(chunks: list[Chunk], token_budget: int) -> list[list[Chunk]]:
    batches: list[list[Chunk]] = []
    current: list[Chunk] = []
    current_tokens = 0
    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk.text)
        if current and current_tokens + chunk_tokens > token_budget:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(chunk)
        current_tokens += chunk_tokens
    if current:
        batches.append(current)
    return batches


async def _analyze_batch(deps: PipelineDeps, chunks: list[Chunk], start_index: int) -> list[TopicItem]:
    system, user = topic_analysis_prompt(chunks, start_index)
    result = await deps.llm_provider.generate_structured(
        messages=[LLMMessage(role="user", content=user)],
        system=system,
        response_model=TopicAnalysisResult,
    )
    return result.topics


def _boundary_candidate_indices(
    all_topics: list[TopicItem],
    batch_topic_ranges: list[tuple[int, int]],
    batch_chunk_ranges: list[tuple[int, int]],
) -> set[int]:
    """A topic is a boundary candidate only if its chunk_sequence_numbers
    actually touch the seam between two consecutive batches -- the last
    chunk index of the earlier batch, or the first chunk index of the
    later one. A topic that stays fully inside one batch cannot possibly
    be a boundary-split artifact, whatever position it happens to occupy
    in that batch's topic list -- this is why membership is checked
    against real chunk numbers rather than "the last/first topic in the
    list" (which would be fragile if a batch's topics aren't returned in
    strict chunk order).
    """
    candidates: set[int] = set()
    for i in range(len(batch_chunk_ranges) - 1):
        _, left_last_chunk = batch_chunk_ranges[i]
        right_first_chunk, _ = batch_chunk_ranges[i + 1]
        left_start, left_end = batch_topic_ranges[i]
        right_start, right_end = batch_topic_ranges[i + 1]
        for idx in range(left_start, left_end):
            if left_last_chunk in all_topics[idx].chunk_sequence_numbers:
                candidates.add(idx)
        for idx in range(right_start, right_end):
            if right_first_chunk in all_topics[idx].chunk_sequence_numbers:
                candidates.add(idx)
    return candidates


def _dedup_claims(claims: list[TopicClaim]) -> list[TopicClaim]:
    seen: set[tuple[str, str | None, str]] = set()
    deduped: list[TopicClaim] = []
    for claim in claims:
        key = (claim.text, claim.speaker, claim.claim_type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(claim)
    return deduped


def _dedup_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _reconstruct_topics(
    all_topics: list[TopicItem],
    candidate_indices: set[int],
    merge_groups: list[TopicMergeGroup],
) -> list[TopicItem]:
    """Applies only the merge decisions that are safe to trust, and
    otherwise leaves every topic exactly as topic analysis produced it --
    never silently dropping a topic, a chunk reference, or a claim because
    of a malformed/invalid model response (see the per-decision checks
    below, each logged and skipped rather than raised: one bad merge
    suggestion should not fail the whole job when every topic involved is
    still available, untouched, as a fallback).
    """
    used_indices: set[int] = set()
    accepted: list[TopicMergeGroup] = []

    for group in merge_groups:
        indices = group.topic_indices
        if len(indices) < 2:
            logger.warning("Topic merge: ignoring a merge group with fewer than 2 topics: %s", indices)
            continue
        if len(set(indices)) != len(indices):
            logger.warning("Topic merge: ignoring a merge group with a repeated topic index: %s", indices)
            continue
        unknown = [i for i in indices if i not in candidate_indices]
        if unknown:
            logger.warning(
                "Topic merge: ignoring a merge group referencing topic(s) not offered as candidates: %s",
                unknown,
            )
            continue
        already_used = [i for i in indices if i in used_indices]
        if already_used:
            logger.warning(
                "Topic merge: ignoring a merge group reusing already-merged topic(s): %s", already_used
            )
            continue
        used_indices.update(indices)
        accepted.append(group)

    # (anchor_index, topic) pairs so untouched and merged topics can be
    # sorted back into one deterministic, original-order-preserving list.
    ordered: list[tuple[int, TopicItem]] = [
        (i, topic) for i, topic in enumerate(all_topics) if i not in used_indices
    ]
    for group in accepted:
        sources = [all_topics[i] for i in sorted(group.topic_indices)]
        merged = TopicItem(
            title=group.merged_title,
            summary=group.merged_summary,
            chunk_sequence_numbers=sorted({n for t in sources for n in t.chunk_sequence_numbers}),
            key_claims=_dedup_claims([c for t in sources for c in t.key_claims]),
            subtopics=_dedup_strings([s for t in sources for s in t.subtopics]),
        )
        ordered.append((min(group.topic_indices), merged))

    ordered.sort(key=lambda pair: pair[0])
    return [topic for _, topic in ordered]


async def _merge_topics(
    deps: PipelineDeps,
    all_topics: list[TopicItem],
    candidate_indices: set[int],
) -> list[TopicItem]:
    if not candidate_indices:
        # No topic touches a batch boundary at all -- nothing could
        # possibly need merging, so there's no reason to spend a call.
        return all_topics

    candidates = [(i, all_topics[i]) for i in sorted(candidate_indices)]
    system, user = topic_boundary_merge_prompt(candidates)
    decision = await deps.llm_provider.generate_structured(
        messages=[LLMMessage(role="user", content=user)],
        system=system,
        response_model=TopicMergeDecision,
    )
    return _reconstruct_topics(all_topics, candidate_indices, decision.merges)


def build(deps: PipelineDeps):
    async def topic_analysis_node(state: ArticlePipelineState) -> dict:
        episode = await deps.episodes.get_by_id(state["episode_id"])
        if episode is not None:
            deps.episodes.set_status(episode, ProcessingStatus.ANALYZING)
            await deps.session.commit()

        # Resumability (Batch 2B): a prior attempt for this episode/job may
        # already have completed topic analysis before a worker crash or
        # ARQ retry -- reuse that work instead of paying for the batch/merge
        # LLM calls again. See _topics_are_complete's docstring for why
        # "any row exists" is already the right completion signal here.
        existing_topics = await deps.topics.get_by_transcript_id(state["transcript_id"])
        if _topics_are_complete(existing_topics):
            logger.info(
                "Topic analysis: reusing %d already-persisted topic(s) for transcript %s",
                len(existing_topics),
                state["transcript_id"],
            )
            return {"topics": existing_topics}

        chunks = state["chunks"]
        batches = _batch_chunks(chunks, deps.settings.topic_analysis_token_budget)
        logger.info("Topic analysis: %d chunks in %d batch(es)", len(chunks), len(batches))

        all_topics: list[TopicItem] = []
        batch_topic_ranges: list[tuple[int, int]] = []
        batch_chunk_ranges: list[tuple[int, int]] = []
        offset = 0
        for batch in batches:
            batch_topics = await _analyze_batch(deps, batch, offset)
            start = len(all_topics)
            all_topics.extend(batch_topics)
            batch_topic_ranges.append((start, len(all_topics)))
            batch_chunk_ranges.append((offset, offset + len(batch) - 1))
            offset += len(batch)

        if len(batches) > 1:
            candidate_indices = _boundary_candidate_indices(all_topics, batch_topic_ranges, batch_chunk_ranges)
            all_topics = await _merge_topics(deps, all_topics, candidate_indices)

        candidates = []
        for sequence_number, item in enumerate(all_topics):
            chunk_ids: list[uuid.UUID] = []
            for idx in item.chunk_sequence_numbers:
                if 0 <= idx < len(chunks):
                    chunk_ids.append(chunks[idx].id)
                else:
                    logger.warning("Topic analysis referenced out-of-range chunk index %s", idx)
            candidates.append(
                TopicCandidate(
                    sequence_number=sequence_number,
                    title=item.title,
                    summary=item.summary,
                    chunk_ids=chunk_ids,
                    key_claims=[c.model_dump() for c in item.key_claims],
                    subtopics=item.subtopics,
                )
            )

        topics = await deps.topics.replace_all(
            transcript_id=state["transcript_id"], episode_id=state["episode_id"], candidates=candidates
        )
        await deps.session.commit()

        return {"topics": topics}

    return topic_analysis_node
