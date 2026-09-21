"""Phase C: topic/idea understanding over canonical chunks. Batches
chunks by an estimated token budget (Settings.topic_analysis_token_budget)
so a long-form podcast's full chunk set doesn't exceed the model's usable
context in one call, then merges per-batch results with one additional
LLM call if there was more than one batch. See app/ai/prompts.py's
topic_analysis_prompt for why per-batch chunk numbering doesn't need
renumbering before the merge.
"""

import json
import logging
import uuid

from app.ai.prompts import topic_analysis_prompt, topic_merge_prompt
from app.ai.schemas import TopicAnalysisResult, TopicItem
from app.ai.state import ArticlePipelineState, PipelineDeps
from app.models.chunk import Chunk
from app.models.episode import ProcessingStatus
from app.providers.llm.base import LLMMessage
from app.repositories.topic_repository import TopicCandidate
from app.services.chunking_service import estimate_tokens

logger = logging.getLogger(__name__)


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


async def _merge_topics(deps: PipelineDeps, topics: list[TopicItem]) -> list[TopicItem]:
    payload = json.dumps([t.model_dump() for t in topics])
    system, user = topic_merge_prompt(payload)
    result = await deps.llm_provider.generate_structured(
        messages=[LLMMessage(role="user", content=user)],
        system=system,
        response_model=TopicAnalysisResult,
    )
    return result.topics


def build(deps: PipelineDeps):
    async def topic_analysis_node(state: ArticlePipelineState) -> dict:
        episode = await deps.episodes.get_by_id(state["episode_id"])
        if episode is not None:
            deps.episodes.set_status(episode, ProcessingStatus.ANALYZING)
            await deps.session.commit()

        chunks = state["chunks"]
        batches = _batch_chunks(chunks, deps.settings.topic_analysis_token_budget)
        logger.info("Topic analysis: %d chunks in %d batch(es)", len(chunks), len(batches))

        all_topics: list[TopicItem] = []
        offset = 0
        for batch in batches:
            all_topics.extend(await _analyze_batch(deps, batch, offset))
            offset += len(batch)

        if len(batches) > 1:
            all_topics = await _merge_topics(deps, all_topics)

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
