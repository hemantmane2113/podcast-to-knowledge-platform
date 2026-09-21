"""Phase F: section-by-section generation -- never one call over the
whole transcript. Each section gets only its own plan instructions and
its own supporting chunks' actual text, plus the fidelity/attribution
constraints in app/ai/prompts.py. generate_section() is also used by
app/ai/nodes/revision.py to regenerate just the sections a failed
validation flagged.
"""

import uuid

from app.ai.prompts import section_generation_prompt
from app.ai.schemas import GeneratedSection
from app.ai.state import ArticlePipelineState, PipelineDeps
from app.models.chunk import Chunk
from app.models.episode import ProcessingStatus
from app.providers.llm.base import LLMMessage
from app.repositories.article_repository import ArticleSectionCandidate


async def generate_section(
    deps: PipelineDeps,
    planned_section: dict,
    chunk_by_id: dict[uuid.UUID, Chunk],
    *,
    revision_feedback: str | None = None,
) -> ArticleSectionCandidate:
    supporting_chunk_ids = [uuid.UUID(cid) for cid in planned_section.get("supporting_chunk_ids", [])]
    supporting_topic_ids = [uuid.UUID(tid) for tid in planned_section.get("supporting_topic_ids", [])]
    supporting_chunks = [chunk_by_id[cid] for cid in supporting_chunk_ids if cid in chunk_by_id]

    system, user = section_generation_prompt(
        heading=planned_section["heading"],
        key_ideas=planned_section.get("key_ideas", []),
        viewpoints=planned_section.get("viewpoints", []),
        attribution_notes=planned_section.get("attribution_notes", []),
        supporting_chunks=supporting_chunks,
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

        candidates = []
        for planned_section in plan.sections:
            candidates.append(await generate_section(deps, planned_section, chunk_by_id))

        article = await deps.articles.replace(
            episode_id=state["episode_id"],
            article_plan_id=plan.id,
            title=plan.title,
            revision_count=state.get("revision_count", 0),
            sections=candidates,
        )
        await deps.session.commit()

        return {"article": article}

    return section_generation_node
