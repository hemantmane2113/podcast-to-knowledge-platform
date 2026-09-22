"""Phase E: article planning. One LLM call over the (compact) topic list
-- never the full chunk text, so this stays well within context even for
a very long transcript."""

from app.ai.prompts import planning_prompt
from app.ai.schemas import ArticlePlanResult
from app.ai.state import ArticlePipelineState, PipelineDeps
from app.models.episode import ProcessingStatus
from app.providers.llm.base import LLMMessage
from app.repositories.article_plan_repository import PlannedSectionData


def build(deps: PipelineDeps):
    async def planning_node(state: ArticlePipelineState) -> dict:
        episode = await deps.episodes.get_by_id(state["episode_id"])
        if episode is not None:
            deps.episodes.set_status(episode, ProcessingStatus.PLANNING)
            await deps.session.commit()

        topics = state["topics"]
        topics_by_sequence = {t.sequence_number: t for t in topics}

        system, user = planning_prompt(
            topics,
            target_section_count_min=deps.settings.section_count_target_min,
            target_section_count_max=deps.settings.section_count_target_max,
        )
        result: ArticlePlanResult = await deps.llm_provider.generate_structured(
            messages=[LLMMessage(role="user", content=user)],
            system=system,
            response_model=ArticlePlanResult,
        )

        planned_sections: list[PlannedSectionData] = []
        for sequence_number, section in enumerate(result.sections):
            supporting_topics = [
                topics_by_sequence[n] for n in section.supporting_topic_sequence_numbers if n in topics_by_sequence
            ]
            supporting_chunk_ids: list[str] = []
            seen_chunk_ids: set[str] = set()
            for topic in supporting_topics:
                for chunk_id in topic.chunk_ids:
                    key = str(chunk_id)
                    if key not in seen_chunk_ids:
                        seen_chunk_ids.add(key)
                        supporting_chunk_ids.append(key)

            planned_sections.append(
                PlannedSectionData(
                    sequence_number=sequence_number,
                    heading=section.heading,
                    key_ideas=section.key_ideas,
                    supporting_topic_ids=[str(t.id) for t in supporting_topics],
                    supporting_chunk_ids=supporting_chunk_ids,
                    viewpoints=section.viewpoints,
                    attribution_notes=section.attribution_notes,
                )
            )

        plan = await deps.article_plans.replace(
            episode_id=state["episode_id"],
            title=result.title,
            introduction_summary=result.introduction_summary,
            conclusion_summary=result.conclusion_summary,
            sections=planned_sections,
        )
        await deps.session.commit()

        return {"plan": plan}

    return planning_node
