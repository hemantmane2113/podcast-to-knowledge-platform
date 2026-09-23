"""Bounded revision loop: on a failed validation, regenerate only the
sections a failing check could be tied to (parsed from the check's own
`details` text, e.g. "section 2: ..."), with that feedback appended to
the regeneration prompt. A check that isn't tied to any specific section
(e.g. overall article_length) becomes shared feedback applied to every
regenerated section. If no section can be identified at all, every
section is regenerated -- simpler and safer than guessing.

This is deliberately NOT a redesign of section_generation -- it reuses
generate_section() so both the first pass and every revision share the
same prompt/fidelity constraints.
"""

import re

from app.ai.nodes.section_generation import (
    collect_preceding_key_ideas,
    excerpt_preceding_section,
    generate_section,
    next_section_narrative_purpose,
    order_plan_sections,
    section_word_target,
)
from app.ai.state import ArticlePipelineState, PipelineDeps
from app.models.episode import ProcessingStatus
from app.repositories.article_repository import ArticleSectionCandidate

_SECTION_REF_RE = re.compile(r"section (\d+)")


def _sections_needing_revision(checks: list[dict]) -> tuple[set[int], list[str]]:
    """Returns (section sequence_numbers to regenerate, shared feedback
    not tied to a specific section)."""
    targeted: set[int] = set()
    shared: list[str] = []
    for check in checks:
        if check.get("passed"):
            continue
        details = check.get("details", "")
        matches = {int(n) for n in _SECTION_REF_RE.findall(details)}
        if matches:
            targeted.update(matches)
        else:
            shared.append(f"{check.get('name')}: {details}")
    return targeted, shared


def build(deps: PipelineDeps):
    async def revision_node(state: ArticlePipelineState) -> dict:
        episode = await deps.episodes.get_by_id(state["episode_id"])
        if episode is not None:
            deps.episodes.set_status(episode, ProcessingStatus.REVISING)
            await deps.session.commit()

        report = state["validation_report"]
        checks = report.to_json()
        targeted, shared_feedback = _sections_needing_revision(checks)

        plan = state["plan"]
        chunk_by_id = {c.id: c for c in state["chunks"]}
        topic_by_id = {t.id: t for t in state["topics"]}
        ordered_sections = order_plan_sections(plan.sections)
        section_headings = [s["heading"] for s in ordered_sections]
        word_target_min, word_target_max = section_word_target(deps.settings, len(ordered_sections))
        current_sections_by_sequence = {s.sequence_number: s for s in state["article"].sections}

        # No specific section identified anywhere -> regenerate everything
        # rather than guessing; otherwise only the flagged sections (plus
        # any check with no section-specific feedback applied to all of them).
        regenerate_all = not targeted
        shared_feedback_text = "; ".join(shared_feedback) if shared_feedback else None

        # candidates is built in the same order as ordered_sections, so
        # candidates[-1] is always the immediately preceding section's
        # CURRENT content for this revision pass -- whether that section
        # was itself just regenerated above or kept as-is, which
        # current_sections_by_sequence alone (a snapshot from BEFORE this
        # pass started) can't reflect for a section revised earlier in the
        # same pass.
        candidates: list[ArticleSectionCandidate] = []
        for planned_section in ordered_sections:
            seq = planned_section["sequence_number"]
            preceding_content = candidates[-1].content if candidates else None
            if regenerate_all or seq in targeted:
                feedback_parts = [f for f in [shared_feedback_text] if f]
                if seq in targeted:
                    section_specific = [c.get("details", "") for c in checks if not c.get("passed") and f"section {seq}" in c.get("details", "")]
                    feedback_parts.extend(section_specific)
                candidates.append(
                    await generate_section(
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
                        revision_feedback="; ".join(feedback_parts) or None,
                    )
                )
            else:
                existing = current_sections_by_sequence.get(seq)
                if existing is not None:
                    candidates.append(
                        ArticleSectionCandidate(
                            sequence_number=existing.sequence_number,
                            heading=existing.heading,
                            content=existing.content,
                            supporting_chunk_ids=list(existing.supporting_chunk_ids),
                            supporting_topic_ids=list(existing.supporting_topic_ids),
                        )
                    )

        revision_count = state.get("revision_count", 0) + 1
        article = await deps.articles.replace(
            episode_id=state["episode_id"],
            article_plan_id=plan.id,
            title=plan.title,
            revision_count=revision_count,
            sections=candidates,
        )
        await deps.session.commit()

        return {"article": article, "revision_count": revision_count}

    return revision_node
