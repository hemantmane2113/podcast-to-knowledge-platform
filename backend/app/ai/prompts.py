"""Prompt text for the article pipeline. Plain Python string templates --
no LangChain prompt-template machinery, since these are single-purpose,
non-reusable-across-providers strings and that dependency would buy
nothing here (see ARCHITECTURE.md's "no premature infrastructure").
"""

from app.ai.schemas import TopicItem
from app.models.chunk import Chunk
from app.models.topic import Topic

FIDELITY_CONSTRAINTS = """\
Ground everything you write ONLY in the transcript excerpts provided below. Follow these rules strictly:
- Do not invent facts or add information not present in the transcript.
- Do not fabricate quotations. If you quote someone, the words must appear in the transcript.
- Preserve who said what -- do not merge different speakers' claims into one, and do not attribute \
one person's statement to another.
- Distinguish opinions and speculation from factual claims; do not turn a speaker's speculation into \
a stated fact.
- Preserve disagreements and contrasting viewpoints between speakers rather than flattening them into \
a single consensus view.
- Do not remove important qualifications, caveats, or uncertainty the speaker expressed.
- If the transcript does not support a point, leave it out rather than filling the gap."""


def _format_chunk(index: int, chunk: Chunk) -> str:
    # Chunk has no single speaker field (it aggregates multiple
    # TranscriptSegments, which may span speakers) -- attribution is left
    # to whatever the transcript text itself makes clear, never invented.
    return f"[chunk {index}]\n{chunk.text}"


def topic_analysis_prompt(chunks: list[Chunk], start_index: int) -> tuple[str, str]:
    """Returns (system, user) for one topic-analysis batch. `start_index`
    is this batch's offset into the FULL transcript's chunk list --
    chunk_sequence_numbers in the response are numbered starting there
    (not from 0 for every batch), so per-batch results can be concatenated
    directly without renumbering in Python.
    """
    system = (
        "You are analyzing a batch of consecutive excerpts (\"chunks\") from a long-form podcast "
        "transcript. Identify the major topics, subtopics, key ideas, and important claims discussed "
        "in THIS batch only. Group chunks into topics by meaning, not by chunk boundaries -- a topic "
        "commonly spans several consecutive chunks. Do not invent a topic transition, a claim, or a "
        "speaker attribution that isn't actually there. Do not attempt to detect deep, abstract "
        "conversational subtext; describe what was actually discussed."
    )
    body = "\n\n".join(_format_chunk(start_index + i, c) for i, c in enumerate(chunks))
    user = (
        f"Transcript excerpts (chunk numbers {start_index}-{start_index + len(chunks) - 1}):\n\n{body}\n\n"
        "Identify the topics discussed in these chunks. For each topic, list the chunk numbers "
        "(exactly as given above, e.g. \"chunk 7\" -> the integer 7) that discuss it."
    )
    return system, user


def topic_boundary_merge_prompt(candidates: list[tuple[int, TopicItem]]) -> tuple[str, str]:
    """Returns (system, user) for the boundary-scoped topic merge call.

    `candidates` are ONLY the topics next to a batch boundary (see
    app/ai/nodes/topic_analysis.py's _boundary_candidate_indices) -- every
    other topic has already been kept as-is by Python and is never shown
    to the model. Each candidate's index (into the full, pre-merge topic
    list) is given so a merge decision can reference it. Only a merged
    title/summary is asked for -- chunk numbers, claims, and subtopics for
    a merged topic are always reconstructed deterministically in Python
    from the original topics (see TopicMergeGroup in app/ai/schemas.py),
    so there's nothing for the model to renumber, invent, or drop.
    """
    system = (
        "You previously analyzed a long podcast transcript in separate batches. The topics below are "
        "only the ones next to a batch boundary -- every other topic has already been kept as-is and "
        "is not shown to you here. Some of these may actually be the same topic split in two by the "
        "batch boundary; others are merely adjacent and should stay separate. Merge only where it is "
        "clearly the same topic continuing across the boundary -- do not merge topics that are merely "
        "related."
    )
    topic_lines = "\n\n".join(f"[topic {index}] {item.title}\n{item.summary}" for index, item in candidates)
    user = (
        f"Boundary-adjacent topics:\n\n{topic_lines}\n\n"
        "For each group of topics that are actually the same topic, return the topic numbers (exactly "
        "as given above, e.g. \"topic 7\" -> the integer 7) being merged, plus a merged title and "
        "summary covering all of them. Omit any topic that should stay separate -- it will be kept as "
        "its own topic automatically. If nothing should merge, return an empty list."
    )
    return system, user


def planning_prompt(
    topics: list[Topic], *, target_section_count_min: int, target_section_count_max: int
) -> tuple[str, str]:
    system = (
        "You are planning a knowledge article that turns a long-form podcast conversation into a "
        "substantially shorter, coherent, standalone article -- not a transcript summary. The article "
        "must be grounded in the topics below; do not invent a section about something not covered in "
        "them. Preserve important disagreements/contrasting viewpoints as their own planning notes so "
        "the writer doesn't flatten them later. Decide a sensible title, introduction, section "
        "ordering, and conclusion. Aim for approximately "
        f"{target_section_count_min}-{target_section_count_max} sections for a conversation of this "
        "length -- fewer is fine if the conversation genuinely covers less ground, and more is fine if "
        "it genuinely needs it, but never split or merge sections merely to hit a number."
    )
    topic_lines = "\n\n".join(
        f"[topic {t.sequence_number}] {t.title}\n{t.summary}\n"
        f"Key claims: {'; '.join(c.get('text', '') for c in t.key_claims) or '(none)'}"
        for t in topics
    )
    user = (
        f"Topics identified in the conversation:\n\n{topic_lines}\n\n"
        "Produce an article plan. For each section, list the topic numbers (exactly as given above, "
        "e.g. \"topic 3\" -> the integer 3) it should draw on."
    )
    return system, user


def _format_topic_note(topic: Topic) -> str:
    lines = [f"Topic: {topic.title}", topic.summary]
    if topic.key_claims:
        claim_lines = [
            f"- ({claim.get('claim_type', 'opinion')}, {claim.get('speaker') or 'unknown speaker'}) "
            f"{claim.get('text', '')}"
            for claim in topic.key_claims
        ]
        lines.append("Claims:\n" + "\n".join(claim_lines))
    if topic.subtopics:
        lines.append(f"Subtopics: {', '.join(topic.subtopics)}")
    return "\n".join(lines)


def section_generation_prompt(
    *,
    heading: str,
    key_ideas: list[str],
    viewpoints: list[str],
    attribution_notes: list[str],
    supporting_chunks: list[Chunk],
    relevant_topics: list[Topic],
    article_title: str,
    section_headings: list[str],
    current_section_number: int,
    revision_feedback: str | None = None,
) -> tuple[str, str]:
    """Returns (system, user) for generating one article section.

    Three kinds of material, in strictly decreasing authority -- see the
    system prompt below and PRODUCT_SPEC.md/ARCHITECTURE.md's fidelity
    requirements:
    1. `supporting_chunks` -- raw transcript excerpts. The only actual
       evidence; every claim in the generated section must trace back to
       these.
    2. `relevant_topics` -- topic analysis's own summary/claims for the
       topic(s) this section draws on (app/ai/nodes/topic_analysis.py).
       An interpretation layered on top of the chunks, included so the
       writer doesn't have to re-derive it from scratch, and to carry
       forward each claim's speaker/claim_type for attribution -- never a
       substitute for the chunks, and never authoritative if the two
       disagree.
    3. `section_headings`/`article_title` -- purely structural context
       (this section's place in the whole article), carrying no factual
       content of its own; used only so this section doesn't repeat
       material another section owns.

    Deliberately narrow on all three: only THIS section's own supporting
    chunks and topics (never the full chunk set or the full knowledge
    layer), and only the other sections' HEADINGS (never their generated
    prose, which doesn't exist yet when sections are generated in
    sequence, and is never resent even during a later revision).
    """
    system = (
        "You are writing one section of a knowledge article derived from a podcast conversation. "
        f"{FIDELITY_CONSTRAINTS}\n\n"
        "You are given three kinds of material, in order of authority. (1) SOURCE MATERIAL -- raw "
        "transcript excerpts; the only actual evidence, and the sole source of truth for what was "
        "said. (2) TOPIC NOTES -- a previously extracted summary and claims for context and "
        "attribution only; this is an interpretation layered on the excerpts, not evidence in its own "
        "right, and never outweighs the raw excerpts if the two ever seem to disagree. Where a claim's "
        "speaker or claim_type (fact/opinion/speculation) is given, use it to distinguish stated facts "
        "from opinions, speculation, or personal anecdotes in your writing -- but only when the source "
        "excerpts actually support it. (3) ARTICLE TITLE/STRUCTURE -- purely structural, so you know "
        "this section's place in the whole piece and avoid repeating material assigned to another "
        "section; it carries no factual content of its own. "
        "Write substantive, readable prose (not bullet points, not a transcript excerpt) that a reader "
        "who never heard the podcast could understand on its own."
    )

    parts = [f"ARTICLE TITLE:\n{article_title}", "", "ARTICLE STRUCTURE:"]
    for i, sec_heading in enumerate(section_headings):
        marker = "  <-- YOU ARE WRITING THIS SECTION" if i == current_section_number else ""
        parts.append(f"{i + 1}. {sec_heading}{marker}")

    parts.append(f"\nCURRENT SECTION ({current_section_number + 1} of {len(section_headings)}): {heading}")
    parts.append(f"Key ideas to cover: {'; '.join(key_ideas) or '(none specified)'}")
    if viewpoints:
        parts.append(f"Viewpoints/disagreements to preserve: {'; '.join(viewpoints)}")
    if attribution_notes:
        parts.append(f"Attribution notes: {'; '.join(attribution_notes)}")

    if relevant_topics:
        topic_notes = "\n\n".join(_format_topic_note(t) for t in relevant_topics)
        parts.append(f"\nTOPIC NOTES (interpretation/context only -- see SOURCE MATERIAL for evidence):\n\n{topic_notes}")

    chunk_text = "\n\n".join(f"[source excerpt {i}]\n{c.text}" for i, c in enumerate(supporting_chunks))
    parts.append(f"\nSOURCE MATERIAL (the actual evidence this section must be grounded in):\n\n{chunk_text}")

    if revision_feedback:
        parts.append(
            f"\nThe previous draft of this section had the following problem(s), fix them in this "
            f"draft: {revision_feedback}"
        )
    user = "\n".join(parts)
    return system, user


def llm_coherence_review_prompt(article_text: str) -> tuple[str, str]:
    system = (
        "You are reviewing a generated knowledge article for basic coherence and obvious problems -- "
        "not a fidelity re-check (that's already done deterministically), just whether the writing "
        "reads as a coherent, well-organized standalone article."
    )
    user = f"Article:\n\n{article_text}\n\nIs this coherent and well-organized? Note any obvious issues."
    return system, user
