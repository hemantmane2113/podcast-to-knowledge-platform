"""Prompt text for the article pipeline. Plain Python string templates --
no LangChain prompt-template machinery, since these are single-purpose,
non-reusable-across-providers strings and that dependency would buy
nothing here (see ARCHITECTURE.md's "no premature infrastructure").
"""

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


def topic_merge_prompt(topics_json: str) -> tuple[str, str]:
    system = (
        "You previously analyzed a long podcast transcript in separate batches and identified topics "
        "in each batch independently. Some topics near a batch boundary may actually be the same topic "
        "split in two, or two adjacent topics that should stay separate. Merge only where it is clearly "
        "the same topic continuing; do not merge topics that are merely related. Preserve every chunk "
        "number exactly as given -- do not renumber, invent, or drop any."
    )
    user = (
        f"Here are the per-batch topics as JSON:\n\n{topics_json}\n\n"
        "Return the final, merged, ordered list of topics in the same schema."
    )
    return system, user


def planning_prompt(topics: list[Topic]) -> tuple[str, str]:
    system = (
        "You are planning a knowledge article that turns a long-form podcast conversation into a "
        "substantially shorter, coherent, standalone article -- not a transcript summary. The article "
        "must be grounded in the topics below; do not invent a section about something not covered in "
        "them. Preserve important disagreements/contrasting viewpoints as their own planning notes so "
        "the writer doesn't flatten them later. Decide a sensible title, introduction, section "
        "ordering, and conclusion."
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


def section_generation_prompt(
    heading: str,
    key_ideas: list[str],
    viewpoints: list[str],
    attribution_notes: list[str],
    supporting_chunks: list[Chunk],
    revision_feedback: str | None = None,
) -> tuple[str, str]:
    system = (
        "You are writing one section of a knowledge article derived from a podcast conversation. "
        f"{FIDELITY_CONSTRAINTS}\n\n"
        "Write substantive, readable prose (not bullet points, not a transcript excerpt) that a reader "
        "who never heard the podcast could understand on its own."
    )
    chunk_text = "\n\n".join(f"[source excerpt {i}]\n{c.text}" for i, c in enumerate(supporting_chunks))
    parts = [
        f"Section heading: {heading}",
        f"Key ideas to cover: {'; '.join(key_ideas) or '(none specified)'}",
    ]
    if viewpoints:
        parts.append(f"Viewpoints/disagreements to preserve: {'; '.join(viewpoints)}")
    if attribution_notes:
        parts.append(f"Attribution notes: {'; '.join(attribution_notes)}")
    parts.append(f"\nSource excerpts this section must be grounded in:\n\n{chunk_text}")
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
