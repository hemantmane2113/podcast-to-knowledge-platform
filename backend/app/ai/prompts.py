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
    topics: list[Topic],
    *,
    target_section_count_min: int,
    target_section_count_max: int,
    target_word_count_min: int,
    target_word_count_max: int,
) -> tuple[str, str]:
    system = (
        "You are planning a knowledge article that turns a long-form podcast conversation into a "
        "substantially shorter, coherent, standalone EDITORIAL article -- not a transcript summary, and "
        "not a sequence of disconnected mini-summaries. The article must be grounded in the topics "
        "below; do not invent a section about something not covered in them.\n\n"
        "As a loose orientation -- never a template to fill in mechanically, the transcript's own "
        "content decides the real structure -- a well-built long-form article often moves from: the "
        "central idea or question, to the important mechanisms/ideas behind it, to practical "
        "implications, to complications or tensions, to personal stories or concrete examples, to "
        "broader meaning. Use only as much of this arc as the conversation actually supports.\n\n"
        "For EACH section, in addition to its heading and content, decide:\n"
        "- narrative_purpose: one sentence, in your own words, explaining this section's editorial role "
        '(e.g. "Establish the central principle that anchors the rest of the article.") -- an actual '
        "explanation of what the section is for, not a category label.\n"
        "- transition_from_previous: one sentence on why this section is the natural next step after "
        "the previous one, not just the next topic on a list (leave blank for the first section).\n\n"
        "Preserve important disagreements/contrasting viewpoints as their own planning notes so the "
        "writer doesn't flatten them later. Decide a sensible title, introduction, section ordering, and "
        "conclusion. The introduction should establish the central question or tension the conversation "
        "explores and give the reader a genuine reason to keep reading -- not a biography of the "
        "speakers unless the biography itself is directly relevant. The conclusion should return to that "
        "central question and offer a synthesis, not a restatement of every section in order.\n\n"
        f"Aim for approximately {target_section_count_min}-{target_section_count_max} sections for a "
        "conversation of this length -- fewer is fine if the conversation genuinely covers less ground, "
        "and more is fine if it genuinely needs it, but never split or merge sections merely to hit a "
        f"number. The finished article should read at roughly {target_word_count_min}-{target_word_count_max} "
        "words in total -- size and scope sections with that in mind, favoring one well-developed "
        "treatment of each idea over spreading it across multiple sections that each re-explain it."
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


def _format_preceding_sections(preceding_sections_key_ideas: list[tuple[str, list[str]]]) -> str:
    lines = []
    for heading, ideas in preceding_sections_key_ideas:
        idea_text = "; ".join(ideas) if ideas else "(no key ideas recorded)"
        lines.append(f'- "{heading}": {idea_text}')
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
    narrative_purpose: str = "",
    transition_from_previous: str = "",
    introduction_summary: str = "",
    conclusion_summary: str = "",
    preceding_sections_key_ideas: list[tuple[str, list[str]]] | None = None,
    preceding_section_excerpt: str | None = None,
    target_word_count_min: int | None = None,
    target_word_count_max: int | None = None,
    revision_feedback: str | None = None,
) -> tuple[str, str]:
    """Returns (system, user) for generating one article section.

    Three kinds of *evidentiary* material, in strictly decreasing
    authority -- see the system prompt below and
    PRODUCT_SPEC.md/ARCHITECTURE.md's fidelity requirements:
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
    prose) reach the ARTICLE STRUCTURE block above.

    Separately, for editorial coherence (not evidence): `narrative_purpose`/
    `transition_from_previous` are this section's own planning notes;
    `preceding_sections_key_ideas` is each EARLIER section's own planned
    key_ideas (never full prose -- see app/ai/nodes/section_generation.py,
    which builds this from the plan, not from the article); and
    `preceding_section_excerpt` is a short, deterministically-truncated
    tail of the IMMEDIATELY preceding section's actual generated content
    (not an LLM summary -- see the same module's excerpt_preceding_section).
    None of these carry new factual claims of their own; they exist purely
    so this section can flow from and build on what came before instead of
    re-explaining it.
    """
    system = (
        "You are writing one section of a knowledge article derived from a podcast conversation. The "
        "article as a whole should read as a coherent, engaging editorial piece -- not a collection of "
        f"transcript summaries stitched together. {FIDELITY_CONSTRAINTS}\n\n"
        "You are given three kinds of evidentiary material, in order of authority. (1) SOURCE MATERIAL "
        "-- raw transcript excerpts; the only actual evidence, and the sole source of truth for what was "
        "said. (2) TOPIC NOTES -- a previously extracted summary and claims for context and "
        "attribution only; this is an interpretation layered on the excerpts, not evidence in its own "
        "right, and never outweighs the raw excerpts if the two ever seem to disagree. Where a claim's "
        "speaker or claim_type (fact/opinion/speculation) is given, use it to distinguish stated facts "
        "from opinions, speculation, or personal anecdotes in your writing -- but only when the source "
        "excerpts actually support it. (3) ARTICLE TITLE/STRUCTURE -- purely structural, so you know "
        "this section's place in the whole piece and avoid repeating material assigned to another "
        "section; it carries no factual content of its own.\n\n"
        "You are also given editorial context (not evidence): this section's intended purpose and how "
        "it follows the previous one, and a compact view of what earlier sections already covered. Use "
        "this to write a natural continuation -- build on ideas already introduced rather than "
        "re-explaining them from scratch, unless you are adding a genuinely new layer to one. Only "
        "repeat something already covered when that new layer earns it.\n\n"
        "Write substantive, readable prose (not bullet points, not a transcript excerpt) that a reader "
        "who never heard the podcast could understand on its own. Vary how paragraphs open -- do not "
        'repeatedly start with constructions like "X says", "X explains", or "X argues" -- and avoid '
        'repetitive AI-style transitions such as "In conclusion", "Furthermore", "Moreover", or '
        '"Another important aspect is". Where the transcript naturally supports it, use questions, '
        "contrasts, concrete examples, or a brief story to give the reader a reason to keep reading -- "
        "never invented drama, sensationalism, or fake suspense."
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
    if narrative_purpose:
        parts.append(f"This section's editorial purpose: {narrative_purpose}")
    if transition_from_previous:
        parts.append(f"Why this section follows the previous one: {transition_from_previous}")

    is_first_section = current_section_number == 0
    is_last_section = current_section_number == len(section_headings) - 1
    if is_first_section and introduction_summary:
        parts.append(
            "\nThis is the ARTICLE'S OPENING SECTION. Beyond the key ideas above, use it to establish "
            "the central question or tension this article explores and give the reader a genuine, "
            "specific reason to keep reading -- not generic scene-setting, and not a biography of the "
            "speaker(s) unless directly relevant to that central question. Set the article's narrative "
            f"direction so later sections read as a natural continuation.\nPlanned introduction intent: "
            f"{introduction_summary}"
        )
    if is_last_section and conclusion_summary:
        parts.append(
            "\nThis is the ARTICLE'S CLOSING SECTION. Beyond the key ideas above, use it to return to "
            "the central question or idea the introduction raised and offer a synthesis -- do not "
            "mechanically restate each earlier section, and do not introduce a completely new major "
            f"topic here.\nPlanned conclusion intent: {conclusion_summary}"
        )

    if preceding_sections_key_ideas or preceding_section_excerpt:
        already_covered_parts = [
            "\nALREADY COVERED BY EARLIER SECTIONS (context only, not evidence -- build on these "
            "rather than re-explaining them; repeating one is fine only if you add a genuinely new "
            "layer to it):"
        ]
        if preceding_sections_key_ideas:
            already_covered_parts.append(_format_preceding_sections(preceding_sections_key_ideas))
        if preceding_section_excerpt:
            already_covered_parts.append(
                f'\nThe immediately preceding section ended with:\n"{preceding_section_excerpt}"'
            )
        parts.append("\n".join(already_covered_parts))

    if relevant_topics:
        topic_notes = "\n\n".join(_format_topic_note(t) for t in relevant_topics)
        parts.append(f"\nTOPIC NOTES (interpretation/context only -- see SOURCE MATERIAL for evidence):\n\n{topic_notes}")

    if target_word_count_min and target_word_count_max:
        parts.append(
            f"\nThis section should run roughly {target_word_count_min}-{target_word_count_max} words "
            "-- an approximate guide, not a hard limit; do not pad to reach it or cut a genuinely "
            "necessary point just to stay under it."
        )

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
