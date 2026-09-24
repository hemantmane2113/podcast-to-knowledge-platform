"""Prompt text for the article pipeline. Plain Python string templates --
no LangChain prompt-template machinery, since these are single-purpose,
non-reusable-across-providers strings and that dependency would buy
nothing here (see ARCHITECTURE.md's "no premature infrastructure").
"""

from dataclasses import dataclass

from app.ai.schemas import TopicItem
from app.models.chunk import Chunk
from app.models.topic import Topic


@dataclass(frozen=True)
class EpisodeContext:
    """Deterministic episode metadata threaded into planning_prompt and
    section_generation_prompt purely for grounding -- always built from
    the stored Episode row (app/models/episode.py), never from the model.
    Both fields are optional since ingestion may not have captured them
    yet; a missing field is simply omitted from the prompt, never
    fabricated.

    Deliberately does NOT carry youtube_url: the URL belongs only to the
    deterministic public-article response/frontend layer
    (app/schemas/article.py::PublicArticleResponse), never to anything
    the model sees or could reconstruct/mangle in prose -- see
    Settings/PHASE 5's "do not let the LLM generate or reconstruct URLs".
    """

    title: str | None = None
    channel_name: str | None = None


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
    episode_context: EpisodeContext | None = None,
) -> tuple[str, str]:
    system = (
        "You are planning a knowledge article that turns a long-form podcast conversation into a "
        "substantially shorter, coherent, standalone EDITORIAL article -- not a transcript summary, and "
        "not a sequence of disconnected mini-summaries. The article must be grounded in the topics "
        "below; do not invent a section about something not covered in them, and do not invent a "
        "relationship between two topics that they don't actually support -- every transition or "
        "purpose you write must describe a real connection, never one invented to make the narrative "
        "read more smoothly.\n\n"
        "Before deciding on sections, identify the single central question, tension, argument, or idea "
        "that gives the whole conversation its coherence -- the thing the article is really about. "
        "State it explicitly as the core of introduction_summary; every section's purpose and every "
        "transition should ultimately trace back to it.\n\n"
        "As a loose orientation -- never a template to fill in mechanically, the transcript's own "
        "content decides the real structure -- a well-built long-form article often moves from: the "
        "central idea or question, to the important mechanisms/ideas behind it, to practical "
        "implications, to complications or tensions, to personal stories or concrete examples, to "
        "broader meaning. Use only as much of this arc as the conversation actually supports.\n\n"
        "For EACH section, in addition to its heading and content, decide:\n"
        "- narrative_purpose: one sentence explaining what the reader understands or gains after this "
        'section that they didn\'t before -- e.g. "Establish the central principle that anchors the '
        'rest of the article." Never a generic label such as "introduction", "body", "conclusion", '
        '"mechanism", or "summary".\n'
        "- transition_from_previous: one sentence describing the actual logical relationship to the "
        "previous section -- a consequence, a complication, a contrast, a deeper layer of the same "
        'idea, and so on. Never filler such as "this continues the discussion", "this builds on the '
        'previous section", "next, we discuss...", or "another important aspect..."; leave it blank '
        "for the first section rather than write a filler sentence.\n\n"
        "Treat repetition deliberately: assign each distinct idea to the ONE section that explains it "
        "most fully. A later section may revisit that idea only if it adds a clearly different layer "
        "(a complication, a concrete example, a consequence) -- say what that new layer is in its "
        "narrative_purpose. If two moments in the conversation cover essentially the same ground, keep "
        "the stronger one and leave the other out rather than give it a thin section of its own.\n\n"
        "Prefer fewer, substantial sections over many small ones -- a section earns its place by doing "
        "real work in the article's progression, not by improving topic coverage; never add a section "
        "just because a topic exists. Personal stories, concrete examples, and moments of disagreement "
        "or tension are often the most engaging material -- place them where they actually help the "
        "reader understand or stay engaged (illustrating a point just made, grounding an abstract idea, "
        "marking a genuine complication), not as a separate section for every anecdote in the "
        "conversation.\n\n"
        "Preserve important disagreements/contrasting viewpoints as their own planning notes so the "
        "writer doesn't flatten them later. Decide a sensible title, introduction, section ordering, and "
        "conclusion. The introduction should establish the central question or tension the conversation "
        "explores and give the reader a genuine reason to keep reading. Do not plan a standalone "
        "biography section merely to introduce a guest -- if who the guest is or their background is "
        "actually relevant to why their perspective matters, and is supported by the conversation or "
        "the podcast context given below, a brief one-to-two-sentence mention belongs naturally inside "
        "the opening section, not a section of its own. Never invent credentials, achievements, "
        "affiliations, or biographical detail that isn't actually supported by what you're given -- if "
        "you don't have it, leave it out rather than guessing. The conversation itself is always the "
        "priority; guest context is a small addition to it, never the article's focus. The conclusion "
        "should return to that same central question and offer a synthesis, not a restatement of every "
        "section in order, and should not introduce a new major topic just because the conversation "
        "happened to cover it late.\n\n"
        "For every section you should be able to answer: what does the reader know after it that they "
        "didn't before, why does it come at this point rather than earlier or later, and what makes "
        "them want to keep reading into the next one -- narrative_purpose and transition_from_previous "
        "are where those answers belong. Write transition_from_previous specifically enough that "
        "whoever writes this section's actual prose could open with a real sentence expressing that "
        "relationship, not just know that some relationship exists -- name the kind of connection (a "
        "consequence, a complication, a contrast, a deeper layer of the same idea, an example, a "
        "resolution) rather than a vague \"this follows the previous section\".\n\n"
        f"Aim for approximately {target_section_count_min}-{target_section_count_max} sections for a "
        "conversation of this length -- fewer is fine if the conversation genuinely covers less ground, "
        "and more is fine if it genuinely needs it, but never split or merge sections merely to hit a "
        "number, and never add one just to raise topic coverage. The finished article should read at "
        f"roughly {target_word_count_min}-{target_word_count_max} words in total -- reach that mainly "
        "through tighter scope and less repetition, favoring one well-developed treatment of each idea "
        "over spreading it across multiple sections that each re-explain it, not by cutting genuinely "
        "important ideas, stories, or qualifications."
    )
    topic_lines = "\n\n".join(
        f"[topic {t.sequence_number}] {t.title}\n{t.summary}\n"
        f"Key claims: {'; '.join(c.get('text', '') for c in t.key_claims) or '(none)'}"
        for t in topics
    )
    context_block = ""
    if episode_context and (episode_context.title or episode_context.channel_name):
        context_lines = ["PODCAST CONTEXT (factual, given -- never embellish or add to it):"]
        if episode_context.title:
            context_lines.append(f"Episode/video title: {episode_context.title}")
        if episode_context.channel_name:
            context_lines.append(f"Channel/show: {episode_context.channel_name}")
        context_block = "\n".join(context_lines) + "\n\n"
    user = (
        f"{context_block}"
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
    next_narrative_purpose: str | None = None,
    target_word_count_min: int | None = None,
    target_word_count_max: int | None = None,
    revision_feedback: str | None = None,
    episode_context: EpisodeContext | None = None,
) -> tuple[str, str]:
    """Returns (system, user) for generating one article section. The
    response is validated against GeneratedSection (app/ai/schemas.py),
    whose `paragraphs: list[str]` (not a single `content` blob) makes
    paragraph boundaries an explicit part of the model's own output --
    the system prompt below tells the model this directly.

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

    Separately, for editorial coherence (not evidence): `introduction_summary`
    is the article's own central idea/question/tension, shown to every
    section (not just the first/last) so a middle section can still write
    toward it, not just the ones with a special opening/closing block;
    `narrative_purpose`/`transition_from_previous` are this section's own
    planning notes; `preceding_sections_key_ideas` is each EARLIER
    section's own planned key_ideas (never full prose -- see
    app/ai/nodes/section_generation.py, which builds this from the plan,
    not from the article); `preceding_section_excerpt` is a short,
    deterministically-truncated tail of the IMMEDIATELY preceding
    section's actual generated content (not an LLM summary -- see the same
    module's excerpt_preceding_section); and `next_narrative_purpose` is a
    single short forward-looking line -- the immediately FOLLOWING
    section's own planned purpose, never its heading (already visible in
    ARTICLE STRUCTURE) or its prose (doesn't exist yet). None of these
    carry new factual claims of their own; they exist purely so this
    section reads as part of ONE continuous article, not an independent
    summary of a transcript slice.
    """
    system = (
        "You are writing one section of a knowledge article derived from a podcast conversation. The "
        "article as a whole must read as ONE continuous, coherent, engaging editorial piece -- not a "
        f"collection of independent summaries stitched together. {FIDELITY_CONSTRAINTS}\n\n"
        "You are given three kinds of evidentiary material, in order of authority. (1) SOURCE MATERIAL "
        "-- raw transcript excerpts; the only actual evidence, and the sole source of truth for what was "
        "said. Do not pull in anything outside these excerpts even if it would improve the prose, and "
        "if the same idea appears in several excerpts, that repetition belongs to the transcript, not a "
        "reason to explain the idea more than once here. (2) TOPIC NOTES -- a previously extracted "
        "summary and claims for context and attribution only; this is an interpretation layered on the "
        "excerpts, not evidence in its own right, and never outweighs the raw excerpts if the two ever "
        "seem to disagree. Where a claim's speaker or claim_type (fact/opinion/speculation) is given, "
        "use it to distinguish a stated fact from an opinion, speculation, personal experience, or "
        "disagreement in your writing -- but only when the source excerpts actually support it. "
        "(3) ARTICLE TITLE/STRUCTURE -- purely structural, so you know this section's place in the whole "
        "piece and avoid repeating material assigned to another section; it carries no factual content "
        "of its own.\n\n"
        "You are also given editorial context (not evidence): the article's central question, this "
        "section's own purpose and how it follows the previous one, sometimes a hint of where the "
        "following section is headed, and a compact view of what earlier sections already covered. Use "
        "this to write a natural continuation of the SAME article, not a standalone piece -- build on "
        "ideas already introduced rather than re-explaining them from scratch, unless you are adding a "
        "genuinely different layer (a complication, a new angle, a consequence). Never invent a fact, "
        "motivation, causal relationship, or speaker intention the source doesn't support, and never "
        "write a transition that implies a connection the source doesn't actually show.\n\n"
        "Do not open by mechanically restating the section heading, and do not open with a "
        'meta-referential phrase like "in the previous section", "building on what we discussed", "as '
        'mentioned earlier", or "next, we turn to" -- enter the idea naturally. A section that isn\'t '
        "the article's first should normally pick up from where the previous one left off and move into "
        "its own territory through the ideas themselves, not through an announced transition. Don't "
        "force a tidy mini-summary at the end of every section either -- where the material genuinely "
        "supports it, let the closing lines open toward what comes next, but never manufacture a bridge "
        "the source doesn't actually support.\n\n"
        "Personal stories, concrete examples, and moments of disagreement or tension are valuable when "
        "they sharpen the reader's understanding or engagement -- use them for that, not because an "
        "anecdote happens to be available in the source, and not as a checklist item; don't give every "
        "anecdote its own point of emphasis. Favor concrete examples, meaningful contrasts, genuine "
        "cause-and-effect, unresolved questions, and real implications over clickbait, manufactured "
        "drama, exaggerated claims, generic motivational language, or artificial suspense.\n\n"
        "Write substantive, precise, concrete prose that a reader who never heard the podcast could "
        "understand on its own -- the minimum prose that communicates the assigned ideas clearly and "
        "engagingly, not the most you could write.\n\n"
        "PARAGRAPH STRUCTURE. Your response is a LIST of paragraphs, not one block of text -- you "
        "decide the paragraph boundaries yourself, and each item you return is exactly one paragraph. "
        "Prefer paragraphs of roughly 2-5 sentences that each develop ONE connected idea. Split a "
        "paragraph when the argument changes, a new example begins, a new implication is introduced, "
        "or the reader would benefit from a breath -- not on a fixed schedule, and not to hit a word "
        "count. A short paragraph is fine, even good, when it creates emphasis or marks a genuine turn "
        "in the argument; don't make every paragraph short just for visual effect, and don't let a "
        "paragraph run on covering two or three separate ideas just because breaking it up felt "
        "unnecessary.\n\n"
        "PARAGRAPH OPENINGS. Do not repeatedly begin paragraphs with the same word or construction -- "
        'watch in particular for starting several paragraphs in the same section with "That", "This", '
        '"He"/"She", "The", or the speaker\'s name. Any one of these is fine once, where it\'s simply '
        "the natural way to say the sentence; repeated use across a section is the actual problem, not "
        "the construction itself. Vary how a paragraph opens based on what that paragraph is actually "
        'doing -- for example (these illustrate the KIND of variety to aim for, never phrases to reuse '
        'verbatim): "The distinction matters because...", "That principle becomes more practical '
        'when...", "The same logic appears in...", "For [speaker], the implication is...", "The '
        'conversation then moves from...", "What makes this important is...", "The practical '
        'consequence is...", "There is another side to this...", "This becomes especially relevant '
        'when...", "At that point, the question shifts...".\n\n'
        "PARAGRAPH-TO-PARAGRAPH TRANSITIONS. This is distinct from the transition between THIS section "
        "and the previous one (already covered by this section's own editorial context below) -- this "
        "is about the paragraphs you write here, next to each other. Not every paragraph needs an "
        "explicit transition sentence, but consecutive paragraphs should have a clear logical "
        "relationship that the prose actually expresses -- moving from a principle to an example, an "
        "example to its implication, a claim to a qualification, a physiological mechanism to the "
        "behavior it explains, a problem to a proposed solution, an anecdote to the broader lesson it "
        "illustrates, or a disagreement to how it resolves (or where it stays unresolved). Express that "
        'relationship through what the sentence actually says, never through generic filler like '
        '"furthermore", "moreover", or "additionally" used just to glue two paragraphs together.\n\n'
        "ATTRIBUTION. Preserve who said what -- never solve repeated attribution by dropping it. Vary "
        'the construction instead of reusing "[speaker] says/explains/argues" every time -- for '
        'example: "[speaker] argues that...", "his/her explanation centers on...", "in this framing, '
        '...", "the point [speaker] returns to is...", "[speaker] describes this as...", "according to '
        '[speaker]\'s account...", "[speaker] distinguishes between...", "the broader argument is..." '
        "-- again, the kind of variety to aim for, not a fixed script, and not something to apply "
        "mechanically. Do not attribute every single sentence -- once a paragraph has clearly "
        "established whose view is being presented, later sentences in that SAME paragraph can use a "
        "pronoun or a plain statement instead of repeating the attribution, as long as it stays "
        "unambiguous whose claim it still is.\n\n"
        "EVIDENCE, OPINION, AND SPECULATION. Where a topic note's claim_type is given (fact / opinion / "
        "speculation), let it guide how you phrase the claim: state a fact plainly, but keep a stated "
        'opinion, interpretation, personal experience, or speculation legibly marked as such (e.g. '
        '"[speaker] suspects..." rather than presenting a hypothesis as settled). Never convert a '
        "speaker's speculation or hypothesis into established fact, never introduce a degree of "
        "scientific certainty the source material doesn't support, and never invent an external "
        "scientific consensus the transcript doesn't actually describe. If the transcript presents "
        "something as uncertain or contested, keep that uncertainty in your prose.\n\n"
        "BULLETS. Prose is still the default, and most sections should have no bullets at all. A "
        "bulleted list is allowed only when the source material itself naturally contains a "
        "multi-step framework, a discrete set of recommendations, a sequence of actions, clearly "
        "separable principles, or a compact comparison that a reader would genuinely find easier to "
        "scan as a list than as a sentence. Introduce a list with a normal sentence rather than "
        "dropping it in unannounced. Do not convert ordinary narrative prose into bullets, and do not "
        "use a list just to make the section look shorter or more skimmable.\n\n"
        "Avoid formulaic AI-writing patterns -- phrases like \"in conclusion\", \"furthermore\", "
        '"moreover", "another important aspect", "it is important to note", "this highlights the '
        'importance of", "in today\'s fast-paced world", or "at the end of the day" -- not as a '
        "mechanical ban (one may occasionally belong) but because they signal generic, templated "
        "prose. Prioritize clarity and editorial quality over artificial stylistic variation for its "
        "own sake -- the goal is writing that doesn't read like a formula, not writing that avoids "
        "every familiar word."
    )

    parts = [f"ARTICLE TITLE:\n{article_title}"]
    if episode_context and (episode_context.title or episode_context.channel_name):
        context_lines = ["", "PODCAST CONTEXT (factual, given -- never embellish or add to it):"]
        if episode_context.title:
            context_lines.append(f"Episode/video title: {episode_context.title}")
        if episode_context.channel_name:
            context_lines.append(f"Channel/show: {episode_context.channel_name}")
        parts.extend(context_lines)
    parts.extend(["", "ARTICLE STRUCTURE:"])
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

    # The article's central question, shown to every MIDDLE section (not
    # just first/last, which already get the fuller opening/closing blocks
    # below that include this same text) -- otherwise a middle section has
    # no access to it at all, despite introduction_summary already being
    # passed to every section by generate_section.
    if not is_first_section and not is_last_section and introduction_summary:
        parts.append(
            f"\nThe article's central question/tension (established in the introduction): "
            f"{introduction_summary}"
        )
    # "Where appropriate, the direction of the next section" -- a single
    # short forward-looking line, never the next section's heading (already
    # visible above) or its prose (doesn't exist yet).
    if next_narrative_purpose:
        parts.append(f"\nWhat comes after this section: {next_narrative_purpose}")

    if is_first_section and introduction_summary:
        parts.append(
            "\nThis is the ARTICLE'S OPENING SECTION. Beyond the key ideas above, use it to establish "
            "the central question or tension this article explores and give the reader a genuine, "
            "specific reason to keep reading -- not generic scene-setting. Do not write a standalone "
            "biography of the guest, but if who they are or their background is genuinely relevant to "
            "why their perspective matters here, and is supported by the podcast context or source "
            "material given below, a brief one-to-two-sentence mention of who they are can appear "
            "naturally as part of this opening -- never invented credentials, achievements, "
            "affiliations, or background beyond what you're actually given. Set the article's narrative "
            f"direction so later sections read as a natural continuation.\nPlanned introduction intent: "
            f"{introduction_summary}"
        )
    if is_last_section and conclusion_summary:
        # Includes introduction_summary (not just conclusion_summary) --
        # "return to the central question the introduction raised" is an
        # empty instruction without the actual text of what that was.
        # introduction_summary already reaches this function for every
        # section (see generate_section), this block just hadn't been
        # rendering it.
        central_question_line = (
            f" The article's central question, as established in the introduction: {introduction_summary}"
            if introduction_summary
            else ""
        )
        parts.append(
            "\nThis is the ARTICLE'S CLOSING SECTION. Beyond the key ideas above, use it to return to "
            "the central question or idea the introduction raised and offer a synthesis -- do not "
            "mechanically restate each earlier section, and do not introduce a completely new major "
            f"topic here.{central_question_line}\nPlanned conclusion intent: {conclusion_summary}"
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
            f"\nThis section should run roughly {target_word_count_min}-{target_word_count_max} words -- "
            "a loose guide, not a target to fill. Use the minimum prose that communicates the assigned "
            "ideas clearly and engagingly, not the most you could write; never pad to reach the range, "
            "and never cut a genuinely necessary point just to stay under it."
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


def article_editorial_review_prompt(
    *,
    article_title: str,
    sections: list[dict],
    article_word_count: int,
    target_word_count_min: int,
    target_word_count_max: int,
) -> tuple[str, str]:
    """Returns (system, user) for the whole-article editorial review
    (Batch 5) -- the ONE call in this pipeline that reads the fully
    ASSEMBLED article (never the raw transcript, never per-chunk retrieval)
    to catch problems that only exist at the article level: repetition
    across non-adjacent sections, weak transitions, disproportionate
    length, an introduction/conclusion not doing its job. Never a fidelity
    re-check (app/services/article_validation.py and FIDELITY_CONSTRAINTS
    already own that) -- this reviews organization, repetition,
    proportion, and flow only, and never itself rewrites anything; its
    findings are handed to the existing per-section revision mechanism
    (app/ai/nodes/revision.py), the same one a failed deterministic check
    already targets.

    `sections` is `[{"sequence_number", "heading", "content"}, ...]` in
    order -- the assembled article, nothing more.
    """
    system = (
        "You are the editorial reviewer for a knowledge article assembled from several independently "
        "generated sections of a podcast-derived article. Section-by-section generation cannot "
        "reliably catch problems that only become visible once the WHOLE article is read straight "
        "through -- that is your job: read it as a single reader would and identify genuine "
        "article-level problems, not section-level nitpicks (wording, style, and fidelity are already "
        "handled elsewhere).\n\n"
        "Look specifically for:\n"
        "- Repetition: the same idea, mechanism, or explanation appearing in more than one section -- "
        "especially non-adjacent sections, which section-by-section generation cannot see. Flag the "
        "LATER section; a later section may legitimately revisit an idea only if it adds a genuinely "
        "different layer (a new implication, a complication, a different angle) -- if it doesn't, "
        "flag it.\n"
        "- Weak or missing connection between consecutive sections -- a section that reads like an "
        "unrelated topic dropped in rather than following from what came before.\n"
        "- An introduction that doesn't establish the article's central question/tension, or reads "
        "like a generic podcast intro, a biography, or a list of topics.\n"
        "- A conclusion that merely repeats earlier sections instead of synthesizing, or that "
        "introduces an unrelated major topic.\n"
        "- Sections that are disproportionately long or short relative to their actual role in the "
        "article.\n"
        "- Tangents that don't serve the article's central question.\n"
        "- An important idea that gets too little explanation, or a minor one that gets too much.\n\n"
        f"The article's length target is a SOFT editorial guide of roughly {target_word_count_min}-"
        f"{target_word_count_max} words; it is currently {article_word_count} words. If it is within "
        "or close to that range and reads well, do not flag length as a problem at all. If it runs "
        "substantially over, identify WHICH specific sections should be tightened and why -- "
        "prioritize cutting duplicated ideas, repeated explanations, unnecessary framing, redundant "
        "examples, and verbose transitions, in that order, never by suggesting every section be "
        "uniformly shortened. Never suggest cutting a qualification, a disagreement, a source of "
        "uncertainty, attribution, a key example that materially helps understanding, or an idea the "
        "narrative arc actually needs.\n\n"
        "For each section you flag in sections_needing_revision, give specific, actionable feedback a "
        "writer could act on without re-reading the whole article -- name what it repeats or where the "
        "actual problem is, never a vague \"improve flow\". Only flag a section for a genuine problem, "
        "not because it could theoretically be tighter. If the article has no real article-level "
        "problems, say so and leave sections_needing_revision empty -- do not invent issues to justify "
        "a review. Use overall_feedback only for a genuine whole-article concern that isn't really any "
        "one section's fault.\n\n"
        "You are reviewing only for these article-level issues -- never suggest adding, removing, or "
        "changing a fact, an attribution, or a source claim; that is out of scope here."
    )
    section_blocks = "\n\n".join(
        f"## Section {s['sequence_number']}: {s['heading']}\n\n{s['content']}" for s in sections
    )
    user = (
        f"ARTICLE TITLE: {article_title}\n\n{section_blocks}\n\n---\n"
        f"Total length: {article_word_count} words (soft target: {target_word_count_min}-"
        f"{target_word_count_max})."
    )
    return system, user
