"""Direct unit tests for the pure-logic helpers inside app/ai/nodes/ --
these are otherwise only exercised indirectly through the full LangGraph
integration test (tests/integration/test_article_pipeline_graph.py),
which proves the pipeline works end-to-end but doesn't pin down this
logic's edge cases on its own (e.g. a chunk larger than the whole token
budget, or a check whose details mention two different sections).
"""

import uuid
from types import SimpleNamespace

from app.ai.nodes.revision import _sections_needing_revision
from app.ai.nodes.section_generation import (
    collect_preceding_key_ideas,
    excerpt_preceding_section,
    generate_section,
    next_section_narrative_purpose,
    section_headings_by_sequence,
    section_word_target,
)
from app.ai.nodes.topic_analysis import (
    _batch_chunks,
    _boundary_candidate_indices,
    _merge_topics,
    _reconstruct_topics,
)
from app.ai.prompts import section_generation_prompt
from app.ai.schemas import (
    GeneratedSection,
    PlannedSection,
    TopicClaim,
    TopicItem,
    TopicMergeDecision,
    TopicMergeGroup,
)
from app.config.settings import Settings
from app.models.chunk import Chunk
from app.models.topic import Topic
from tests.fakes import FakeLLMProvider


def _chunk(text: str, sequence_number: int = 0) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        transcript_id=uuid.uuid4(),
        episode_id=uuid.uuid4(),
        sequence_number=sequence_number,
        text=text,
        start_ms=0,
        end_ms=1000,
        source_segment_ids=[],
        token_count=len(text) // 4 or 1,
    )


# --- _batch_chunks (app/ai/nodes/topic_analysis.py) ---------------------------------


def test_batch_chunks_empty_input_returns_no_batches() -> None:
    assert _batch_chunks([], token_budget=100) == []


def test_batch_chunks_single_chunk_under_budget_is_one_batch() -> None:
    chunk = _chunk("short text")
    batches = _batch_chunks([chunk], token_budget=100)
    assert batches == [[chunk]]


def test_batch_chunks_multiple_small_chunks_fit_in_one_batch() -> None:
    chunks = [_chunk("word " * 5, i) for i in range(5)]
    batches = _batch_chunks(chunks, token_budget=1000)
    assert len(batches) == 1
    assert batches[0] == chunks


def test_batch_chunks_splits_when_budget_is_exceeded() -> None:
    # Each chunk is ~10 tokens (40 chars / 4); a budget of 15 fits one
    # chunk comfortably but not two.
    chunks = [_chunk("x" * 40, i) for i in range(4)]
    batches = _batch_chunks(chunks, token_budget=15)

    assert len(batches) == 4  # one chunk per batch
    # Every chunk appears exactly once, in order, across all batches.
    flattened = [c for batch in batches for c in batch]
    assert flattened == chunks


def test_batch_chunks_groups_as_many_as_fit_under_the_budget() -> None:
    # Four ~10-token chunks with a budget of 25 -> batches of 2, then 2.
    chunks = [_chunk("x" * 40, i) for i in range(4)]
    batches = _batch_chunks(chunks, token_budget=25)

    assert len(batches) == 2
    assert batches[0] == chunks[:2]
    assert batches[1] == chunks[2:]


def test_batch_chunks_a_single_oversized_chunk_still_gets_its_own_batch() -> None:
    # A chunk whose own token estimate exceeds the budget must not be
    # dropped -- it becomes a (necessarily over-budget) batch of one,
    # rather than being silently skipped or merged incorrectly.
    huge = _chunk("x" * 4000, 0)  # ~1000 tokens
    small = _chunk("y" * 40, 1)  # ~10 tokens
    batches = _batch_chunks([huge, small], token_budget=50)

    assert len(batches) == 2
    assert batches[0] == [huge]
    assert batches[1] == [small]


def test_batch_chunks_preserves_order_across_batches() -> None:
    chunks = [_chunk("x" * 40, i) for i in range(6)]
    batches = _batch_chunks(chunks, token_budget=20)
    flattened = [c for batch in batches for c in batch]
    assert [c.sequence_number for c in flattened] == list(range(6))


# --- _sections_needing_revision (app/ai/nodes/revision.py) --------------------------


def test_sections_needing_revision_all_passed_returns_nothing() -> None:
    checks = [{"name": "x", "passed": True, "details": "ok"}]
    targeted, shared = _sections_needing_revision(checks)
    assert targeted == set()
    assert shared == []


def test_sections_needing_revision_targets_the_section_named_in_details() -> None:
    checks = [{"name": "no_empty_sections", "passed": False, "details": "section 2 is empty or too short"}]
    targeted, shared = _sections_needing_revision(checks)
    assert targeted == {2}
    assert shared == []


def test_sections_needing_revision_falls_back_to_shared_feedback_with_no_section_number() -> None:
    checks = [{"name": "article_length", "passed": False, "details": "Article is only 50 words."}]
    targeted, shared = _sections_needing_revision(checks)
    assert targeted == set()
    assert len(shared) == 1
    assert "article_length" in shared[0]


def test_sections_needing_revision_extracts_multiple_section_numbers_from_one_check() -> None:
    checks = [
        {
            "name": "no_duplicate_sections",
            "passed": False,
            "details": "section 0 duplicates heading of section 2",
        }
    ]
    targeted, shared = _sections_needing_revision(checks)
    assert targeted == {0, 2}
    assert shared == []


def test_sections_needing_revision_combines_targeted_and_shared_across_checks() -> None:
    checks = [
        {"name": "no_empty_sections", "passed": False, "details": "section 1 is empty or too short"},
        {"name": "article_length", "passed": False, "details": "Article is only 50 words."},
        {"name": "source_coverage", "passed": True, "details": "3/3 chunks (100%) referenced."},
    ]
    targeted, shared = _sections_needing_revision(checks)
    assert targeted == {1}
    assert len(shared) == 1
    assert "article_length" in shared[0]


def test_sections_needing_revision_ignores_passed_checks_entirely() -> None:
    checks = [
        {"name": "source_traceability", "passed": True, "details": "section 5 looks fine"},
    ]
    targeted, shared = _sections_needing_revision(checks)
    assert targeted == set()
    assert shared == []


# --- boundary-scoped topic merge (app/ai/nodes/topic_analysis.py) -------------------
#
# Two batches, chunk index 2 is batch 0's last chunk and chunk index 3 is
# batch 1's first chunk -- topics 1 and 2 straddle that seam and are the
# only ones that can be boundary candidates; topics 0 and 3 stay fully
# inside their own batch and can never be candidates, however they're
# positioned in their batch's own topic list.

_TWO_BATCH_TOPIC_RANGES = [(0, 2), (2, 4)]
_TWO_BATCH_CHUNK_RANGES = [(0, 2), (3, 5)]


def _topic(
    title: str,
    summary: str,
    chunk_sequence_numbers: list[int],
    claims: list[TopicClaim] | None = None,
    subtopics: list[str] | None = None,
) -> TopicItem:
    return TopicItem(
        title=title,
        summary=summary,
        chunk_sequence_numbers=chunk_sequence_numbers,
        key_claims=claims or [],
        subtopics=subtopics or [],
    )


def _two_batch_topics() -> list[TopicItem]:
    return [
        _topic("Topic 0", "s0", [0, 1]),  # batch 0, doesn't touch the seam
        _topic(
            "Topic 1",
            "s1",
            [1, 2],  # touches batch 0's last chunk (2)
            claims=[TopicClaim(text="claim1", speaker="Alice", claim_type="fact")],
            subtopics=["sub1"],
        ),
        _topic(
            "Topic 2",
            "s2",
            [3, 4],  # touches batch 1's first chunk (3)
            claims=[TopicClaim(text="claim2", speaker="Bob", claim_type="opinion")],
            subtopics=["sub2"],
        ),
        _topic("Topic 3", "s3", [4, 5]),  # batch 1, doesn't touch the seam
    ]


def test_boundary_candidate_indices_only_includes_topics_touching_the_seam() -> None:
    topics = _two_batch_topics()
    candidates = _boundary_candidate_indices(topics, _TWO_BATCH_TOPIC_RANGES, _TWO_BATCH_CHUNK_RANGES)
    assert candidates == {1, 2}


async def test_merge_topics_no_candidates_skips_the_llm_call_entirely() -> None:
    # No programmed responses at all -- FakeLLMProvider raises if it's
    # ever called, so this also proves no call was attempted.
    deps = SimpleNamespace(llm_provider=FakeLLMProvider(structured_responses=[]))
    topics = _two_batch_topics()

    result = await _merge_topics(deps, topics, candidate_indices=set())

    assert result == topics


# 1. Multiple batches with no boundary merge -> final output exactly preserves all topics.
async def test_merge_topics_llm_finds_no_boundary_merge_preserves_all_topics() -> None:
    deps = SimpleNamespace(llm_provider=FakeLLMProvider(structured_responses=[TopicMergeDecision(merges=[])]))
    topics = _two_batch_topics()

    result = await _merge_topics(deps, topics, candidate_indices={1, 2})

    assert result == topics


# 2. One valid boundary merge -> two topics become one; chunk references preserved.
async def test_merge_topics_applies_one_valid_boundary_merge() -> None:
    decision = TopicMergeDecision(
        merges=[TopicMergeGroup(topic_indices=[1, 2], merged_title="Merged", merged_summary="merged summary")]
    )
    deps = SimpleNamespace(llm_provider=FakeLLMProvider(structured_responses=[decision]))
    topics = _two_batch_topics()

    result = await _merge_topics(deps, topics, candidate_indices={1, 2})

    assert [t.title for t in result] == ["Topic 0", "Merged", "Topic 3"]
    merged = result[1]
    assert merged.summary == "merged summary"
    assert merged.chunk_sequence_numbers == [1, 2, 3, 4]  # union of topics 1 and 2
    assert {c.text for c in merged.key_claims} == {"claim1", "claim2"}
    assert set(merged.subtopics) == {"sub1", "sub2"}


# 3. Multiple boundary merges.
def _three_batch_topics() -> list[TopicItem]:
    return [
        _topic("T0", "s0", [0]),  # batch 0, not a candidate
        _topic("T1", "s1", [1]),  # batch 0, touches boundary 0 (chunk 1)
        _topic("T2", "s2", [2]),  # batch 1, touches boundary 0 (chunk 2)
        _topic("T3", "s3", [3]),  # batch 1, touches boundary 1 (chunk 3)
        _topic("T4", "s4", [4]),  # batch 2, touches boundary 1 (chunk 4)
        _topic("T5", "s5", [5]),  # batch 2, not a candidate
    ]


_THREE_BATCH_TOPIC_RANGES = [(0, 2), (2, 4), (4, 6)]
_THREE_BATCH_CHUNK_RANGES = [(0, 1), (2, 3), (4, 5)]


def test_boundary_candidate_indices_across_multiple_boundaries() -> None:
    topics = _three_batch_topics()
    candidates = _boundary_candidate_indices(topics, _THREE_BATCH_TOPIC_RANGES, _THREE_BATCH_CHUNK_RANGES)
    assert candidates == {1, 2, 3, 4}


async def test_merge_topics_applies_multiple_independent_boundary_merges() -> None:
    decision = TopicMergeDecision(
        merges=[
            TopicMergeGroup(topic_indices=[1, 2], merged_title="M1", merged_summary="s1"),
            TopicMergeGroup(topic_indices=[3, 4], merged_title="M2", merged_summary="s2"),
        ]
    )
    deps = SimpleNamespace(llm_provider=FakeLLMProvider(structured_responses=[decision]))
    topics = _three_batch_topics()

    result = await _merge_topics(deps, topics, candidate_indices={1, 2, 3, 4})

    assert [t.title for t in result] == ["T0", "M1", "M2", "T5"]


# 8. Deterministic ordering of reconstructed topics.
def test_reconstruct_topics_ordering_is_independent_of_merge_list_order() -> None:
    topics = _three_batch_topics()
    group_a = TopicMergeGroup(topic_indices=[1, 2], merged_title="M1", merged_summary="s1")
    group_b = TopicMergeGroup(topic_indices=[3, 4], merged_title="M2", merged_summary="s2")

    forward = _reconstruct_topics(topics, {1, 2, 3, 4}, [group_a, group_b])
    reversed_order = _reconstruct_topics(topics, {1, 2, 3, 4}, [group_b, group_a])

    assert [t.title for t in forward] == ["T0", "M1", "M2", "T5"]
    assert [t.title for t in forward] == [t.title for t in reversed_order]


# --- a single topic spanning three consecutive batches (one 3-way merge) -----------
#
# Topic A' (the middle batch's topic) touches BOTH batch boundaries at
# once -- it's a candidate for boundary 0 (as the right-hand side) and for
# boundary 1 (as the left-hand side) simultaneously. This is the case a
# naive "only pairwise, only adjacent" implementation could get wrong by
# producing two separate 2-way merges (A+A', A'+A'') that both include A'
# instead of one 3-way merge.


def _three_way_span_topics() -> list[TopicItem]:
    return [
        _topic(
            "Topic A",
            "sA",
            [0],  # batch 0's only/last chunk
            claims=[TopicClaim(text="claimA", speaker="Alice", claim_type="fact")],
            subtopics=["subA"],
        ),
        _topic(
            "Topic A'",
            "sA'",
            [1],  # batch 1's only chunk -- touches both boundary 0 and boundary 1
            claims=[TopicClaim(text="claimA'", speaker="Bob", claim_type="opinion")],
            subtopics=["subA'"],
        ),
        _topic(
            "Topic A''",
            "sA''",
            [2],  # batch 2's only/first chunk
            claims=[TopicClaim(text="claimA''", speaker="Carol", claim_type="speculation")],
            subtopics=["subA''"],
        ),
    ]


_SPAN_TOPIC_RANGES = [(0, 1), (1, 2), (2, 3)]
_SPAN_CHUNK_RANGES = [(0, 0), (1, 1), (2, 2)]


def test_boundary_candidate_indices_topic_spanning_three_batches_is_a_candidate_at_both_seams() -> None:
    topics = _three_way_span_topics()
    candidates = _boundary_candidate_indices(topics, _SPAN_TOPIC_RANGES, _SPAN_CHUNK_RANGES)
    assert candidates == {0, 1, 2}


async def test_merge_topics_merges_a_topic_spanning_three_consecutive_batches_into_one() -> None:
    # The merge LLM returns ONE group containing all three candidates;
    # reconstruction must produce exactly one final topic that carries
    # forward everything from all three sources -- not two overlapping
    # pairwise merges that would duplicate the middle topic.
    decision = TopicMergeDecision(
        merges=[
            TopicMergeGroup(
                topic_indices=[0, 1, 2], merged_title="Unified Topic", merged_summary="the full story"
            )
        ]
    )
    deps = SimpleNamespace(llm_provider=FakeLLMProvider(structured_responses=[decision]))
    topics = _three_way_span_topics()

    result = await _merge_topics(deps, topics, candidate_indices={0, 1, 2})

    # Exactly one final topic -- no duplicate topics left behind.
    assert len(result) == 1
    merged = result[0]
    # 4. The merged title/summary returned by the LLM.
    assert merged.title == "Unified Topic"
    assert merged.summary == "the full story"
    # 1. All original chunk_sequence_numbers from A + A' + A''.
    # 7. No duplicate chunk references.
    assert merged.chunk_sequence_numbers == [0, 1, 2]
    # 2. All original key_claims from A + A' + A''; none dropped, none duplicated.
    assert {c.text for c in merged.key_claims} == {"claimA", "claimA'", "claimA''"}
    assert len(merged.key_claims) == 3
    # 3. All relevant subtopics.
    assert set(merged.subtopics) == {"subA", "subA'", "subA''"}


def test_reconstruct_topics_a_three_way_span_cannot_become_two_overlapping_pairwise_merges() -> None:
    # If the model incorrectly returned two separate pairwise groups that
    # both include the middle topic -- A+A' and A'+A'' -- applying both
    # would fold A' into two different merged topics at once. The second
    # group must be rejected the moment A' is already used, never
    # silently accepted (8. no source topic merged twice).
    topics = _three_way_span_topics()
    a_plus_a_prime = TopicMergeGroup(topic_indices=[0, 1], merged_title="A+A'", merged_summary="s")
    a_prime_plus_a_double_prime = TopicMergeGroup(topic_indices=[1, 2], merged_title="A'+A''", merged_summary="s")

    result = _reconstruct_topics(topics, {0, 1, 2}, [a_plus_a_prime, a_prime_plus_a_double_prime])

    # The first group is applied; the second (reusing topic 1) is
    # rejected, so A'' is preserved untouched -- never silently merged
    # into a second, conflicting topic. 6. No duplicate topics result.
    assert [t.title for t in result] == ["A+A'", "Topic A''"]
    assert len(result) == 2
    # A' appears exactly once across the whole result -- inside the one
    # merge that was actually applied, never duplicated into a second one.
    all_claim_texts = [c.text for t in result for c in t.key_claims]
    assert all_claim_texts.count("claimA'") == 1


# 4. LLM attempts to reference a nonexistent topic -> fails safely / rejects the merge.
async def test_merge_topics_rejects_a_merge_referencing_a_non_candidate_topic() -> None:
    decision = TopicMergeDecision(
        merges=[TopicMergeGroup(topic_indices=[1, 99], merged_title="Bogus", merged_summary="bogus")]
    )
    deps = SimpleNamespace(llm_provider=FakeLLMProvider(structured_responses=[decision]))
    topics = _two_batch_topics()

    result = await _merge_topics(deps, topics, candidate_indices={1, 2})

    # The invalid group is ignored entirely -- every original topic
    # (including the ones it tried to reference) is preserved untouched.
    assert result == topics


def test_reconstruct_topics_rejects_a_group_with_fewer_than_two_topics() -> None:
    topics = _two_batch_topics()
    group = TopicMergeGroup(topic_indices=[1], merged_title="Bogus", merged_summary="bogus")

    result = _reconstruct_topics(topics, {1, 2}, [group])

    assert result == topics


def test_reconstruct_topics_rejects_a_group_reusing_an_already_merged_topic() -> None:
    topics = _three_batch_topics()
    first = TopicMergeGroup(topic_indices=[1, 2], merged_title="M1", merged_summary="s1")
    conflicting = TopicMergeGroup(topic_indices=[2, 3], merged_title="Bogus", merged_summary="bogus")

    result = _reconstruct_topics(topics, {1, 2, 3, 4}, [first, conflicting])

    # `first` is applied; `conflicting` reuses topic 2 and is rejected, so
    # topic 3 is preserved untouched rather than silently dropped.
    assert [t.title for t in result] == ["T0", "M1", "T3", "T4", "T5"]


# 5. LLM attempts to drop a chunk reference -> must not silently lose source traceability.
async def test_merge_topics_chunk_references_always_come_from_python_not_the_model() -> None:
    # TopicMergeGroup has no field for chunk numbers at all -- the merged
    # topic's chunk_sequence_numbers can only ever be the union of its
    # sources' own chunk numbers, computed in Python, regardless of
    # anything the model returns.
    decision = TopicMergeDecision(
        merges=[TopicMergeGroup(topic_indices=[1, 2], merged_title="Merged", merged_summary="merged summary")]
    )
    deps = SimpleNamespace(llm_provider=FakeLLMProvider(structured_responses=[decision]))
    topics = _two_batch_topics()

    result = await _merge_topics(deps, topics, candidate_indices={1, 2})

    merged = next(t for t in result if t.title == "Merged")
    original_1 = topics[1]
    original_2 = topics[2]
    assert set(merged.chunk_sequence_numbers) == set(original_1.chunk_sequence_numbers) | set(
        original_2.chunk_sequence_numbers
    )


# 6. Claims from untouched topics remain unchanged.
async def test_merge_topics_untouched_topics_keep_their_claims_unchanged() -> None:
    decision = TopicMergeDecision(
        merges=[TopicMergeGroup(topic_indices=[1, 2], merged_title="Merged", merged_summary="merged summary")]
    )
    deps = SimpleNamespace(llm_provider=FakeLLMProvider(structured_responses=[decision]))
    topics = _two_batch_topics()

    result = await _merge_topics(deps, topics, candidate_indices={1, 2})

    untouched = next(t for t in result if t.title == "Topic 0")
    assert untouched == topics[0]
    assert untouched.key_claims == topics[0].key_claims


def test_dedup_claims_and_subtopics_removes_only_exact_duplicates() -> None:
    from app.ai.nodes.topic_analysis import _dedup_claims, _dedup_strings

    claims = [
        TopicClaim(text="same", speaker="A", claim_type="fact"),
        TopicClaim(text="same", speaker="A", claim_type="fact"),  # exact duplicate, dropped
        TopicClaim(text="same", speaker="B", claim_type="fact"),  # different speaker, kept
    ]
    deduped = _dedup_claims(claims)
    assert len(deduped) == 2

    subtopics = ["a", "b", "a"]
    assert _dedup_strings(subtopics) == ["a", "b"]


# --- section generation: topic knowledge + article outline (app/ai/nodes/section_generation.py,
# --- app/ai/prompts.py::section_generation_prompt) -----------------------------------------


def _topic_row(
    title: str,
    summary: str,
    *,
    claims: list[dict] | None = None,
    subtopics: list[str] | None = None,
) -> Topic:
    return Topic(
        id=uuid.uuid4(),
        transcript_id=uuid.uuid4(),
        episode_id=uuid.uuid4(),
        sequence_number=0,
        title=title,
        summary=summary,
        chunk_ids=[],
        key_claims=claims or [],
        subtopics=subtopics or [],
    )


def _planned_section(
    *,
    sequence_number: int,
    heading: str,
    supporting_chunk_ids: list[uuid.UUID],
    supporting_topic_ids: list[uuid.UUID],
) -> dict:
    return {
        "sequence_number": sequence_number,
        "heading": heading,
        "key_ideas": [],
        "supporting_chunk_ids": [str(cid) for cid in supporting_chunk_ids],
        "supporting_topic_ids": [str(tid) for tid in supporting_topic_ids],
        "viewpoints": [],
        "attribution_notes": [],
    }


def test_section_headings_by_sequence_orders_by_sequence_number_not_storage_order() -> None:
    sections = [
        {"sequence_number": 2, "heading": "Conclusion"},
        {"sequence_number": 0, "heading": "Intro"},
        {"sequence_number": 1, "heading": "Middle"},
    ]
    assert section_headings_by_sequence(sections) == ["Intro", "Middle", "Conclusion"]


async def test_generate_section_includes_relevant_topic_knowledge_but_excludes_irrelevant_topics() -> None:
    relevant = _topic_row(
        "Relevant Topic",
        "relevant summary",
        claims=[{"text": "relevant claim text", "speaker": "Alice", "claim_type": "fact"}],
        subtopics=["relevant subtopic"],
    )
    irrelevant = _topic_row(
        "Irrelevant Topic",
        "irrelevant summary",
        claims=[{"text": "irrelevant claim text", "speaker": "Bob", "claim_type": "opinion"}],
        subtopics=["irrelevant subtopic"],
    )
    chunk = _chunk("this section's own raw transcript excerpt", 0)
    chunk_by_id = {chunk.id: chunk}
    topic_by_id = {relevant.id: relevant, irrelevant.id: irrelevant}
    planned_section = _planned_section(
        sequence_number=0,
        heading="Section A",
        supporting_chunk_ids=[chunk.id],
        supporting_topic_ids=[relevant.id],  # only the relevant topic is referenced
    )
    deps = SimpleNamespace(
        llm_provider=FakeLLMProvider(
            structured_responses=[GeneratedSection(heading="Section A", content="generated prose")]
        )
    )

    result = await generate_section(
        deps, planned_section, chunk_by_id, topic_by_id,
        article_title="The Article", section_headings=["Section A"],
    )

    messages, system, response_model = deps.llm_provider.structured_calls[0]
    prompt_text = system + messages[0].content

    # A1. Relevant topic knowledge reaches the prompt.
    assert "Relevant Topic" in prompt_text
    assert "relevant summary" in prompt_text
    # A3. Relevant claims preserve text/speaker/claim_type -- checked as
    # one exact formatted line (not a bare "fact"/"Alice" substring
    # check, which "factual"/etc. elsewhere in the fidelity constraints
    # text would satisfy trivially either way).
    assert "- (fact, Alice) relevant claim text" in prompt_text
    assert "relevant subtopic" in prompt_text
    # A2. Irrelevant topics do not reach the prompt.
    assert "Irrelevant Topic" not in prompt_text
    assert "irrelevant summary" not in prompt_text
    assert "irrelevant claim text" not in prompt_text
    assert "Bob" not in prompt_text
    assert "irrelevant subtopic" not in prompt_text
    # A4. Raw supporting chunks are still included.
    assert "this section's own raw transcript excerpt" in prompt_text
    # A5. Source references are not altered.
    assert result.supporting_chunk_ids == [chunk.id]
    assert result.supporting_topic_ids == [relevant.id]


async def test_generate_section_omits_topic_notes_block_when_no_topics_are_relevant() -> None:
    chunk = _chunk("raw text", 0)
    planned_section = _planned_section(
        sequence_number=0, heading="Section A", supporting_chunk_ids=[chunk.id], supporting_topic_ids=[]
    )
    deps = SimpleNamespace(
        llm_provider=FakeLLMProvider(
            structured_responses=[GeneratedSection(heading="Section A", content="prose")]
        )
    )

    await generate_section(
        deps, planned_section, {chunk.id: chunk}, {},
        article_title="T", section_headings=["Section A"],
    )

    messages, _, _ = deps.llm_provider.structured_calls[0]
    assert "TOPIC NOTES" not in messages[0].content


def test_section_generation_prompt_includes_article_title_and_ordered_headings() -> None:
    _, user = section_generation_prompt(
        heading="Middle Section",
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[],
        relevant_topics=[],
        article_title="The Grand Article",
        section_headings=["Intro", "Middle Section", "Conclusion"],
        current_section_number=1,
    )

    # B1. Article title included.
    assert "The Grand Article" in user
    # B2. Ordered section headings included, in order.
    assert user.index("1. Intro") < user.index("2. Middle Section") < user.index("3. Conclusion")


def test_section_generation_prompt_structure_block_contains_only_headings() -> None:
    # B4 (structural proof): the ARTICLE STRUCTURE block can only ever
    # contain the given headings -- there is no parameter through which
    # another section's generated prose could reach it.
    headings = ["Intro", "Middle Section", "Conclusion"]
    _, user = section_generation_prompt(
        heading="Middle Section",
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[],
        relevant_topics=[],
        article_title="T",
        section_headings=headings,
        current_section_number=1,
    )

    structure_block = user.split("ARTICLE STRUCTURE:\n", 1)[1].split("\nCURRENT SECTION", 1)[0]
    lines = [line for line in structure_block.splitlines() if line.strip()]
    assert len(lines) == len(headings)
    for i, (line, heading) in enumerate(zip(lines, headings)):
        assert line.startswith(f"{i + 1}. {heading}")


def test_section_generation_prompt_identifies_the_current_section() -> None:
    _, user = section_generation_prompt(
        heading="Middle Section",
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[],
        relevant_topics=[],
        article_title="T",
        section_headings=["Intro", "Middle Section", "Conclusion"],
        current_section_number=1,
    )

    # B3. Current section is identifiable, unambiguously, in two places:
    # marked inline in the structure list, and restated with its position.
    lines = user.splitlines()
    marked_line = next(line for line in lines if "Middle Section" in line and line.strip().startswith("2."))
    assert "YOU ARE WRITING THIS SECTION" in marked_line
    assert "CURRENT SECTION (2 of 3): Middle Section" in user


def test_section_generation_prompt_never_includes_other_sections_source_text() -> None:
    own_chunk = _chunk("this section's own excerpt", 0)
    _, user = section_generation_prompt(
        heading="Middle Section",
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[own_chunk],  # only this section's own chunk, never the full transcript
        relevant_topics=[],
        article_title="T",
        section_headings=["Intro", "Middle Section", "Conclusion"],
        current_section_number=1,
    )

    # B5. Full transcript is NOT accidentally included -- only the chunk
    # actually passed in appears; nothing from any other section's chunks
    # (which were simply never given to this call) can leak in.
    assert own_chunk.text in user
    assert user.count("[source excerpt") == 1


# --- Batch 3 (editorial rework): narrative_purpose / transition / intro-conclusion / already-covered ---


def test_planned_section_narrative_fields_default_to_blank() -> None:
    # Backward compatible with an already-persisted ArticlePlan.sections
    # row from before these fields existed, and with any fixture/test that
    # constructs a PlannedSection without them (see _planned_section above).
    section = PlannedSection(heading="Some Section")
    assert section.narrative_purpose == ""
    assert section.transition_from_previous == ""


def test_section_generation_prompt_includes_narrative_purpose_and_transition_when_given() -> None:
    _, user = section_generation_prompt(
        heading="Middle Section",
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[],
        relevant_topics=[],
        article_title="T",
        section_headings=["Intro", "Middle Section", "Conclusion"],
        current_section_number=1,
        narrative_purpose="Show the mechanism behind the central claim.",
        transition_from_previous="Having established the claim, the reader now needs to see how it works.",
    )

    assert "Show the mechanism behind the central claim." in user
    assert "Having established the claim, the reader now needs to see how it works." in user


def test_section_generation_prompt_omits_narrative_fields_when_blank() -> None:
    _, user = section_generation_prompt(
        heading="Middle Section",
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[],
        relevant_topics=[],
        article_title="T",
        section_headings=["Intro", "Middle Section", "Conclusion"],
        current_section_number=1,
    )

    assert "editorial purpose" not in user
    assert "follows the previous one" not in user


def test_section_generation_prompt_adds_opening_block_only_for_first_section_with_intro_summary() -> None:
    _, first = section_generation_prompt(
        heading="Intro",
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[],
        relevant_topics=[],
        article_title="T",
        section_headings=["Intro", "Middle", "Conclusion"],
        current_section_number=0,
        introduction_summary="the central tension is X",
    )
    _, middle = section_generation_prompt(
        heading="Middle",
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[],
        relevant_topics=[],
        article_title="T",
        section_headings=["Intro", "Middle", "Conclusion"],
        current_section_number=1,
        introduction_summary="the central tension is X",  # even if passed, must not apply mid-article
    )

    assert "ARTICLE'S OPENING SECTION" in first
    assert "the central tension is X" in first
    assert "ARTICLE'S OPENING SECTION" not in middle


def test_section_generation_prompt_adds_closing_block_only_for_last_section_with_conclusion_summary() -> None:
    _, last = section_generation_prompt(
        heading="Conclusion",
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[],
        relevant_topics=[],
        article_title="T",
        section_headings=["Intro", "Middle", "Conclusion"],
        current_section_number=2,
        conclusion_summary="return to the central question",
    )
    _, middle = section_generation_prompt(
        heading="Middle",
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[],
        relevant_topics=[],
        article_title="T",
        section_headings=["Intro", "Middle", "Conclusion"],
        current_section_number=1,
        conclusion_summary="return to the central question",
    )

    assert "ARTICLE'S CLOSING SECTION" in last
    assert "return to the central question" in last
    assert "ARTICLE'S CLOSING SECTION" not in middle


def test_section_generation_prompt_closing_block_includes_the_actual_introduction_text() -> None:
    """Regression test: "return to the central question the introduction
    raised" is an empty instruction without the actual introduction text --
    the closing section must be shown introduction_summary, not just
    conclusion_summary (introduction_summary already reaches this function
    for every section via generate_section; this only checks the closing
    block actually renders it)."""
    _, last = section_generation_prompt(
        heading="Conclusion",
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[],
        relevant_topics=[],
        article_title="T",
        section_headings=["Intro", "Middle", "Conclusion"],
        current_section_number=2,
        introduction_summary="the central tension is whether X or Y",
        conclusion_summary="return to the central question",
    )

    assert "the central tension is whether X or Y" in last


def test_section_generation_prompt_includes_already_covered_block_from_both_layers() -> None:
    _, user = section_generation_prompt(
        heading="Conclusion",
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[],
        relevant_topics=[],
        article_title="T",
        section_headings=["Intro", "Middle", "Conclusion"],
        current_section_number=2,
        preceding_sections_key_ideas=[("Intro", ["idea one"]), ("Middle", ["idea two"])],
        preceding_section_excerpt="the previous section's closing thought",
    )

    assert "ALREADY COVERED" in user
    assert '"Intro": idea one' in user
    assert '"Middle": idea two' in user
    assert "the previous section's closing thought" in user


def test_section_generation_prompt_omits_already_covered_block_when_nothing_precedes() -> None:
    _, user = section_generation_prompt(
        heading="Intro",
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[],
        relevant_topics=[],
        article_title="T",
        section_headings=["Intro", "Middle", "Conclusion"],
        current_section_number=0,
    )

    assert "ALREADY COVERED" not in user


def test_section_generation_prompt_includes_soft_word_target_when_given() -> None:
    _, with_target = section_generation_prompt(
        heading="Middle",
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[],
        relevant_topics=[],
        article_title="T",
        section_headings=["Intro", "Middle", "Conclusion"],
        current_section_number=1,
        target_word_count_min=1000,
        target_word_count_max=1500,
    )
    _, without_target = section_generation_prompt(
        heading="Middle",
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[],
        relevant_topics=[],
        article_title="T",
        section_headings=["Intro", "Middle", "Conclusion"],
        current_section_number=1,
    )

    assert "1000" in with_target and "1500" in with_target
    assert "roughly" in with_target
    assert "1000" not in without_target


def test_section_generation_prompt_shows_the_central_question_to_middle_sections_only() -> None:
    """Regression test: introduction_summary was already reaching this
    function for every section, but only the first/last section's own
    elaborate opening/closing block ever rendered it -- a middle section
    had no access to the article's central question at all. Middle
    sections now get a compact standalone line; first/last sections still
    get it via their existing elaborate block, not duplicated."""
    kwargs = dict(
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[],
        relevant_topics=[],
        article_title="T",
        section_headings=["Intro", "Middle", "Conclusion"],
        introduction_summary="the central tension is whether X or Y",
    )
    _, first = section_generation_prompt(heading="Intro", current_section_number=0, **kwargs)
    _, middle = section_generation_prompt(heading="Middle", current_section_number=1, **kwargs)
    _, last = section_generation_prompt(
        heading="Conclusion", current_section_number=2, conclusion_summary="c", **kwargs
    )

    assert "the central tension is whether X or Y" in first  # via the opening block
    assert "the central tension is whether X or Y" in middle  # via the new compact line
    assert "the central tension is whether X or Y" in last  # via the closing block
    # The compact standalone line itself only applies to middle sections --
    # first/last already state the central question through their own,
    # more elaborate block.
    assert "central question/tension (established in the introduction)" in middle
    assert "central question/tension (established in the introduction)" not in first
    assert "central question/tension (established in the introduction)" not in last


def test_section_generation_prompt_includes_next_section_direction_when_given() -> None:
    _, with_next = section_generation_prompt(
        heading="Middle",
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[],
        relevant_topics=[],
        article_title="T",
        section_headings=["Intro", "Middle", "Conclusion"],
        current_section_number=1,
        next_narrative_purpose="Explore the practical implications of the mechanism just shown.",
    )
    _, without_next = section_generation_prompt(
        heading="Middle",
        key_ideas=[],
        viewpoints=[],
        attribution_notes=[],
        supporting_chunks=[],
        relevant_topics=[],
        article_title="T",
        section_headings=["Intro", "Middle", "Conclusion"],
        current_section_number=1,
    )

    assert "What comes after this section" in with_next
    assert "Explore the practical implications of the mechanism just shown." in with_next
    assert "What comes after this section" not in without_next


def test_collect_preceding_key_ideas_returns_only_earlier_sections_in_order() -> None:
    ordered = [
        {"sequence_number": 0, "heading": "Intro", "key_ideas": ["a"]},
        {"sequence_number": 1, "heading": "Middle", "key_ideas": ["b"]},
        {"sequence_number": 2, "heading": "Conclusion", "key_ideas": ["c"]},
    ]

    assert collect_preceding_key_ideas(ordered, 0) == []
    assert collect_preceding_key_ideas(ordered, 1) == [("Intro", ["a"])]
    assert collect_preceding_key_ideas(ordered, 2) == [("Intro", ["a"]), ("Middle", ["b"])]


def test_next_section_narrative_purpose_returns_the_following_sections_purpose() -> None:
    ordered = [
        {"sequence_number": 0, "heading": "Intro", "narrative_purpose": "purpose A"},
        {"sequence_number": 1, "heading": "Middle", "narrative_purpose": "purpose B"},
        {"sequence_number": 2, "heading": "Conclusion", "narrative_purpose": "purpose C"},
    ]

    assert next_section_narrative_purpose(ordered, 0) == "purpose B"
    assert next_section_narrative_purpose(ordered, 1) == "purpose C"
    assert next_section_narrative_purpose(ordered, 2) is None  # no section follows the last one


def test_next_section_narrative_purpose_handles_missing_or_blank_purpose() -> None:
    ordered = [
        {"sequence_number": 0, "heading": "Intro"},  # no narrative_purpose key at all
        {"sequence_number": 1, "heading": "Middle", "narrative_purpose": ""},
    ]

    # Next section (seq 1) has no usable purpose (missing key or blank) --
    # None either way, never an error, and never an empty-string line in
    # the prompt.
    assert next_section_narrative_purpose(ordered, 0) is None


def test_excerpt_preceding_section_uses_the_last_paragraph() -> None:
    content = "First paragraph, with earlier material.\n\nSecond paragraph, the actual ending."
    assert excerpt_preceding_section(content) == "Second paragraph, the actual ending."


def test_excerpt_preceding_section_truncates_a_long_tail_keeping_the_end() -> None:
    long_tail = "word " * 300 + "the very last words"
    excerpt = excerpt_preceding_section(long_tail)
    assert excerpt.startswith("...")
    assert excerpt.endswith("the very last words")
    assert len(excerpt) <= 503  # "..." + 500 char budget


def test_excerpt_preceding_section_handles_single_paragraph_content() -> None:
    assert excerpt_preceding_section("just one paragraph, no blank-line breaks") == (
        "just one paragraph, no blank-line breaks"
    )


def test_section_word_target_splits_evenly_across_sections() -> None:
    settings = Settings(
        _env_file=None,
        app_env="development",
        llm_provider="groq",
        groq_api_key="x",
        llm_model="m",
        article_target_word_count_min=6000,
        article_target_word_count_max=9000,
    )

    assert section_word_target(settings, 3) == (2000, 3000)
    assert section_word_target(settings, 0) == (None, None)


async def test_generate_section_forwards_narrative_and_already_covered_context_to_the_prompt() -> None:
    chunk = _chunk("raw text", 0)
    planned_section = _planned_section(
        sequence_number=1,
        heading="Body",
        supporting_chunk_ids=[chunk.id],
        supporting_topic_ids=[],
    )
    planned_section["narrative_purpose"] = "Show the mechanism."
    planned_section["transition_from_previous"] = "Because the claim was just established."
    deps = SimpleNamespace(
        llm_provider=FakeLLMProvider(
            structured_responses=[GeneratedSection(heading="Body", content="prose")]
        )
    )

    await generate_section(
        deps,
        planned_section,
        {chunk.id: chunk},
        {},
        article_title="T",
        section_headings=["Intro", "Body", "Conclusion"],
        introduction_summary="the central tension is X",
        preceding_sections_key_ideas=[("Intro", ["idea one"])],
        preceding_section_excerpt="how the intro ended",
        next_narrative_purpose="Explore what this implies in practice.",
        target_word_count_min=500,
        target_word_count_max=800,
    )

    messages, system, _ = deps.llm_provider.structured_calls[0]
    prompt_text = system + messages[0].content
    assert "Show the mechanism." in prompt_text
    assert "Because the claim was just established." in prompt_text
    assert "the central tension is X" in prompt_text
    assert "idea one" in prompt_text
    assert "how the intro ended" in prompt_text
    assert "Explore what this implies in practice." in prompt_text
    assert "500" in prompt_text and "800" in prompt_text


async def test_generate_section_omits_narrative_fields_absent_from_planned_section_dict() -> None:
    """A planned_section dict that predates narrative_purpose/
    transition_from_previous (e.g. an ArticlePlan persisted before this
    fix, or an old-style test fixture) must not raise -- .get(..., "")."""
    chunk = _chunk("raw text", 0)
    planned_section = _planned_section(
        sequence_number=0, heading="Intro", supporting_chunk_ids=[chunk.id], supporting_topic_ids=[]
    )
    assert "narrative_purpose" not in planned_section  # confirms the premise
    deps = SimpleNamespace(
        llm_provider=FakeLLMProvider(
            structured_responses=[GeneratedSection(heading="Intro", content="prose")]
        )
    )

    await generate_section(  # must not raise
        deps, planned_section, {chunk.id: chunk}, {}, article_title="T", section_headings=["Intro"]
    )
