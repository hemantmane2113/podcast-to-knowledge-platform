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
from app.ai.nodes.topic_analysis import (
    _batch_chunks,
    _boundary_candidate_indices,
    _merge_topics,
    _reconstruct_topics,
)
from app.ai.schemas import TopicClaim, TopicItem, TopicMergeDecision, TopicMergeGroup
from app.models.chunk import Chunk
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
