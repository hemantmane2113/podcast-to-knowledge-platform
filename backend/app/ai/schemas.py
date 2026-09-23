"""Structured LLM I/O schemas for the article pipeline (Phase C/E/F).

Chunks and topics are referenced by small integers (a chunk's position in
the batch sent to the model, a topic's `sequence_number`), never by raw
UUID -- models reliably mangle/hallucinate UUIDs but handle small integers
well. app/ai/nodes/ resolves these integers back to real database UUIDs
before anything is persisted; nothing in this module or in the prompts
ever asks the model to invent or transcribe an ID.
"""

from typing import Literal

from pydantic import BaseModel, Field


class TopicClaim(BaseModel):
    text: str
    speaker: str | None = None
    claim_type: Literal["fact", "opinion", "speculation"] = "opinion"


class TopicItem(BaseModel):
    title: str
    summary: str
    # Positions (0-based) into the numbered chunk list given in the
    # prompt for this batch -- see app/ai/nodes/topic_analysis.py, which
    # offsets these to global chunk-list indices before persisting.
    chunk_sequence_numbers: list[int] = Field(default_factory=list)
    key_claims: list[TopicClaim] = Field(default_factory=list)
    subtopics: list[str] = Field(default_factory=list)


class TopicAnalysisResult(BaseModel):
    topics: list[TopicItem]


class TopicMergeGroup(BaseModel):
    """Two or more boundary-candidate topics (see
    app/ai/nodes/topic_analysis.py's _boundary_candidate_indices) that the
    model has decided are the same topic split across a batch boundary.
    `topic_indices` refers to the index each candidate was given in the
    boundary-merge prompt -- never a raw UUID (same reasoning as
    TopicItem.chunk_sequence_numbers above).

    Deliberately narrow: only the two fields that are genuinely a semantic
    judgment (a new title/summary covering the merged topic) come from the
    model. chunk_sequence_numbers, key_claims, and subtopics for the
    merged topic are always reconstructed in Python from the ORIGINAL
    topics being merged, never taken from the model's response -- so the
    model has no opportunity to drop a chunk reference or a claim, even
    accidentally.
    """

    topic_indices: list[int] = Field(default_factory=list)
    merged_title: str
    merged_summary: str


class TopicMergeDecision(BaseModel):
    """Response schema for the boundary-scoped topic merge call. Empty
    `merges` means none of the boundary candidates should merge -- every
    one of them is kept as its own original topic."""

    merges: list[TopicMergeGroup] = Field(default_factory=list)


class PlannedSection(BaseModel):
    heading: str
    key_ideas: list[str] = Field(default_factory=list)
    # References into the topic list given in the prompt, by each topic's
    # own (already-persisted, stable) sequence_number.
    supporting_topic_sequence_numbers: list[int] = Field(default_factory=list)
    viewpoints: list[str] = Field(default_factory=list)
    attribution_notes: list[str] = Field(default_factory=list)
    # Editorial planning notes (free text, not a rigid enum -- the
    # transcript determines the actual structure, this just carries the
    # planner's own reasoning forward to section_generation_prompt so a
    # section isn't written from a heading alone). Both default to "" --
    # backward compatible with an already-persisted ArticlePlan.sections
    # row from before this field existed (see app/ai/nodes/section_generation.py's
    # planned_section.get(..., "") reads) and with existing test fixtures
    # that construct a PlannedSection without them.
    narrative_purpose: str = Field(
        default="",
        description=(
            "One sentence, in your own words, explaining this section's editorial role in the "
            'article -- e.g. "Establish the central principle that anchors the rest of the article." '
            "Not a category label."
        ),
    )
    transition_from_previous: str = Field(
        default="",
        description=(
            "One sentence explaining why this section naturally follows the previous one -- what "
            "makes it the next step for the reader, not just the next topic on a list. Blank for the "
            "first section."
        ),
    )


class ArticlePlanResult(BaseModel):
    title: str
    introduction_summary: str
    sections: list[PlannedSection]
    conclusion_summary: str


class GeneratedSection(BaseModel):
    heading: str
    content: str


class LLMCoherenceReview(BaseModel):
    """Optional, supplementary LLM-based check (Settings.enable_llm_validation)
    -- never folds into ValidationReport.passed (app/services/article_validation.py
    stays entirely deterministic); see app/ai/nodes/validation.py."""

    coherent: bool
    notes: str
