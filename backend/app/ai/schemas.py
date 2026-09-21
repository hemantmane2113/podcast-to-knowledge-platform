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


class PlannedSection(BaseModel):
    heading: str
    key_ideas: list[str] = Field(default_factory=list)
    # References into the topic list given in the prompt, by each topic's
    # own (already-persisted, stable) sequence_number.
    supporting_topic_sequence_numbers: list[int] = Field(default_factory=list)
    viewpoints: list[str] = Field(default_factory=list)
    attribution_notes: list[str] = Field(default_factory=list)


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
