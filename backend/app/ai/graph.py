"""The AI workflow graph (Phase C-H), and only the AI workflow -- ordinary
CRUD/DB workflows in this codebase stay plain services (see
app/services/ingestion_service.py, transcript_processing_service.py),
per the explicit "LangGraph only for the AI workflow, never for CRUD"
decision.

    chunks -> topic_analysis -> planning -> section_generation -> validation
                                                                        |
                                                        passed? --no--> revision -> (back to validation)
                                                                        |
                                                                       yes
                                                                        |
                                                                       END

No checkpointer is configured (see app/ai/state.py's docstring) -- this
compiles to a plain in-process graph for one arq job invocation.
"""

from typing import Literal

from langgraph.graph import END, StateGraph

from app.ai.nodes import planning, revision, section_generation, topic_analysis, validation
from app.ai.state import ArticlePipelineState, PipelineDeps


def _should_revise(state: ArticlePipelineState) -> Literal["revise", "end"]:
    report = state["validation_report"]
    if report.passed:
        return "end"
    if state.get("revision_count", 0) >= state.get("max_revision_attempts", 2):
        return "end"
    return "revise"


def build_graph(deps: PipelineDeps):
    graph = StateGraph(ArticlePipelineState)

    graph.add_node("topic_analysis", topic_analysis.build(deps))
    graph.add_node("planning", planning.build(deps))
    graph.add_node("section_generation", section_generation.build(deps))
    graph.add_node("validation", validation.build(deps))
    graph.add_node("revision", revision.build(deps))

    graph.set_entry_point("topic_analysis")
    graph.add_edge("topic_analysis", "planning")
    graph.add_edge("planning", "section_generation")
    graph.add_edge("section_generation", "validation")
    graph.add_conditional_edges("validation", _should_revise, {"revise": "revision", "end": END})
    graph.add_edge("revision", "validation")

    return graph.compile()


async def run_article_pipeline(deps: PipelineDeps, initial_state: ArticlePipelineState) -> ArticlePipelineState:
    compiled = build_graph(deps)
    # recursion_limit guards against an unexpected infinite loop -- the
    # revise/validate cycle is already bounded by max_revision_attempts,
    # this is a hard backstop, not the primary control.
    final_state = await compiled.ainvoke(initial_state, config={"recursion_limit": 25})
    return final_state
