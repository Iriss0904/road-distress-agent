"""Retrieve construction safety norms from the dedicated safety document."""

from __future__ import annotations

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.state import AgentState, AuditEvent, RetrievalAttempt
from road_distress_agent.tools.qdrant_rag import (
    QdrantRAGTool,
    get_qdrant_rag_tool_for_top_k,
)
from road_distress_agent.tracing import retrieval_trace_event

SAFETY_SOURCE_DOC_ID: str | None = None
SAFETY_TOP_K = 5


def _query(state: AgentState) -> str:
    query = state.get("safety_query") or ""
    if not query.strip():
        raise ValueError("safety_norm_retriever requires a non-empty safety_query.")
    return query


def _retrieve(query: str, tool: QdrantRAGTool) -> AgentState:
    filters = {"source_doc_id": SAFETY_SOURCE_DOC_ID} if SAFETY_SOURCE_DOC_ID else {}
    chunks = tool.search_documents(query, filters)
    return {
        "safety_norm_chunks": chunks,
        "retrieval_attempts": [
            RetrievalAttempt(
                tool_name="QdrantRAGTool.search_documents",
                query=query,
                filters=filters,
                citation_count=len(chunks),
            )
        ],
        "audit_log": [
            retrieval_trace_event(
                node_name="safety_norm_retriever",
                title="Safety norm document search",
                query=query,
                filters=filters,
                chunks=chunks,
                tool=tool,
            ),
            AuditEvent(
                node_name="safety_norm_retriever",
                message="Retrieved construction safety norm chunks.",
                metadata={"chunk_count": len(chunks), "source_doc_id": SAFETY_SOURCE_DOC_ID},
            ),
        ],
    }


def safety_norm_retriever(state: AgentState) -> AgentState:
    query = _query(state)
    result = _retrieve(query, get_qdrant_rag_tool_for_top_k(SAFETY_TOP_K))
    return {
        **result,
        "phase": WorkflowPhase.EVIDENCE,
        "next_action": "load_weather_or_advise_arrangement",
    }
