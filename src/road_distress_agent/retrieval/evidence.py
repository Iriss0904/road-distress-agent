"""Unified evidence retrieval with local reranking and LLM-aware selection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol, TypeVar

from road_distress_agent.retrieval.channels import (
    ChannelRetrievalRequest,
    ChannelRetrievalResult,
    ExactRetrievalTool,
    retrieve_channel_candidates,
)
from road_distress_agent.retrieval.cross_refs import (
    ChunkLookupTool,
    expand_cross_references,
)
from road_distress_agent.retrieval.evidence_options import EvidenceRetrievalOptions
from road_distress_agent.retrieval.evidence_selector import select_evidence
from road_distress_agent.retrieval.llm_reranker import (
    EvidencePick,
    EvidenceRerankOutput,
)
from road_distress_agent.retrieval.protection import merge_protected_chunks
from road_distress_agent.retrieval.rerank_passages import (
    ChunkRerankScore,
    build_rerank_passage_batch,
)
from road_distress_agent.runtime_timing import elapsed_ms_since, node_timing_trace_event
from road_distress_agent.state import AuditEvent, RetrievalAttempt, RetrievedChunk
from road_distress_agent.tools.bm25_retrieval import (
    QdrantBM25RetrievalTool,
    get_default_bm25_retrieval_tool,
)
from road_distress_agent.tools.chunk_lookup import QdrantChunkLookupTool
from road_distress_agent.tools.local_reranker import get_default_bge_reranker
from road_distress_agent.tools.qdrant_rag import (
    QdrantRAGTool,
    get_qdrant_rag_tool_for_top_k,
)
from road_distress_agent.tracing import trace_event

DEFAULT_RECALL_TOP_K = 20
DEFAULT_LOCAL_TOP_K = 10
DEFAULT_FINAL_TOP_K = 3
DEFAULT_PROTECTED_EXACT_TOP_K = 2
SEMANTIC_ROLE_FILTER = "semantic_role"
RECALL_SUBSTAGE = "recall"
CROSS_REFERENCE_SUBSTAGE = "cross_reference_expansion"
LOCAL_RERANK_SUBSTAGE = "local_bge_rerank"
LLM_SELECTOR_SUBSTAGE = "llm_evidence_selector"
__all__ = [
    "EvidencePick",
    "EvidenceRerankOutput",
    "EvidenceRetrievalOptions",
    "retrieve_evidence",
]


class Reranker(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


StageResult = TypeVar("StageResult")


@dataclass(frozen=True)
class _RecallStage:
    tool: QdrantRAGTool
    exact_tool: ExactRetrievalTool | None
    result: ChannelRetrievalResult
    timing: AuditEvent


def retrieve_evidence(
    options: EvidenceRetrievalOptions,
    *,
    tool: QdrantRAGTool | None = None,
    exact_tool: ExactRetrievalTool | None = None,
    reranker: Reranker | None = None,
    llm_output: EvidenceRerankOutput | None = None,
) -> tuple[list[RetrievedChunk], list[RetrievalAttempt], list[AuditEvent]]:
    """Retrieve top20, local-rerank top10, then LLM-select top3 evidence."""
    if not options.query.strip():
        raise ValueError("Evidence retrieval requires a non-empty query.")
    recall_stage = _initial_recall(options, tool, exact_tool)
    recall = recall_stage.result
    if not recall.chunks:
        return [], recall.attempts, [*recall.traces, recall_stage.timing]
    expanded, cross_ref_timing = _timed_stage(
        options.node_name,
        CROSS_REFERENCE_SUBSTAGE,
        lambda: expand_cross_references(
            options,
            recall.chunks,
            _chunk_lookup_tool(recall_stage.tool, recall_stage.exact_tool),
        ),
    )
    local_chunks, local_timing = _timed_stage(
        options.node_name,
        LOCAL_RERANK_SUBSTAGE,
        lambda: _local_rerank(options, expanded.chunks, reranker),
    )
    selection, selector_timing = _timed_stage(
        options.node_name,
        LLM_SELECTOR_SUBSTAGE,
        lambda: _llm_select(options, local_chunks, llm_output),
    )
    selected, llm_trace = selection
    return (
        selected,
        [*recall.attempts, *expanded.attempts],
        [
            *recall.traces,
            recall_stage.timing,
            *expanded.traces,
            cross_ref_timing,
            _local_trace(options, local_chunks),
            local_timing,
            llm_trace,
            selector_timing,
        ],
    )


def _initial_recall(
    options: EvidenceRetrievalOptions,
    supplied_tool: QdrantRAGTool | None,
    supplied_exact_tool: ExactRetrievalTool | None,
) -> _RecallStage:
    active_tool = supplied_tool or get_qdrant_rag_tool_for_top_k(options.recall_top_k)
    active_exact_tool = _exact_tool(supplied_tool, active_tool, supplied_exact_tool)
    filters = _recall_filters(options)
    result, timing = _timed_stage(
        options.node_name,
        RECALL_SUBSTAGE,
        lambda: retrieve_channel_candidates(
            _channel_request(options, filters), active_tool, active_exact_tool
        ),
    )
    return _RecallStage(active_tool, active_exact_tool, result, timing)


def _timed_stage(
    node_name: str,
    substage_name: str,
    call: Callable[[], StageResult],
) -> tuple[StageResult, AuditEvent]:
    started = perf_counter()
    result = call()
    timing = node_timing_trace_event(
        node_name=node_name,
        substage_name=substage_name,
        duration_ms=elapsed_ms_since(started),
    )
    return result, timing


def _exact_tool(
    supplied_semantic_tool: QdrantRAGTool | None,
    active_tool: QdrantRAGTool,
    supplied_exact_tool: ExactRetrievalTool | None,
) -> ExactRetrievalTool | None:
    if supplied_exact_tool is not None:
        return supplied_exact_tool
    if supplied_semantic_tool is None:
        return get_default_bm25_retrieval_tool()
    if not isinstance(active_tool, QdrantRAGTool):
        return None
    return QdrantBM25RetrievalTool(
        client=active_tool._client,
        collection=active_tool._collection,
    )


def _chunk_lookup_tool(
    active_tool: QdrantRAGTool,
    active_exact_tool: ExactRetrievalTool | None,
) -> ChunkLookupTool | None:
    for tool in (active_exact_tool, active_tool):
        if tool is not None and hasattr(tool, "fetch_by_chunk_ids"):
            return tool  # type: ignore[return-value]
    if isinstance(active_tool, QdrantRAGTool):
        return QdrantChunkLookupTool(
            client=active_tool._client,
            collection=active_tool._collection,
            url=active_tool._url,
        )
    return None


def _channel_request(
    options: EvidenceRetrievalOptions,
    filters: dict[str, Any],
) -> ChannelRetrievalRequest:
    return ChannelRetrievalRequest(
        node_name=options.node_name,
        title=options.title,
        query=options.query,
        filters=filters,
        requested_filters=options.filters,
        stage_goal=options.stage_goal,
        recall_top_k=options.recall_top_k,
        bm25_top_k=options.bm25_top_k,
        protected_exact_top_k=options.protected_exact_top_k,
        explicit_clause_ids=options.explicit_clause_ids,
    )


def _recall_filters(options: EvidenceRetrievalOptions) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in options.filters.items()
        if options.preserve_semantic_role_filter or key != SEMANTIC_ROLE_FILTER
    }


def _local_rerank(
    options: EvidenceRetrievalOptions,
    chunks: list[RetrievedChunk],
    reranker: Reranker | None,
) -> list[RetrievedChunk]:
    active_reranker = reranker or get_default_bge_reranker()
    batch = build_rerank_passage_batch(
        chunks,
        split_numbered_clauses=options.split_numbered_rerank_passages,
    )
    passage_scores = active_reranker.score(options.query, batch.passages)
    scores = batch.aggregate(passage_scores)
    scored = [_with_rerank_score(chunk, score) for chunk, score in zip(chunks, scores, strict=True)]
    ranked = sorted(scored, key=_score, reverse=True)
    kept = merge_protected_chunks(
        ranked,
        scored,
        options.local_top_k,
        options.protected_exact_top_k,
    )
    return [_with_rank(chunk, rank) for rank, chunk in enumerate(kept, 1)]


def _with_rerank_score(chunk: RetrievedChunk, result: ChunkRerankScore) -> RetrievedChunk:
    existing = chunk.annotations or {}
    annotations = {
        **existing,
        "qdrant_score": existing.get("qdrant_score", chunk.score),
        "recall_score": chunk.score,
        "rerank_score": result.score,
        "evidence_quality": result.score,
        "rerank_passage_count": result.passage_count,
        "rerank_best_passage_index": result.best_passage_index,
    }
    return chunk.model_copy(
        update={"score": result.score, "similarity": result.score, "annotations": annotations}
    )


def _with_rank(chunk: RetrievedChunk, rank: int) -> RetrievedChunk:
    return chunk.model_copy(update={"rank": rank})


def _score(chunk: RetrievedChunk) -> float:
    return chunk.score if chunk.score is not None else chunk.similarity or 0.0


def _local_trace(options: EvidenceRetrievalOptions, chunks: list[RetrievedChunk]) -> AuditEvent:
    return trace_event(
        node_name=options.node_name,
        kind="rerank",
        title="Local BGE evidence rerank",
        inputs={"query": options.query, "stage_goal": options.stage_goal},
        output={"top_k": len(chunks), "chunks": chunks},
    )


def _llm_select(
    options: EvidenceRetrievalOptions,
    chunks: list[RetrievedChunk],
    supplied: EvidenceRerankOutput | None,
) -> tuple[list[RetrievedChunk], AuditEvent]:
    return select_evidence(options, chunks, supplied)
