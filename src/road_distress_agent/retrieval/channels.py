"""Retrieve and fuse semantic, BM25, and clause lookup channels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from road_distress_agent.retrieval.fusion import ChannelResults, fuse_channel_results
from road_distress_agent.retrieval.query_features import (
    KB_RETRIEVAL_NODE_NAMES,
    QueryFeatures,
    detect_query_features,
    should_run_bm25,
)
from road_distress_agent.state import (
    AuditEvent,
    RetrievalAttempt,
    RetrievalChannel,
    RetrievedChunk,
)
from road_distress_agent.tracing import retrieval_trace_event

QDRANT_CHANNEL = "qdrant"
BM25_CHANNEL = "bm25"
CLAUSE_CHANNEL = "clause"
QDRANT_WEIGHT = 1.0
BM25_KB_WEIGHT = 1.25
BM25_DIAGNOSIS_WEIGHT = 0.75
CLAUSE_WEIGHT = 3.0
DEFAULT_PROTECTED_EXACT_TOP_K = 2


class SemanticRetrievalTool(Protocol):
    def search_documents(
        self,
        query: str,
        filters: dict[str, Any],
    ) -> Sequence[RetrievedChunk]: ...


class ExactRetrievalTool(Protocol):
    def search_bm25(
        self,
        query: str,
        filters: Mapping[str, Any],
        top_k: int,
    ) -> Sequence[RetrievedChunk]: ...

    def fetch_by_clause_ids(
        self,
        clause_ids: Sequence[str],
        filters: Mapping[str, Any],
        limit: int,
    ) -> Sequence[RetrievedChunk]: ...


@dataclass(frozen=True)
class ChannelRetrievalRequest:
    node_name: str
    title: str
    query: str
    filters: Mapping[str, Any]
    requested_filters: Mapping[str, Any]
    stage_goal: str
    recall_top_k: int
    bm25_top_k: int
    protected_exact_top_k: int = DEFAULT_PROTECTED_EXACT_TOP_K
    explicit_clause_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChannelRetrievalResult:
    chunks: list[RetrievedChunk]
    attempts: list[RetrievalAttempt]
    traces: list[AuditEvent]
    features: QueryFeatures
    channel_counts: dict[str, int] = field(default_factory=dict)


def retrieve_channel_candidates(
    request: ChannelRetrievalRequest,
    semantic_tool: SemanticRetrievalTool,
    exact_tool: ExactRetrievalTool | None,
) -> ChannelRetrievalResult:
    features = _request_features(request)
    channels, attempts, traces = _semantic_channel(request, semantic_tool, features)
    _extend_exact_channels(request, exact_tool, features, channels, attempts, traces)
    fused = fuse_channel_results(channels, request.recall_top_k)
    return ChannelRetrievalResult(
        chunks=fused,
        attempts=attempts,
        traces=traces,
        features=features,
        channel_counts={channel.name: len(channel.chunks) for channel in channels},
    )


def _request_features(request: ChannelRetrievalRequest) -> QueryFeatures:
    features = detect_query_features(request.query, request.filters)
    if not request.explicit_clause_ids:
        return features
    clause_ids = tuple(dict.fromkeys(request.explicit_clause_ids))
    return replace(features, clause_ids=clause_ids)


def _semantic_channel(
    request: ChannelRetrievalRequest,
    tool: SemanticRetrievalTool,
    features: QueryFeatures,
) -> tuple[list[ChannelResults], list[RetrievalAttempt], list[AuditEvent]]:
    chunks = list(tool.search_documents(request.query, dict(request.filters)))
    attempts = [
        _attempt(
            "QdrantRAGTool.search_documents",
            request,
            len(chunks),
            channel=QDRANT_CHANNEL,
        )
    ]
    traces = [_trace(request, QDRANT_CHANNEL, chunks, tool, features)]
    channels = [
        ChannelResults(
            name=QDRANT_CHANNEL,
            chunks=chunks,
            weight=QDRANT_WEIGHT,
        )
    ]
    return channels, attempts, traces


def _extend_exact_channels(
    request: ChannelRetrievalRequest,
    exact_tool: ExactRetrievalTool | None,
    features: QueryFeatures,
    channels: list[ChannelResults],
    attempts: list[RetrievalAttempt],
    traces: list[AuditEvent],
) -> None:
    if not _needs_exact_tool(request, features):
        return
    if exact_tool is None:
        raise ValueError("Exact retrieval requested but no ExactRetrievalTool was provided.")
    _append_bm25_channel(request, exact_tool, features, channels, attempts, traces)
    _append_clause_channel(request, exact_tool, features, channels, attempts, traces)


def _needs_exact_tool(request: ChannelRetrievalRequest, features: QueryFeatures) -> bool:
    return should_run_bm25(features, request.node_name) or features.needs_clause_lookup


def _append_bm25_channel(
    request: ChannelRetrievalRequest,
    tool: ExactRetrievalTool,
    features: QueryFeatures,
    channels: list[ChannelResults],
    attempts: list[RetrievalAttempt],
    traces: list[AuditEvent],
) -> None:
    if not should_run_bm25(features, request.node_name):
        return
    chunks = list(tool.search_bm25(request.query, request.filters, request.bm25_top_k))
    attempts.append(
        _attempt(
            "QdrantBM25RetrievalTool.search_bm25",
            request,
            len(chunks),
            channel=BM25_CHANNEL,
        )
    )
    traces.append(_trace(request, BM25_CHANNEL, chunks, tool, features))
    channels.append(ChannelResults(name=BM25_CHANNEL, chunks=chunks, weight=_bm25_weight(request)))


def _append_clause_channel(
    request: ChannelRetrievalRequest,
    tool: ExactRetrievalTool,
    features: QueryFeatures,
    channels: list[ChannelResults],
    attempts: list[RetrievalAttempt],
    traces: list[AuditEvent],
) -> None:
    if not features.needs_clause_lookup:
        return
    chunks = list(
        tool.fetch_by_clause_ids(
            features.clause_ids,
            request.filters,
            request.protected_exact_top_k,
        )
    )
    attempts.append(
        _attempt(
            "QdrantBM25RetrievalTool.fetch_by_clause_ids",
            request,
            len(chunks),
            channel=CLAUSE_CHANNEL,
            requested_clause_ids=features.clause_ids,
        )
    )
    traces.append(_trace(request, CLAUSE_CHANNEL, chunks, tool, features))
    channels.append(
        ChannelResults(name=CLAUSE_CHANNEL, chunks=chunks, weight=CLAUSE_WEIGHT, protected=True)
    )


def _attempt(
    tool_name: str,
    request: ChannelRetrievalRequest,
    count: int,
    *,
    channel: RetrievalChannel,
    requested_clause_ids: tuple[str, ...] = (),
) -> RetrievalAttempt:
    return RetrievalAttempt(
        tool_name=tool_name,
        query=request.query,
        filters=dict(request.filters),
        citation_count=count,
        channel=channel,
        requested_clause_ids=requested_clause_ids,
    )


def _trace(
    request: ChannelRetrievalRequest,
    channel: str,
    chunks: list[RetrievedChunk],
    tool: Any,
    features: QueryFeatures,
) -> AuditEvent:
    return retrieval_trace_event(
        node_name=request.node_name,
        title=f"{request.title} ({channel})",
        query=request.query,
        filters=dict(request.filters),
        chunks=chunks,
        tool=tool,
        metadata=_metadata(request, channel, features),
    )


def _metadata(
    request: ChannelRetrievalRequest,
    channel: str,
    features: QueryFeatures,
) -> dict[str, Any]:
    return {
        "stage_goal": request.stage_goal,
        "channel": channel,
        "requested_filters": dict(request.requested_filters),
        "semantic_role_filter_stripped": _semantic_role_stripped(request),
        "recall_top_k": request.recall_top_k,
        "bm25_top_k": request.bm25_top_k,
        "query_features": {
            "clause_ids": list(features.clause_ids),
            "has_exact_signal": features.has_exact_signal,
            "has_standard_ref": features.has_standard_ref,
            "has_exact_term": features.has_exact_term,
            "has_numeric_condition": features.has_numeric_condition,
            "is_short_lookup": features.is_short_lookup,
        },
    }


def _bm25_weight(request: ChannelRetrievalRequest) -> float:
    if request.node_name in KB_RETRIEVAL_NODE_NAMES:
        return BM25_KB_WEIGHT
    return BM25_DIAGNOSIS_WEIGHT


def _semantic_role_stripped(request: ChannelRetrievalRequest) -> bool:
    return "semantic_role" in request.requested_filters and "semantic_role" not in request.filters
