"""BM25 result annotation and boundary error helpers."""

from __future__ import annotations

from road_distress_agent.errors import ErrorCategory, ErrorInfo, make_error_info
from road_distress_agent.state import Citation


def with_bm25_score(chunk: Citation, score: float, rank: int) -> Citation:
    annotations = {**(chunk.annotations or {}), "bm25_score": score}
    return chunk.model_copy(
        update={
            "score": score,
            "similarity": score,
            "rank": rank,
            "annotations": annotations,
        }
    )


def with_clause_score(chunk: Citation, rank: int) -> Citation:
    annotations = {**(chunk.annotations or {}), "clause_lookup": True}
    return chunk.model_copy(
        update={
            "score": 1.0,
            "similarity": 1.0,
            "rank": rank,
            "annotations": annotations,
        }
    )


def empty_index_info(collection: str) -> ErrorInfo:
    return make_error_info(
        domain="BM25",
        step="INDEX",
        category=ErrorCategory.NOT_FOUND,
        responsibility="BM25 检索为空",
        reason=f'集合 "{collection}" 无可索引 payload',
        hint="先运行 build_qdrant_index 建库并确认 payload 已写入。",
        raw=f"collection={collection}",
        retriable=False,
    )
