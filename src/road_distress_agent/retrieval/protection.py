"""Helpers for preserving exact clause hits through reranking."""

from __future__ import annotations

from collections.abc import Sequence

from road_distress_agent.state import RetrievedChunk


def merge_protected_chunks(
    primary: Sequence[RetrievedChunk],
    candidates: Sequence[RetrievedChunk],
    limit: int,
    protected_limit: int,
) -> list[RetrievedChunk]:
    protected = [chunk for chunk in candidates if is_protected_exact_hit(chunk)]
    return _dedupe([*protected[:protected_limit], *primary])[:limit]


def is_protected_exact_hit(chunk: RetrievedChunk) -> bool:
    return bool((chunk.annotations or {}).get("protected_exact_hit"))


def _dedupe(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    unique: list[RetrievedChunk] = []
    for chunk in chunks:
        key = _chunk_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def _chunk_key(chunk: RetrievedChunk) -> str:
    if chunk.chunk_id or chunk.citation_id:
        return str(chunk.chunk_id or chunk.citation_id)
    return f"{chunk.source_doc}:{chunk.clause_id}:{chunk.rank}"
