"""Expand resolved cross-reference targets already stored in chunk payloads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from road_distress_agent.retrieval.evidence_options import EvidenceRetrievalOptions
from road_distress_agent.state import AuditEvent, RetrievalAttempt, RetrievedChunk
from road_distress_agent.tracing import retrieval_trace_event

CROSS_REF_CHANNEL = "cross_ref"
MAX_SOURCE_NOTES = 2


class ChunkLookupTool(Protocol):
    def fetch_by_chunk_ids(
        self,
        chunk_ids: Sequence[str],
        limit: int,
    ) -> Sequence[RetrievedChunk]: ...


@dataclass(frozen=True)
class CrossRefLink:
    target_id: str
    raw_ref: str
    source_chunk_id: str | None
    source_clause_id: str | None
    source_doc: str | None
    source_context: str | None


@dataclass(frozen=True)
class CrossRefExpansion:
    chunks: list[RetrievedChunk]
    attempts: list[RetrievalAttempt]
    traces: list[AuditEvent]


def expand_cross_references(
    options: EvidenceRetrievalOptions,
    chunks: list[RetrievedChunk],
    lookup_tool: ChunkLookupTool | None,
) -> CrossRefExpansion:
    links = _links_by_target(chunks)
    if not links:
        return CrossRefExpansion(chunks, [], [])
    annotated = [_annotate_existing(chunk, links) for chunk in chunks]
    missing_ids = _missing_target_ids(links, annotated)
    if not missing_ids:
        return CrossRefExpansion(annotated, [], [])
    if lookup_tool is None:
        raise ValueError("Resolved cross-reference targets require a chunk lookup tool.")
    fetched_chunks = lookup_tool.fetch_by_chunk_ids(missing_ids, len(missing_ids))
    fetched = _annotate_fetched(fetched_chunks, links)
    return CrossRefExpansion(
        [*annotated, *fetched],
        [_attempt(options, lookup_tool, missing_ids, len(fetched))],
        [_trace(options, lookup_tool, missing_ids, fetched)],
    )


def _links_by_target(chunks: Sequence[RetrievedChunk]) -> dict[str, list[CrossRefLink]]:
    links: dict[str, list[CrossRefLink]] = {}
    for chunk in chunks:
        for link in _chunk_links(chunk):
            links.setdefault(link.target_id, []).append(link)
    return links


def _chunk_links(chunk: RetrievedChunk) -> list[CrossRefLink]:
    refs = (chunk.annotations or {}).get("resolved_cross_refs") or []
    links: list[CrossRefLink] = []
    for ref in refs:
        if not isinstance(ref, dict) or not ref.get("resolved_within_doc"):
            continue
        for target_id in ref.get("target_chunk_ids") or []:
            links.append(_link(chunk, str(target_id), ref))
    return links


def _link(chunk: RetrievedChunk, target_id: str, ref: dict[str, Any]) -> CrossRefLink:
    return CrossRefLink(
        target_id=target_id,
        raw_ref=str(ref.get("raw_ref") or ""),
        source_chunk_id=chunk.chunk_id,
        source_clause_id=chunk.clause_id,
        source_doc=chunk.source_doc or chunk.document_id,
        source_context=chunk.context_prefix,
    )


def _annotate_existing(
    chunk: RetrievedChunk,
    links: dict[str, list[CrossRefLink]],
) -> RetrievedChunk:
    if not chunk.chunk_id or chunk.chunk_id not in links:
        return chunk
    return _annotate_chunk(chunk, links[chunk.chunk_id])


def _annotate_fetched(
    chunks: Sequence[RetrievedChunk],
    links: dict[str, list[CrossRefLink]],
) -> list[RetrievedChunk]:
    return [
        _annotate_chunk(chunk, links.get(str(chunk.chunk_id), []))
        for chunk in chunks
        if chunk.chunk_id
    ]


def _annotate_chunk(chunk: RetrievedChunk, links: list[CrossRefLink]) -> RetrievedChunk:
    annotations = {
        **(chunk.annotations or {}),
        "cross_ref_expanded": True,
        "cross_ref_expanded_from": [_link_payload(link) for link in links],
    }
    return chunk.model_copy(
        update={"context_prefix": _expanded_prefix(chunk, links), "annotations": annotations}
    )


def _expanded_prefix(chunk: RetrievedChunk, links: list[CrossRefLink]) -> str:
    notes = "；".join(_source_note(link) for link in links[:MAX_SOURCE_NOTES])
    base = chunk.context_prefix or ""
    return "\n".join(part for part in (notes, base) if part)


def _source_note(link: CrossRefLink) -> str:
    source = " ".join(part for part in (link.source_doc, link.source_clause_id) if part)
    context = f"（{link.source_context}）" if link.source_context else ""
    return f"由{source}参见{link.raw_ref}带入{context}"


def _link_payload(link: CrossRefLink) -> dict[str, Any]:
    return {
        "target_id": link.target_id,
        "raw_ref": link.raw_ref,
        "source_chunk_id": link.source_chunk_id,
        "source_clause_id": link.source_clause_id,
        "source_doc": link.source_doc,
        "source_context": link.source_context,
    }


def _missing_target_ids(
    links: dict[str, list[CrossRefLink]],
    chunks: Sequence[RetrievedChunk],
) -> list[str]:
    existing = {str(chunk.chunk_id) for chunk in chunks if chunk.chunk_id}
    return [target_id for target_id in links if target_id not in existing]


def _attempt(
    options: EvidenceRetrievalOptions,
    tool: ChunkLookupTool,
    target_ids: Sequence[str],
    count: int,
) -> RetrievalAttempt:
    return RetrievalAttempt(
        tool_name=f"{tool.__class__.__name__}.fetch_by_chunk_ids",
        query=options.query,
        filters={"chunk_id": list(target_ids)},
        citation_count=count,
    )


def _trace(
    options: EvidenceRetrievalOptions,
    tool: ChunkLookupTool,
    target_ids: Sequence[str],
    chunks: list[RetrievedChunk],
) -> AuditEvent:
    return retrieval_trace_event(
        node_name=options.node_name,
        title=f"{options.title} ({CROSS_REF_CHANNEL})",
        query=options.query,
        filters={"chunk_id": list(target_ids)},
        chunks=chunks,
        tool=tool,
        metadata={"stage_goal": options.stage_goal, "channel": CROSS_REF_CHANNEL},
    )
