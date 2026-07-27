"""Pass 0 deterministic indexing for contextual raw-standard chunks."""

from __future__ import annotations

import re

from road_distress_agent.ingestion.chunk_builder import build_heading_chunks
from road_distress_agent.ingestion.models import (
    HeadingTopology,
    ParsedDocument,
    Pass0Artifacts,
    RawIngestionChunk,
)
from road_distress_agent.ingestion.sidecar_index import (
    annotate_cross_refs,
    build_chunk_index,
    build_clause_index,
    build_sequential_group_index,
    chunks_as_dicts,
)
from road_distress_agent.ingestion.step_chunk_builder import build_step_chunks

__all__ = ["build_pass0_artifacts", "chunks_as_dicts"]

SOURCE_PAGE_RE = re.compile(r"\d+")


def build_pass0_artifacts(
    document: ParsedDocument,
    topology: HeadingTopology,
) -> tuple[list[RawIngestionChunk], Pass0Artifacts]:
    heading_chunks = build_heading_chunks(document, topology)
    chunks = _source_ordered_chunks(
        [
            *heading_chunks,
            *build_step_chunks(heading_chunks),
        ]
    )
    clause_index = build_clause_index(chunks)
    cross_refs = annotate_cross_refs(chunks, clause_index)
    pass0 = Pass0Artifacts(
        source_doc_id=document.source_doc_id,
        clause_index=clause_index,
        chunk_index=build_chunk_index(chunks),
        cross_refs=cross_refs,
        sequential_groups=build_sequential_group_index(chunks),
        anomalies=topology.anomalies,
    )
    return chunks, pass0


def _source_ordered_chunks(chunks: list[RawIngestionChunk]) -> list[RawIngestionChunk]:
    return sorted(chunks, key=_source_page_start)


def _source_page_start(chunk: RawIngestionChunk) -> int:
    match = SOURCE_PAGE_RE.search(chunk.source_pages)
    if not match:
        raise ValueError(f"chunk missing source page: {chunk.chunk_id}")
    return int(match.group())
