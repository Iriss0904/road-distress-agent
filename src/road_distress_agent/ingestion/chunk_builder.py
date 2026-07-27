"""Build structure-aware chunks from parsed headings and blocks."""

from __future__ import annotations

from typing import Any

from road_distress_agent.ingestion.chunk_artifacts import (
    block_artifacts,
    block_page_span,
    line_page_span,
)
from road_distress_agent.ingestion.chunk_context import (
    chunk_headings,
    prior_non_table_lines,
    table_chunk_text,
    table_heading_path,
    table_local_context,
    table_raw_captions,
)
from road_distress_agent.ingestion.content_window import content_lines
from road_distress_agent.ingestion.ids import stable_id
from road_distress_agent.ingestion.models import (
    HeadingCandidate,
    HeadingTopology,
    ParsedBlock,
    ParsedDocument,
    ParsedLine,
    RawIngestionChunk,
)
from road_distress_agent.ingestion.national_standard import NATIONAL_STANDARD_PROFILE
from road_distress_agent.ingestion.table_helpers import (
    TABLE_LAYOUT,
    is_continued_caption,
    is_table_caption,
    primary_caption,
    table_clause_id,
    table_label,
)


def build_heading_chunks(
    document: ParsedDocument,
    topology: HeadingTopology,
) -> list[RawIngestionChunk]:
    if not topology.headings:
        return [_whole_document_chunk(document)]

    headings = chunk_headings(sorted(topology.headings, key=lambda item: item.order_key))
    # Use the same body window as heading detection so the last heading does not
    # absorb back-matter text (用词说明 / 条文说明 …) that was excluded upstream.
    lines = content_lines(document.lines)
    chunks: list[RawIngestionChunk] = []
    for index, heading in enumerate(headings):
        next_heading = headings[index + 1] if index + 1 < len(headings) else None
        body_lines = _lines_for_heading(lines, heading, next_heading)
        chunks.extend(_chunks_for_heading(document, topology, heading, body_lines))
    return chunks


def _whole_document_chunk(document: ParsedDocument) -> RawIngestionChunk:
    raw_text = "\n".join(line.text for line in document.lines)
    return RawIngestionChunk(
        chunk_id=f"{document.source_doc_id}:whole:{stable_id(document.source_path)}",
        source_doc_id=document.source_doc_id,
        source_path=document.source_path,
        source_pages=line_page_span(document.lines),
        text=raw_text,
        raw_text=raw_text,
        heading_path=[document.source_name],
        **block_artifacts(document, document.lines),
    )


def _heading_chunk(
    document: ParsedDocument,
    heading: HeadingCandidate,
    body_lines: list[ParsedLine],
) -> RawIngestionChunk | None:
    raw_text = "\n".join(line.text for line in body_lines).strip()
    if not raw_text:
        return None
    return RawIngestionChunk(
        chunk_id=_heading_chunk_id(document, heading, raw_text),
        source_doc_id=document.source_doc_id,
        source_path=document.source_path,
        source_pages=line_page_span(body_lines),
        text=raw_text,
        raw_text=raw_text,
        heading_path=heading.heading_path,
        raw_clause_id=heading.raw_clause_id,
        canonical_clause_id=heading.canonical_clause_id,
        clause_aliases=heading.clause_aliases,
        metadata=_heading_metadata(heading),
        **block_artifacts(document, body_lines),
    )


def _chunks_for_heading(
    document: ParsedDocument,
    topology: HeadingTopology,
    heading: HeadingCandidate,
    body_lines: list[ParsedLine],
) -> list[RawIngestionChunk]:
    chunks: list[RawIngestionChunk] = []
    text_chunk = _heading_chunk(document, heading, _non_table_lines(body_lines))
    if text_chunk:
        chunks.append(text_chunk)
    for blocks in _table_groups_for_heading(document, body_lines):
        chunks.append(_table_chunk(document, topology, heading, blocks, body_lines))
    return chunks


def _table_chunk(
    document: ParsedDocument,
    topology: HeadingTopology,
    heading: HeadingCandidate,
    blocks: list[ParsedBlock],
    body_lines: list[ParsedLine],
) -> RawIngestionChunk:
    captions = [_table_caption(block, body_lines) for block in blocks]
    caption = primary_caption(captions) or f"表格 {blocks[0].block_id}"
    local_context = table_local_context(blocks, body_lines)
    fields = _table_clause_fields(topology, heading, caption)
    table_raw = [
        _table_raw_with_caption(block, block_caption)
        for block, block_caption in zip(blocks, table_raw_captions(captions, caption), strict=True)
    ]
    text = table_chunk_text(caption, local_context)
    heading_path = table_heading_path(heading.heading_path, caption, local_context)
    return RawIngestionChunk(
        chunk_id=_table_chunk_id(document, heading, blocks, caption),
        source_doc_id=document.source_doc_id,
        source_path=document.source_path,
        source_pages=block_page_span(blocks),
        text=text,
        raw_text=text,
        heading_path=heading_path,
        raw_clause_id=fields["raw_clause_id"],
        canonical_clause_id=fields["canonical_clause_id"],
        clause_aliases=fields["clause_aliases"],
        table_raw=table_raw,
        table_paths=[block.table_path for block in blocks if block.table_path],
        image_paths=[path for block in blocks for path in block.image_paths],
        metadata=_table_metadata(heading, fields, heading_path, caption, blocks, local_context),
    )


def _heading_chunk_id(document: ParsedDocument, heading: HeadingCandidate, raw_text: str) -> str:
    return (
        f"{document.source_doc_id}:clause:{heading.canonical_clause_id}:"
        f"{stable_id(heading.line_id, raw_text)}"
    )


def _table_chunk_id(
    document: ParsedDocument,
    heading: HeadingCandidate,
    blocks: list[ParsedBlock],
    caption: str,
) -> str:
    block_ids = [block.block_id for block in blocks]
    table_paths = [block.table_path or "" for block in blocks]
    return (
        f"{document.source_doc_id}:table:{heading.canonical_clause_id}:"
        f"{stable_id(*block_ids, caption, *table_paths)}"
    )


def _heading_metadata(heading: HeadingCandidate) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "raw_clause_id": heading.raw_clause_id,
        "canonical_clause_id": heading.canonical_clause_id,
        "clause_aliases": heading.clause_aliases,
        "heading_path": heading.heading_path,
    }
    if heading.anomaly_type:
        metadata["numbering_anomaly"] = {
            "type": heading.anomaly_type,
            "reason": heading.anomaly_reason,
        }
    return metadata


def _table_clause_fields(
    topology: HeadingTopology,
    heading: HeadingCandidate,
    caption: str,
) -> dict[str, Any]:
    table_id = table_clause_id(caption)
    if topology.profile != NATIONAL_STANDARD_PROFILE or not table_id:
        return {
            "raw_clause_id": heading.raw_clause_id,
            "canonical_clause_id": heading.canonical_clause_id,
            "clause_aliases": heading.clause_aliases,
        }
    aliases = [table_id]
    label = table_label(caption)
    if label:
        aliases.append(label)
    return {
        "raw_clause_id": table_id,
        "canonical_clause_id": table_id,
        "clause_aliases": aliases,
    }


def _table_metadata(
    heading: HeadingCandidate,
    fields: dict[str, Any],
    heading_path: list[str],
    caption: str,
    blocks: list[ParsedBlock],
    local_context: str | None,
) -> dict[str, Any]:
    metadata = {
        **_heading_metadata(heading),
        "raw_clause_id": fields["raw_clause_id"],
        "canonical_clause_id": fields["canonical_clause_id"],
        "clause_aliases": fields["clause_aliases"],
    }
    metadata.update(
        {
            "chunk_kind": "table",
            "parent_clause_id": heading.canonical_clause_id,
            "table_caption": caption,
            "table_label": table_label(caption),
            "table_path": blocks[0].table_path,
            "heading_path": heading_path,
        }
    )
    if local_context:
        metadata["local_context_title"] = local_context
    return metadata


def _lines_for_heading(
    lines: list[ParsedLine],
    heading: HeadingCandidate,
    next_heading: HeadingCandidate | None,
) -> list[ParsedLine]:
    selected: list[ParsedLine] = []
    for line in lines:
        if line.order_key < heading.order_key:
            continue
        if next_heading and line.order_key >= next_heading.order_key:
            break
        selected.append(line)
    return selected


def _non_table_lines(lines: list[ParsedLine]) -> list[ParsedLine]:
    return [
        line
        for line in lines
        if line.layout_type != TABLE_LAYOUT and not is_table_caption(line.normalized_text())
    ]


def _table_blocks_for_heading(
    document: ParsedDocument,
    body_lines: list[ParsedLine],
) -> list[ParsedBlock]:
    block_ids = {line.block_id for line in body_lines if line.layout_type == TABLE_LAYOUT}
    blocks = [block for block in document.blocks if block.block_id in block_ids]
    return sorted(blocks, key=lambda block: (block.page_number, block.order))


def _table_groups_for_heading(
    document: ParsedDocument,
    body_lines: list[ParsedLine],
) -> list[list[ParsedBlock]]:
    groups: list[list[ParsedBlock]] = []
    current: list[ParsedBlock] = []
    current_label: str | None = None
    for block in _table_blocks_for_heading(document, body_lines):
        caption = _table_caption(block, body_lines)
        label = table_label(caption) or block.block_id
        if is_continued_caption(caption):
            if current_label != label:
                raise ValueError(f"continued table without prior table: {caption}")
            current.append(block)
            continue
        if current:
            groups.append(current)
        current = [block]
        current_label = label
    if current:
        groups.append(current)
    return groups


def _table_caption(block: ParsedBlock, body_lines: list[ParsedLine]) -> str:
    raw = block.table_raw if isinstance(block.table_raw, dict) else {}
    caption = str(raw.get("caption") or "").strip()
    return caption or _inferred_table_caption(block, body_lines)


def _table_raw_with_caption(block: ParsedBlock, caption: str) -> dict[str, Any]:
    if not isinstance(block.table_raw, dict):
        raise ValueError(f"table block missing table_raw: {block.block_id}")
    table_raw = dict(block.table_raw)
    table_raw["caption"] = caption
    return table_raw


def _inferred_table_caption(block: ParsedBlock, body_lines: list[ParsedLine]) -> str:
    for line in reversed(prior_non_table_lines(block, body_lines)):
        text = line.normalized_text()
        if is_table_caption(text):
            return text
    return ""
