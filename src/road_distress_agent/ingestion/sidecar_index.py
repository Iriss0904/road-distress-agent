"""Build deterministic sidecar indexes for raw-standard chunks."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict
from typing import Any

from road_distress_agent.ingestion.models import RawIngestionChunk
from road_distress_agent.ingestion.table_helpers import normalise_table_label, table_label

CROSS_REF_RE = re.compile(
    r"(?:参见|详见|见|按|按照|依据|执行|符合)\s*"
    r"(?P<clause>\d+(?:\.\d+){1,5})(?![\d.])"
)
TABLE_REF_RE = re.compile(
    r"表\s*(?P<label>\d+(?:\s*\.\s*\d+){0,5}(?:\s*[－-]\s*\d+)?|"
    r"[A-ZＡ-Ｚ](?:\s*[.-]\s*\d+)?)(?=\s|$|[\u4e00-\u9fff]|[，,、。；;）)])"
)
FULLWIDTH_TABLE = str.maketrans(
    {chr(ord("Ａ") + index): chr(ord("A") + index) for index in range(26)}
)


def build_chunk_index(chunks: list[RawIngestionChunk]) -> dict[str, dict[str, Any]]:
    return {
        chunk.chunk_id: {
            "source_doc_id": chunk.source_doc_id,
            "source_path": chunk.source_path,
            "source_pages": chunk.source_pages,
            "raw_clause_id": chunk.raw_clause_id,
            "canonical_clause_id": chunk.canonical_clause_id,
            "clause_aliases": chunk.clause_aliases,
            "heading_path": chunk.heading_path,
            "table_paths": chunk.table_paths,
            "image_paths": chunk.image_paths,
            "sequential_group_id": chunk.sequential_group_id,
            "step_index": chunk.step_index,
            "step_title": chunk.step_title,
            "parent_chunk_id": chunk.parent_chunk_id,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    }


def build_clause_index(chunks: list[RawIngestionChunk]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        for alias in _clause_aliases(chunk):
            if chunk.chunk_id not in index[alias]:
                index[alias].append(chunk.chunk_id)
    return dict(index)


def build_table_index(chunks: list[RawIngestionChunk]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        for alias in _table_aliases(chunk):
            if chunk.chunk_id not in index[alias]:
                index[alias].append(chunk.chunk_id)
    return dict(index)


def build_clause_table_index(
    chunks: list[RawIngestionChunk],
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    index: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for chunk in chunks:
        if not chunk.table_raw or not chunk.canonical_clause_id:
            continue
        parent_clause_id = chunk.metadata.get("parent_clause_id")
        clause_id = (
            parent_clause_id if isinstance(parent_clause_id, str) else chunk.canonical_clause_id
        )
        key = (chunk.source_doc_id, clause_id)
        for label in _table_aliases(chunk):
            entry = (label, chunk.chunk_id)
            if entry not in index[key]:
                index[key].append(entry)
    return dict(index)


def _clause_aliases(chunk: RawIngestionChunk) -> list[str]:
    aliases = set(chunk.clause_aliases)
    if chunk.raw_clause_id:
        aliases.add(chunk.raw_clause_id)
    if chunk.canonical_clause_id:
        aliases.add(chunk.canonical_clause_id)
    return sorted(alias for alias in aliases if alias)


def annotate_cross_refs(
    chunks: list[RawIngestionChunk],
    clause_index: dict[str, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    table_index = build_table_index(chunks)
    clause_table_index = build_clause_table_index(chunks)
    cross_refs: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        refs = _chunk_cross_refs(chunk, clause_index, table_index, clause_table_index)
        chunk.cross_refs = [item["raw_ref"] for item in refs]
        chunk.resolved_cross_refs = refs
        if refs:
            cross_refs[chunk.chunk_id] = refs
            chunk.metadata["cross_refs"] = chunk.cross_refs
    return cross_refs


def _chunk_cross_refs(
    chunk: RawIngestionChunk,
    clause_index: dict[str, list[str]],
    table_index: dict[str, list[str]],
    clause_table_index: dict[tuple[str, str], list[tuple[str, str]]],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in CROSS_REF_RE.finditer(chunk.raw_text):
        raw_ref = match.group("clause")
        if raw_ref in seen:
            continue
        seen.add(raw_ref)
        target_ids = _without_self(clause_index.get(raw_ref, []), chunk.chunk_id)
        refs.append(_cross_ref_payload(raw_ref, target_ids, "clause"))
    for match in TABLE_REF_RE.finditer(chunk.raw_text):
        raw_ref = _table_ref(match.group("label"))
        if raw_ref in seen:
            continue
        seen.add(raw_ref)
        target_ids = _without_self(table_index.get(raw_ref, []), chunk.chunk_id)
        if not target_ids and table_index.get(raw_ref):
            continue
        refs.append(_cross_ref_payload(raw_ref, target_ids, "table"))
    for raw_ref, target_id in _same_clause_table_refs(chunk, clause_table_index):
        if raw_ref in seen:
            continue
        seen.add(raw_ref)
        refs.append(_cross_ref_payload(raw_ref, [target_id], "table"))
    return refs


def _cross_ref_payload(raw_ref: str, target_ids: list[str], ref_type: str) -> dict[str, Any]:
    return {
        "raw_ref": raw_ref,
        "ref_type": ref_type,
        "resolved_within_doc": bool(target_ids),
        "target_chunk_ids": target_ids,
    }


def _without_self(target_ids: list[str], chunk_id: str) -> list[str]:
    return [target_id for target_id in target_ids if target_id != chunk_id]


def _same_clause_table_refs(
    chunk: RawIngestionChunk,
    clause_table_index: dict[tuple[str, str], list[tuple[str, str]]],
) -> list[tuple[str, str]]:
    if chunk.table_raw or not chunk.canonical_clause_id:
        return []
    key = (chunk.source_doc_id, chunk.canonical_clause_id)
    return clause_table_index.get(key, [])


def _table_aliases(chunk: RawIngestionChunk) -> list[str]:
    if not chunk.table_raw:
        return []
    labels: set[str] = set()
    label = chunk.metadata.get("table_label")
    if isinstance(label, str) and label:
        labels.add(_normalize_table_ref(label))
    for entry in chunk.table_raw if isinstance(chunk.table_raw, list) else [chunk.table_raw]:
        if isinstance(entry, dict):
            labels.update(_labels_from_caption(str(entry.get("caption") or "")))
    return sorted(labels)


def _labels_from_caption(caption: str) -> set[str]:
    labels: set[str] = set()
    parsed_label = table_label(caption)
    if parsed_label:
        labels.add(_normalize_table_ref(parsed_label))
    for match in TABLE_REF_RE.finditer(caption):
        labels.add(_table_ref(match.group("label")))
    return labels


def _table_ref(label: str) -> str:
    return f"表{normalise_table_label(label)}"


def _normalize_table_ref(value: str) -> str:
    compact = re.sub(r"\s+", "", value.translate(FULLWIDTH_TABLE))
    compact = compact.replace("．", ".").replace("－", "-")
    if compact.startswith("表"):
        return f"表{normalise_table_label(compact.removeprefix('表'))}"
    return normalise_table_label(compact)


def build_sequential_group_index(chunks: list[RawIngestionChunk]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[RawIngestionChunk]] = defaultdict(list)
    for chunk in chunks:
        if chunk.sequential_group_id:
            groups[chunk.sequential_group_id].append(chunk)
    return {
        group_id: _sequential_group_payload(group_chunks)
        for group_id, group_chunks in groups.items()
    }


def _sequential_group_payload(group_chunks: list[RawIngestionChunk]) -> dict[str, Any]:
    sorted_chunks = sorted(group_chunks, key=lambda item: item.step_index or 0)
    return {
        "source_doc_id": sorted_chunks[0].source_doc_id,
        "parent_chunk_id": sorted_chunks[0].parent_chunk_id,
        "chunk_ids": [chunk.chunk_id for chunk in sorted_chunks],
        "steps": [_step_payload(chunk) for chunk in sorted_chunks],
    }


def _step_payload(chunk: RawIngestionChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "step_index": chunk.step_index,
        "raw_step_index": chunk.metadata.get("raw_step_index"),
        "step_title": chunk.step_title,
        "source_pages": chunk.source_pages,
        "heading_path": chunk.heading_path,
    }


def chunks_as_dicts(chunks: list[RawIngestionChunk]) -> list[dict[str, Any]]:
    return [asdict(chunk) for chunk in chunks]
