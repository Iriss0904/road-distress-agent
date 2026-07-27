"""Build a directional reference lookup table for retrieve-time expansion.

When the RAG layer retrieves a chunk identified by ``{doc_id}:{clause_id}``,
the table answers "which other clauses does this chunk's body text point at?"
so the retrieve stage can pull those companion chunks into the LLM context.

The table is intentionally one-way: ``source_key -> [target_key, ...]``.
External references (CJJ / GB standards, ``《...》`` titles) are not in the
local corpus, so they are excluded; only resolvable in-corpus targets land
in the table.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

CHUNKS_FILE = "chunks.jsonl"
KEY_SCHEME = "{doc_id}:{clause_id}"
SCHEMA_VERSION = "1"


def build_reference_table(corpus_root: str | Path) -> dict[str, Any]:
    root = Path(corpus_root)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus root not found: {root}")
    chunk_key_lookup = _build_chunk_key_lookup(root)
    references: dict[str, list[str]] = {}
    for doc_id, chunks_path in _iter_doc_chunks(root):
        _accumulate_doc_references(doc_id, chunks_path, chunk_key_lookup, references)
    return {
        "version": SCHEMA_VERSION,
        "key_scheme": KEY_SCHEME,
        "corpus_root": str(root),
        "references": dict(sorted(references.items())),
        "stats": {
            "source_clause_count": len(references),
            "total_target_count": sum(len(v) for v in references.values()),
        },
    }


def write_reference_table(table: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _iter_doc_chunks(root: Path) -> Iterable[tuple[str, Path]]:
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        chunks_path = child / CHUNKS_FILE
        if chunks_path.is_file():
            yield child.name, chunks_path


def _build_chunk_key_lookup(root: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for doc_id, chunks_path in _iter_doc_chunks(root):
        for chunk in _iter_chunks(chunks_path):
            clause_id = chunk.get("canonical_clause_id")
            if not clause_id:
                continue
            lookup[chunk["chunk_id"]] = KEY_SCHEME.format(doc_id=doc_id, clause_id=clause_id)
    return lookup


def _accumulate_doc_references(
    doc_id: str,
    chunks_path: Path,
    chunk_key_lookup: dict[str, str],
    references: dict[str, list[str]],
) -> None:
    for chunk in _iter_chunks(chunks_path):
        targets = _resolve_targets(chunk.get("resolved_cross_refs") or [], chunk_key_lookup)
        if not targets:
            continue
        source_key = _chunk_source_key(doc_id, chunk)
        if source_key is None:
            continue
        bucket = references.setdefault(source_key, [])
        for target in targets:
            if target == source_key or target in bucket:
                continue
            bucket.append(target)


def _resolve_targets(
    resolved_refs: list[dict[str, Any]],
    chunk_key_lookup: dict[str, str],
) -> list[str]:
    targets: list[str] = []
    for ref in resolved_refs:
        if not ref.get("resolved_within_doc"):
            continue
        for chunk_id in ref.get("target_chunk_ids") or []:
            key = chunk_key_lookup.get(chunk_id)
            if key and key not in targets:
                targets.append(key)
    return targets


def _chunk_source_key(doc_id: str, chunk: dict[str, Any]) -> str | None:
    clause_id = chunk.get("canonical_clause_id")
    if not clause_id:
        return None
    return KEY_SCHEME.format(doc_id=doc_id, clause_id=clause_id)


def _iter_chunks(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if text:
                yield json.loads(text)


__all__ = ["build_reference_table", "write_reference_table", "KEY_SCHEME", "SCHEMA_VERSION"]
