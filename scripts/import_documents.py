#!/usr/bin/env python3
"""Import user-owned Markdown or text documents into index-ready chunks.

Usage:
    python scripts/import_documents.py data/raw --out data/processed

This command intentionally accepts only .md and .txt files. For PDF input,
use ``scripts/ingest_pdf_full.py`` after installing optional ingestion
dependencies; its parsed artifacts are written under the same output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SUPPORTED_SUFFIXES = frozenset({".md", ".txt"})
PARAGRAPH_SEPARATOR = re.compile(r"\n\s*\n+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_dir", type=Path, help="Directory containing user-owned .md/.txt files"
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_dir}")
    files = sorted(
        path for path in source_dir.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise SystemExit("No .md or .txt files found. Add user-owned documents and run again.")
    for path in files:
        write_document_chunks(path, source_dir=source_dir, output_root=args.out.resolve())


def write_document_chunks(path: Path, *, source_dir: Path, output_root: Path) -> Path:
    doc_id = document_id(path.relative_to(source_dir))
    output = output_root / doc_id / "rag_chunks.jsonl"
    rows = chunk_rows(path, doc_id=doc_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    output.write_text(payload, encoding="utf-8")
    print(f"Imported {len(rows)} chunks: {path.name} -> {output}")
    return output


def chunk_rows(path: Path, *, doc_id: str) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    paragraphs = [
        normalize(paragraph)
        for paragraph in PARAGRAPH_SEPARATOR.split(text)
        if normalize(paragraph)
    ]
    if not paragraphs:
        raise ValueError(f"Document has no non-empty text: {path}")
    return [
        chunk_row(doc_id, path.name, position, paragraph)
        for position, paragraph in enumerate(paragraphs, 1)
    ]


def chunk_row(doc_id: str, filename: str, position: int, text: str) -> dict[str, object]:
    chunk_id = hashlib.sha256(f"{doc_id}:{position}:{text}".encode()).hexdigest()[:24]
    return {
        "chunk_id": chunk_id,
        "source_doc_id": doc_id,
        "source_filename": filename,
        "source_pages": None,
        "clause_id": None,
        "heading_path": [],
        "semantic_role": "general_info",
        "rawtext": text,
        "context_prefix": "",
        "cross_refs": [],
        "embed_text": text,
    }


def document_id(path: Path) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_-]+", "_", str(path.with_suffix(""))).strip("_")
    if not normalized:
        raise ValueError(f"Cannot create document id for: {path}")
    return normalized.lower()


def normalize(value: str) -> str:
    return " ".join(value.split())


if __name__ == "__main__":
    main()
