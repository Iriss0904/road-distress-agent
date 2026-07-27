"""I/O helpers for contextual ingestion."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from road_distress_agent.ingestion.jsonl import read_jsonl, write_jsonl


def sort_rag_chunks(path: Path, chunk_order: dict[str, int]) -> None:
    if not path.exists():
        return
    rows = read_jsonl(path)
    rows.sort(key=lambda row: chunk_order.get(row["chunk_id"], len(chunk_order)))
    write_jsonl(path, rows)


def stage_paths(doc_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        doc_dir / "chunks_with_rawtext.jsonl",
        doc_dir / "heading_tree.txt",
        doc_dir / "rag_chunks.jsonl",
        doc_dir / "rag_chunks_errors.jsonl",
    )


def done_ids(dst: Path) -> set[str]:
    done = (
        {row["chunk_id"] for row in read_jsonl(dst) if not row.get("_error")}
        if dst.exists()
        else set()
    )
    if done:
        print(f"Resuming: {len(done)} chunks already processed, skipping.")
    return done


def pending_chunks(chunks: list[dict], done: set[str], limit: int | None) -> list[dict]:
    pending = [chunk for chunk in chunks if chunk["chunk_id"] not in done]
    return pending[:limit] if limit else pending


def group_totals(chunks: list[dict]) -> dict[str, int]:
    return Counter(
        chunk["sequential_group_id"] for chunk in chunks if chunk.get("sequential_group_id")
    )
