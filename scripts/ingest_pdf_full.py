#!/usr/bin/env python3
"""One-click PDF ingestion: PDF → Qdrant knowledge base.

Runs 4 stages sequentially:
  Stage 0  MinerU parse            PDF → chunks.jsonl + heading_tree.txt
  Stage 1  rawtext clean           → chunks_with_rawtext.jsonl
  Stage 2  LLM contextual          → rag_chunks.jsonl  (DeepSeek API, with retry)
  Stage 3  BGE-M3 + Qdrant upsert → points added to collection (no data loss)

Re-running is safe:
  Stages 0 and 1 are skipped if output already exists.
  Stage 2 resumes from where it left off.
  Stage 3 upserts (never recreates the collection).

Usage:
    python scripts/ingest_pdf_full.py --pdf path/to/doc.pdf
    python scripts/ingest_pdf_full.py --pdf path/to/doc.pdf \\
        --out data/processed/ --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

MAX_STAGE2_RETRIES = 3


# Mirrors source_doc_id_from_path in mineru_adapter.py — keep in sync.
def _doc_id(pdf: Path) -> str:
    stem = pdf.stem.lower()
    stem = re.sub(r"[^0-9a-zA-Z一-鿿]+", "_", stem).strip("_")
    return stem or "raw_standard"


def _run(label: str, cmd: list[str]) -> bool:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"\n[FAIL] {label} (exit {result.returncode}, {elapsed:.1f}s)")
        return False
    print(f"\n[OK]   {label} ({elapsed:.1f}s)")
    return True


def _processed_count(doc_dir: Path) -> tuple[int, int]:
    """Return (total_chunks, successfully_processed)."""
    src = doc_dir / "chunks_with_rawtext.jsonl"
    dst = doc_dir / "rag_chunks.jsonl"
    if not src.exists():
        return 0, 0
    total = sum(1 for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip())
    if not dst.exists():
        return total, 0
    ok = sum(
        1
        for ln in dst.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not json.loads(ln).get("_error")
    )
    return total, ok


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path, help="Input PDF path")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/"),
        help="Output root (default: data/processed/)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Initial parallel LLM calls for Stage 2 (default: 8). "
        "Halved automatically on rate-limit errors.",
    )
    return parser.parse_args()


def _require_pdf(pdf: Path) -> None:
    if not pdf.exists():
        print(f"ERROR: PDF not found: {pdf}", file=sys.stderr)
        sys.exit(1)


def _print_context(pdf: Path, doc_id: str, doc_dir: Path) -> None:
    print(f"\nPDF            : {pdf}")
    print(f"source_doc_id  : {doc_id}")
    print(f"Output dir     : {doc_dir}")


def _stage0(pdf: Path, out: Path, doc_dir: Path) -> None:
    chunks_path = doc_dir / "chunks.jsonl"
    if chunks_path.exists():
        print(f"\n[SKIP] Stage 0 — {chunks_path.name} already exists")
        return
    cmd = [
        sys.executable,
        "scripts/run_ingestion.py",
        "--pdf",
        str(pdf),
        "--out",
        str(out),
    ]
    if not _run("Stage 0 / MinerU parse", cmd):
        sys.exit(1)


def _stage1(doc_dir: Path) -> None:
    rawtext_path = doc_dir / "chunks_with_rawtext.jsonl"
    if rawtext_path.exists():
        print(f"[SKIP] Stage 1 — {rawtext_path.name} already exists")
        return
    command = [sys.executable, "scripts/clean_rawtext.py", str(doc_dir)]
    if not _run("Stage 1 / rawtext clean", command):
        sys.exit(1)


def _stage2(doc_dir: Path, initial_concurrency: int) -> None:
    concurrency = initial_concurrency
    for attempt in range(MAX_STAGE2_RETRIES + 1):
        if not _run(_stage2_label(concurrency, attempt), _stage2_cmd(doc_dir, concurrency)):
            sys.exit(1)
        total, processed = _processed_count(doc_dir)
        remaining = total - processed
        if remaining == 0:
            print(f"  All {total} chunks processed successfully.")
            return
        if attempt < MAX_STAGE2_RETRIES:
            concurrency = _retry_concurrency(concurrency, remaining, total)
    _fail_incomplete_stage2(doc_dir, remaining, total)


def _stage2_label(concurrency: int, attempt: int) -> str:
    label = f"Stage 2 / LLM contextual (concurrency={concurrency})"
    if attempt > 0:
        label += f"  [retry {attempt}/{MAX_STAGE2_RETRIES}]"
    return label


def _stage2_cmd(doc_dir: Path, concurrency: int) -> list[str]:
    return [
        sys.executable,
        "scripts/pass1_contextual.py",
        str(doc_dir),
        "--concurrency",
        str(concurrency),
    ]


def _retry_concurrency(concurrency: int, remaining: int, total: int) -> int:
    next_concurrency = max(1, concurrency // 2)
    print(
        f"  {remaining}/{total} chunks still pending "
        f"(transient API or structured-output error). "
        f"Retrying with concurrency={next_concurrency}…"
    )
    return next_concurrency


def _fail_incomplete_stage2(doc_dir: Path, remaining: int, total: int) -> None:
    print(
        f"\nERROR: {remaining}/{total} chunks could not be processed "
        f"after {MAX_STAGE2_RETRIES} retries.",
        file=sys.stderr,
    )
    print(f"  Details: {doc_dir / 'rag_chunks_errors.jsonl'}", file=sys.stderr)
    print("  Stage 3 skipped because rag_chunks.jsonl is incomplete.", file=sys.stderr)
    sys.exit(1)


def _require_complete_stage2(doc_dir: Path) -> None:
    total, processed = _processed_count(doc_dir)
    if processed != total:
        print(
            f"ERROR: Stage 2 incomplete ({processed}/{total}); Stage 3 skipped.",
            file=sys.stderr,
        )
        sys.exit(1)


def _stage3(doc_dir: Path) -> None:
    if not _run(
        "Stage 3 / BGE-M3 encode + Qdrant upsert",
        [sys.executable, "scripts/build_qdrant_index.py", str(doc_dir)],
    ):
        sys.exit(1)


def _print_done(doc_id: str, doc_dir: Path, elapsed: float) -> None:
    print(f"\n{'═' * 60}")
    print(f"  Ingestion complete for '{doc_id}'")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Output:     {doc_dir}")
    print(f"{'═' * 60}\n")


def main() -> None:
    args = _parse_args()
    pdf = args.pdf.resolve()
    _require_pdf(pdf)
    out = args.out.resolve()
    doc_id = _doc_id(pdf)
    doc_dir = out / doc_id
    _print_context(pdf, doc_id, doc_dir)
    wall_start = time.perf_counter()
    _stage0(pdf, out, doc_dir)
    _stage1(doc_dir)
    _stage2(doc_dir, args.concurrency)
    _require_complete_stage2(doc_dir)
    _stage3(doc_dir)
    _print_done(doc_id, doc_dir, time.perf_counter() - wall_start)


if __name__ == "__main__":
    main()
