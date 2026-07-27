#!/usr/bin/env python3
"""Stage 1: Add rawtext + chunk_type to chunks.jsonl → chunks_with_rawtext.jsonl.

Usage:
    python scripts/clean_rawtext.py <doc_dir>

Example:
    python scripts/clean_rawtext.py \
        data/processed/my_document
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from road_distress_agent.ingestion.rawtext_cleaner import build_rawtext, classify_chunk


def main(doc_dir: Path) -> None:
    src = doc_dir / "chunks.jsonl"
    dst = doc_dir / "chunks_with_rawtext.jsonl"

    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        sys.exit(1)

    chunks = [
        json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    print(f"Loaded {len(chunks)} chunks from {src}")

    results = []
    stats: dict[str, int] = {"text": 0, "table": 0, "step": 0}
    for chunk in chunks:
        internal_type = classify_chunk(chunk)  # "step"|"table"|"text" — drives rawtext logic
        rawtext = build_rawtext(chunk)
        out = dict(chunk)
        out["chunk_type"] = internal_type
        out["rawtext"] = rawtext
        results.append(out)
        stats[internal_type] += 1

    dst.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
        encoding="utf-8",
    )
    print(f"Written {len(results)} chunks to {dst}")
    print(f"  text={stats['text']}  table={stats['table']}  step={stats['step']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(Path(sys.argv[1]))
