#!/usr/bin/env python3
"""Stage 2: LLM contextual retrieval for cleaned raw-standard chunks."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from road_distress_agent.ingestion.contextual_pass import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("doc_dir", type=Path)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N chunks")
    args = parser.parse_args()
    asyncio.run(run(args.doc_dir, args.concurrency, args.limit))


if __name__ == "__main__":
    main()
