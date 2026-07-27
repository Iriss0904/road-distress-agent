"""End-to-end ingestion runner: PDF -> chunks.jsonl + pass0 + artifacts."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from road_distress_agent.ingestion.pipeline import (
    build_ingestion_middle_layer,
    write_middle_layer_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path, help="Input PDF path")
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output root directory (one subdir per source_doc_id)",
    )
    parser.add_argument(
        "--mineru-workdir",
        type=Path,
        default=None,
        help="Where MinerU writes its raw outputs. Defaults to <out>/_mineru_workdir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = args.mineru_workdir.resolve() if args.mineru_workdir else out_dir / "_mineru_workdir"

    started = time.perf_counter()
    layer = build_ingestion_middle_layer(
        args.pdf,
        artifact_root=out_dir,
        mineru_workdir=workdir,
    )
    paths = write_middle_layer_outputs(layer, out_dir)
    elapsed = time.perf_counter() - started

    print(f"[ingestion] {layer.parsed_document.source_doc_id} done in {elapsed:.1f}s")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
