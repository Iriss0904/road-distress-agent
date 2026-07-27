"""Orchestration helpers for the ingestion middle layer."""

from __future__ import annotations

from pathlib import Path

from road_distress_agent.ingestion.heading_output import heading_tree_to_text, headings_as_dicts
from road_distress_agent.ingestion.heading_topology import build_heading_topology
from road_distress_agent.ingestion.jsonl import write_json, write_jsonl
from road_distress_agent.ingestion.mineru_adapter import MinerUAdapter, source_doc_id_from_path
from road_distress_agent.ingestion.models import IngestionMiddleLayer, dataclass_to_dict
from road_distress_agent.ingestion.pass0 import build_pass0_artifacts, chunks_as_dicts


def build_ingestion_middle_layer(
    pdf_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    mineru_workdir: str | Path | None = None,
    parse_method: str = "auto",
) -> IngestionMiddleLayer:
    path = Path(pdf_path)
    doc_id = source_doc_id_from_path(path)
    adapter = MinerUAdapter(parse_method=parse_method)
    parsed = adapter.parse_pdf(
        path,
        source_doc_id=doc_id,
        artifact_dir=_artifact_dir(artifact_root, doc_id),
        mineru_workdir=mineru_workdir,
    )
    topology = build_heading_topology(parsed)
    chunks, pass0 = build_pass0_artifacts(parsed, topology)
    return IngestionMiddleLayer(
        parsed_document=parsed,
        topology=topology,
        chunks=chunks,
        pass0=pass0,
    )


def write_middle_layer_outputs(
    layer: IngestionMiddleLayer,
    output_dir: str | Path,
) -> dict[str, Path]:
    out = Path(output_dir)
    parsed = layer.parsed_document
    doc_dir = out / parsed.source_doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "headings": doc_dir / "headings.jsonl",
        "heading_tree": doc_dir / "heading_tree.txt",
        "chunks": doc_dir / "chunks.jsonl",
        "pass0": doc_dir / "pass0_sidecar_indexes.json",
        "summary": doc_dir / "summary.json",
    }

    write_jsonl(paths["headings"], headings_as_dicts(layer.topology.headings))
    paths["heading_tree"].write_text(
        heading_tree_to_text(layer.topology.roots) + "\n",
        encoding="utf-8",
    )
    write_jsonl(paths["chunks"], chunks_as_dicts(layer.chunks))
    write_json(paths["pass0"], dataclass_to_dict(layer.pass0))
    write_json(paths["summary"], _summary(layer))
    return paths


def _artifact_dir(artifact_root: str | Path | None, doc_id: str) -> Path | None:
    if artifact_root is None:
        return None
    return Path(artifact_root) / doc_id / "artifacts"


def _summary(layer: IngestionMiddleLayer) -> dict[str, object]:
    parsed = layer.parsed_document
    return {
        "source_doc_id": parsed.source_doc_id,
        "source_path": parsed.source_path,
        "parser_backend": parsed.parser_backend,
        "page_count": parsed.page_count,
        "block_count": len(parsed.blocks),
        "line_count": len(parsed.lines),
        "heading_count": len(layer.topology.headings),
        "chunk_count": len(layer.chunks),
        "table_block_count": sum(1 for block in parsed.blocks if block.table_path),
        "inline_image_count": sum(len(block.image_paths) for block in parsed.blocks),
        "anomaly_count": len(layer.topology.anomalies),
        "sequential_group_count": len(layer.pass0.sequential_groups),
        "cross_ref_chunk_count": len(layer.pass0.cross_refs),
        "warnings": parsed.warnings,
    }
