"""MinerU PDF parsing adapter for raw road-standard ingestion.

Translates MinerU's ``content_list.json`` into the project's
``ParsedDocument`` / ``ParsedBlock`` / ``ParsedLine`` schema so that the L2
heading-topology, chunk, and Pass-0 sidecar stages keep working unchanged.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mineru.cli.common import do_parse

from road_distress_agent.ingestion.mineru_table_lines import inline_image_refs, table_line_texts
from road_distress_agent.ingestion.models import ParsedBlock, ParsedDocument, ParsedLine

DEFAULT_LANG = "ch"
DEFAULT_BACKEND = "pipeline"
DEFAULT_PARSE_METHOD = "auto"
PARSE_METHODS = frozenset({"auto", "txt", "ocr"})
TITLE_LEVEL_LAYOUT = "title"
TABLE_LAYOUT = "table"
TEXT_LAYOUT = "text"


def source_doc_id_from_path(pdf_path: Path) -> str:
    stem = pdf_path.stem.lower()
    stem = re.sub(r"[^0-9a-zA-Z一-鿿]+", "_", stem).strip("_")
    return stem or "raw_standard"


@dataclass(frozen=True)
class _MinerURun:
    content_list_path: Path
    images_dir: Path


class MinerUAdapter:
    """Parse PDFs through MinerU into the project-owned block/line schema."""

    def __init__(
        self,
        *,
        lang: str = DEFAULT_LANG,
        backend: str = DEFAULT_BACKEND,
        parse_method: str = DEFAULT_PARSE_METHOD,
    ) -> None:
        if parse_method not in PARSE_METHODS:
            raise ValueError(f"Unsupported MinerU parse_method: {parse_method}")
        self.lang = lang
        self.backend = backend
        self.parse_method = parse_method

    def parse_pdf(
        self,
        pdf_path: str | Path,
        *,
        source_doc_id: str | None = None,
        artifact_dir: str | Path | None = None,
        mineru_workdir: str | Path | None = None,
    ) -> ParsedDocument:
        path = Path(pdf_path).resolve()
        doc_id = source_doc_id or source_doc_id_from_path(path)
        artifacts = Path(artifact_dir).resolve() if artifact_dir else None
        workdir = (
            Path(mineru_workdir).resolve() if mineru_workdir else path.parent / "_mineru_workdir"
        )
        run = self._run_mineru(path, doc_id, workdir)
        content_list = json.loads(run.content_list_path.read_text(encoding="utf-8"))
        blocks, lines = _items_to_blocks(content_list, doc_id, str(path), run.images_dir, artifacts)
        return ParsedDocument(
            source_doc_id=doc_id,
            source_path=str(path),
            parser_backend="mineru",
            page_count=_page_count_from_items(content_list),
            blocks=blocks,
            lines=lines,
        )

    def _run_mineru(self, pdf_path: Path, doc_id: str, workdir: Path) -> _MinerURun:
        workdir.mkdir(parents=True, exist_ok=True)
        do_parse(
            output_dir=str(workdir),
            pdf_file_names=[pdf_path.stem],
            pdf_bytes_list=[pdf_path.read_bytes()],
            p_lang_list=[self.lang],
            backend=self.backend,
            parse_method=self.parse_method,
            formula_enable=False,
            table_enable=True,
        )
        run_dir = workdir / pdf_path.stem / self.parse_method
        content_list_path = run_dir / f"{pdf_path.stem}_content_list.json"
        if not content_list_path.is_file():
            raise FileNotFoundError(
                f"MinerU did not produce content_list at {content_list_path}; doc_id={doc_id}"
            )
        return _MinerURun(content_list_path=content_list_path, images_dir=run_dir / "images")


def _page_count_from_items(items: list[dict[str, Any]]) -> int | None:
    if not items:
        return None
    return max(int(item.get("page_idx", 0)) for item in items) + 1


def _items_to_blocks(
    items: list[dict[str, Any]],
    doc_id: str,
    source_path: str,
    images_src: Path,
    artifact_dir: Path | None,
) -> tuple[list[ParsedBlock], list[ParsedLine]]:
    blocks: list[ParsedBlock] = []
    lines: list[ParsedLine] = []
    for order, item in enumerate(items, start=1):
        block = _block_from_item(item, doc_id, source_path, order, images_src, artifact_dir)
        if block is None:
            continue
        blocks.append(block)
        lines.extend(_lines_for_block(block, item))
    return blocks, lines


def _block_from_item(
    item: dict[str, Any],
    doc_id: str,
    source_path: str,
    order: int,
    images_src: Path,
    artifact_dir: Path | None,
) -> ParsedBlock | None:
    item_type = item.get("type")
    if item_type == "text":
        return _text_block(item, doc_id, source_path, order)
    if item_type == "table":
        return _table_block(item, doc_id, source_path, order, images_src, artifact_dir)
    return None


def _text_block(
    item: dict[str, Any],
    doc_id: str,
    source_path: str,
    order: int,
) -> ParsedBlock | None:
    text = str(item.get("text") or "").strip()
    if not text:
        return None
    layout = TITLE_LEVEL_LAYOUT if item.get("text_level") else TEXT_LAYOUT
    return ParsedBlock(
        block_id=_block_id(doc_id, order),
        source_doc_id=doc_id,
        source_path=source_path,
        page_number=_page_number(item),
        order=order,
        text=text,
        layout_type=layout,
        bbox=_bbox(item),
        positions=_positions(item),
    )


def _table_block(
    item: dict[str, Any],
    doc_id: str,
    source_path: str,
    order: int,
    images_src: Path,
    artifact_dir: Path | None,
) -> ParsedBlock | None:
    body = str(item.get("table_body") or "").strip()
    if not body:
        return None
    block_id = _block_id(doc_id, order)
    caption = " ".join(item.get("table_caption") or []).strip()
    text = f"{caption}\n{body}" if caption else body
    rel_table = _persist_table(block_id, body, artifact_dir)
    rel_images = _persist_inline_images(body, images_src, artifact_dir)
    return ParsedBlock(
        block_id=block_id,
        source_doc_id=doc_id,
        source_path=source_path,
        page_number=_page_number(item),
        order=order,
        text=text,
        layout_type=TABLE_LAYOUT,
        bbox=_bbox(item),
        positions=_positions(item),
        table_raw={
            "format": "html",
            "caption": caption,
            "text": body,
            "path": rel_table,
        },
        table_path=rel_table,
        image_paths=rel_images,
        raw={
            "table_caption": item.get("table_caption", []),
            "table_footnote": item.get("table_footnote", []),
        },
    )


def _lines_for_block(block: ParsedBlock, item: dict[str, Any]) -> list[ParsedLine]:
    texts = table_line_texts(item) if block.layout_type == TABLE_LAYOUT else _text_lines(block.text)
    if not texts:
        return []
    return [_make_line(block, idx, text) for idx, text in enumerate(texts, start=1)]


def _text_lines(text: str) -> list[str]:
    pieces = [" ".join(line.replace("　", " ").split()) for line in text.splitlines()]
    return [piece for piece in pieces if piece]


def _persist_table(block_id: str, body: str, artifact_dir: Path | None) -> str | None:
    if artifact_dir is None:
        return None
    tables_dir = artifact_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / f"{block_id.replace(':', '_')}.html"
    path.write_text(body, encoding="utf-8")
    return path.as_posix()


def _persist_inline_images(
    body: str,
    images_src: Path,
    artifact_dir: Path | None,
) -> list[str]:
    refs = inline_image_refs(body)
    if not refs or artifact_dir is None:
        return []
    dst_dir = artifact_dir / "images"
    dst_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        name = Path(ref).name
        if name in seen:
            continue
        src = images_src / name
        if not src.is_file():
            continue
        seen.add(name)
        dst = dst_dir / name
        if not dst.exists():
            shutil.copy(src, dst)
        paths.append(dst.as_posix())
    return paths


def _make_line(block: ParsedBlock, line_order: int, text: str) -> ParsedLine:
    return ParsedLine(
        line_id=f"{block.block_id}:l{line_order:03d}",
        block_id=block.block_id,
        source_doc_id=block.source_doc_id,
        source_path=block.source_path,
        page_number=block.page_number,
        block_order=block.order,
        line_order=line_order,
        text=text,
        layout_type=block.layout_type,
        bbox=block.bbox,
        positions=block.positions,
    )


def _block_id(doc_id: str, order: int) -> str:
    return f"{doc_id}:b{order:06d}"


def _page_number(item: dict[str, Any]) -> int:
    return int(item.get("page_idx", 0)) + 1


def _bbox(item: dict[str, Any]) -> list[float] | None:
    raw = item.get("bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    return [float(raw[0]), float(raw[2]), float(raw[1]), float(raw[3])]


def _positions(item: dict[str, Any]) -> list[list[float]]:
    bbox = item.get("bbox")
    page = _page_number(item)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return []
    return [[float(page), float(bbox[0]), float(bbox[2]), float(bbox[1]), float(bbox[3])]]


__all__ = ["MinerUAdapter", "source_doc_id_from_path"]
