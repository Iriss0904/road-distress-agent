"""Chunk page span and block artifact helpers."""

from __future__ import annotations

from typing import Any

from road_distress_agent.ingestion.models import ParsedBlock, ParsedDocument, ParsedLine


def block_page_span(blocks: list[ParsedBlock]) -> str:
    pages = sorted({block.page_number for block in blocks if block.page_number})
    return _page_span_from_numbers(pages)


def line_page_span(lines: list[ParsedLine]) -> str:
    pages = sorted({line.page_number for line in lines if line.page_number})
    return _page_span_from_numbers(pages)


def block_artifacts(document: ParsedDocument, lines: list[ParsedLine]) -> dict[str, Any]:
    block_ids = {line.block_id for line in lines}
    blocks = [block for block in document.blocks if block.block_id in block_ids]
    table_raw = [block.table_raw for block in blocks if block.table_raw is not None]
    return {
        "table_raw": table_raw or None,
        "table_paths": [block.table_path for block in blocks if block.table_path],
        "image_paths": [path for block in blocks for path in block.image_paths],
    }


def _page_span_from_numbers(pages: list[int]) -> str:
    if not pages:
        return ""
    ranges = _page_ranges(pages)
    return ",".join(str(start) if start == end else f"{start}-{end}" for start, end in ranges)


def _page_ranges(pages: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = prev = pages[0]
    for page in pages[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append((start, prev))
        start = prev = page
    ranges.append((start, prev))
    return ranges
