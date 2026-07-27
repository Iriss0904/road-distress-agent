"""Chunk-boundary and nearby-context helpers for ingestion."""

from __future__ import annotations

import re

from road_distress_agent.ingestion.models import (
    HeadingCandidate,
    ParsedBlock,
    ParsedLine,
)
from road_distress_agent.ingestion.table_helpers import TABLE_LAYOUT, is_table_caption

MAX_GENERIC_CHUNK_DEPTH = 3
LOCAL_CONTEXT_MAX_CHARS = 40

_CLAUSE_ID_RE = re.compile(r"^\d+(?:\.\d+)*$")
_CLAUSE_HEADING_RE = re.compile(r"^\d+(?:\.\d+){1,5}\s")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ENUM_TITLE_RE = re.compile(
    r"^\s*(?:"
    r"[（(]\s*(?:\d{1,2}|[一二三四五六七八九十]{1,3})\s*[）)]"
    r"|(?:\d{1,2}|[一二三四五六七八九十]{1,3})[、.)）]"
    r"|(?:方案|方法|工法|措施)\s*(?:[一二三四五六七八九十]|\d{1,2})"
    r")\s*(?P<title>.+)$"
)
_SENTENCE_ENDINGS = ("。", "；", ";", "！", "？")


def chunk_headings(
    headings: list[HeadingCandidate],
) -> list[HeadingCandidate]:
    return [heading for heading in headings if is_chunk_heading(heading)]


def is_chunk_heading(heading: HeadingCandidate) -> bool:
    depth = _clause_depth(heading.canonical_clause_id)
    return depth is None or depth <= MAX_GENERIC_CHUNK_DEPTH


def table_local_context(
    blocks: list[ParsedBlock],
    body_lines: list[ParsedLine],
) -> str | None:
    if not blocks:
        return None
    for line in reversed(_candidate_lines_before_table(blocks[0], body_lines)):
        title = _local_title(line)
        if title:
            return title
    return None


def table_chunk_text(caption: str, local_context: str | None) -> str:
    if local_context:
        return f"{local_context}\n{caption}".strip()
    return caption


def table_raw_captions(captions: list[str], primary: str) -> list[str]:
    if any(captions):
        return captions
    if not captions:
        return []
    return [primary, *captions[1:]]


def table_heading_path(
    heading_path: list[str],
    caption: str,
    local_context: str | None,
) -> list[str]:
    path = [*heading_path]
    if local_context:
        path.append(f"局部上下文：{local_context}")
    path.append(caption)
    return path


def prior_non_table_lines(
    block: ParsedBlock,
    body_lines: list[ParsedLine],
) -> list[ParsedLine]:
    after_order = _previous_table_order(block, body_lines)
    return [
        line
        for line in body_lines
        if line.layout_type != TABLE_LAYOUT and after_order < line.block_order < block.order
    ]


def _candidate_lines_before_table(
    block: ParsedBlock,
    body_lines: list[ParsedLine],
) -> list[ParsedLine]:
    return [
        line
        for line in prior_non_table_lines(block, body_lines)
        if not is_table_caption(line.normalized_text())
    ]


def _previous_table_order(block: ParsedBlock, body_lines: list[ParsedLine]) -> int:
    orders = [
        line.block_order
        for line in body_lines
        if line.layout_type == TABLE_LAYOUT and line.block_order < block.order
    ]
    return max(orders, default=-1)


def _local_title(line: ParsedLine) -> str | None:
    raw_text = line.normalized_text()
    text = _clean_local_title(raw_text)
    if not _valid_local_title(text):
        return None
    if _ENUM_TITLE_RE.match(text):
        return text
    if raw_text.rstrip().endswith(_SENTENCE_ENDINGS):
        return None
    if line.layout_type == "title" or _looks_like_short_title(text):
        return text
    return None


def _clean_local_title(text: str) -> str:
    compact = " ".join(text.replace("\u3000", " ").split())
    return compact.strip(" ：:。；;，,")


def _valid_local_title(text: str) -> bool:
    if not text or len(text) > LOCAL_CONTEXT_MAX_CHARS:
        return False
    if not _CJK_RE.search(text):
        return False
    if is_table_caption(text) or _CLAUSE_HEADING_RE.match(text):
        return False
    return True


def _looks_like_short_title(text: str) -> bool:
    return len(text) <= 24


def _clause_depth(clause_id: str) -> int | None:
    if not _CLAUSE_ID_RE.fullmatch(clause_id):
        return None
    return len([part for part in clause_id.split(".") if part])
