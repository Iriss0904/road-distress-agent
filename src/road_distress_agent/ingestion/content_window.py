"""Select the body content window of a parsed standard.

Two boundaries matter for heading topology:
  * the start of the numbered body (skip cover / 前言 / 目次 front matter), and
  * the start of the back matter (用词说明 / 条文说明 …), which restarts list or
    chapter numbering and must be dropped to avoid spurious / duplicate chapters.
"""

from __future__ import annotations

import re
from dataclasses import replace

from road_distress_agent.ingestion.models import ParsedLine

CONTENT_START_RE = re.compile(r"^\s*1(?:\.0)?\s*总\s*则\s*[。.]?\s*$")
# Standalone back-matter section titles that mark the end of the normative body.
# 用词说明 (word-usage notes) and 条文说明 (provision commentary) both carry their
# own numbering that would otherwise pollute the chapter tree.
_BACK_MATTER_RE = re.compile(r"条文说明|(?:本规程|本标准)?用词(?:用语)?说明")


def content_lines(lines: list[ParsedLine]) -> list[ParsedLine]:
    ordered = sorted(lines, key=lambda item: item.order_key)
    start_key = _content_start_key(ordered)
    body = (
        ordered if start_key is None else [line for line in ordered if line.order_key >= start_key]
    )
    return _truncate_at_back_matter(body)


def _content_start_key(lines: list[ParsedLine]) -> tuple[int, int, int] | None:
    for line in lines:
        if CONTENT_START_RE.match(line.normalized_text()):
            return line.order_key
    return None


def _truncate_at_back_matter(lines: list[ParsedLine]) -> list[ParsedLine]:
    """Drop the non-normative tail (用词说明 / 条文说明 …).

    These sections restart numbering (列表序号 or 章号), which would otherwise add
    spurious or duplicate chapters. The boundary is a standalone section title, so
    we cut at the first content line that is exactly such a marker once whitespace
    is stripped (MinerU sometimes emits it as "条 文说明").
    """
    for idx, line in enumerate(lines):
        before = _text_before_back_matter(line)
        if before is None:
            continue
        if before:
            return [*lines[:idx], replace(line, text=before)]
        return lines[:idx]
    return lines


def _text_before_back_matter(line: ParsedLine) -> str | None:
    compact = line.normalized_text().replace(" ", "")
    match = _BACK_MATTER_RE.search(compact)
    if not match:
        return None
    raw = line.text
    marker = _BACK_MATTER_RE.search(raw.replace(" ", ""))
    if marker and marker.start() == 0:
        return ""
    for token in ("本规程用词用语说明", "本标准用词说明", "条文说明", "条 文说明"):
        index = raw.find(token)
        if index > 0:
            return raw[:index].strip()
    return ""
