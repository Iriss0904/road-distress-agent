"""Turn MinerU table HTML into the line texts the L2 step builder expects.

MinerU emits one HTML cell per step (number, full body, image). The step
builder expects lines like ``[number, short_title, body]`` so we synthesise
that triple here while stripping ``<img>`` references for the text view.
"""

from __future__ import annotations

import re
from typing import Any

TITLE_MAX_CHARS = 12
TITLE_MIN_CHARS = 2

TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL | re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
IMG_RE = re.compile(r'<img\s[^>]*?src="([^"]+)"[^>]*>', re.IGNORECASE)
LATEX_RE = re.compile(r"\$[^$]*\$")
PUNCT_BOUNDARY_RE = re.compile(r"[，,。.；;:：]|[（(]")
DESCRIPTION_LEAD_RE = re.compile(
    r"(?<=[一-鿿]{2})"
    r"(用|将|对|检查|进行|采用|铺|碾|清|凿|涂|开放|加热|新|按|应|配制|检测|抽样|测)"
)
NUM_ONLY_RE = re.compile(r"^\d{1,2}$")


def table_line_texts(item: dict[str, Any]) -> list[str]:
    body = str(item.get("table_body") or "")
    caption = " ".join(item.get("table_caption") or []).strip()
    rows = _table_rows(body)
    if not rows:
        return [caption] if caption else []
    output: list[str] = []
    if caption:
        output.append(caption)
    output.extend(_step_lines(rows))
    return output


def inline_image_refs(body_html: str) -> list[str]:
    return IMG_RE.findall(body_html)


def _table_rows(body_html: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr_html in TR_RE.findall(body_html):
        cells = [_clean_cell(cell_html) for cell_html in TD_RE.findall(tr_html)]
        if any(cells):
            rows.append(cells)
    return rows


def _clean_cell(cell_html: str) -> str:
    text = IMG_RE.sub("", cell_html)
    text = TAG_RE.sub("", text)
    text = LATEX_RE.sub("", text)
    text = text.replace("&nbsp;", " ").replace("　", " ")
    return " ".join(text.split())


def _step_lines(rows: list[list[str]]) -> list[str]:
    output: list[str] = []
    for cells in rows:
        first = cells[0].strip() if cells else ""
        if NUM_ONLY_RE.match(first) and len(cells) >= 2:
            output.append(first)
            title, body = _split_step_cell(" ".join(cells[1:]).strip())
            if title:
                output.append(title)
            if body and body != title:
                output.append(body)
            continue
        joined = " ".join(cell for cell in cells if cell)
        if joined:
            output.append(joined)
    return output


def _split_step_cell(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    boundary = _title_boundary(text)
    if boundary is None or boundary < TITLE_MIN_CHARS:
        boundary = min(TITLE_MAX_CHARS, len(text))
    title = text[:boundary].strip()
    body = text[boundary:].strip()
    return title, body


def _title_boundary(text: str) -> int | None:
    punct = PUNCT_BOUNDARY_RE.search(text)
    verb = DESCRIPTION_LEAD_RE.search(text)
    candidates = [m.start() for m in (punct, verb) if m and m.start() <= TITLE_MAX_CHARS]
    return min(candidates) if candidates else None


__all__ = ["table_line_texts", "inline_image_refs"]
