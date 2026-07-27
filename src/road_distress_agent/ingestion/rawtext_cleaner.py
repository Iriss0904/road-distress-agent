"""Clean and normalize chunk text for RAG ingestion.

Converts MinerU's raw output (HTML tables, OCR artifacts, LaTeX math)
into clean plaintext or Markdown pipe tables suitable for embedding.
"""

from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup


def clean_latex(text: str) -> str:
    """Convert inline LaTeX math expressions to readable Unicode text.

    Handles patterns produced by MinerU's PDF math extraction, e.g.:
      "$0 . 0 4 \\mathrm { m } ^ { 2 }$"  →  "0.04m²"
    """

    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        # Collapse OCR-spaced digits: "0 . 0 4" → "0.04", "1 5" → "15"
        inner = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", inner)
        inner = re.sub(r"(\d)\s+(\d)", r"\1\2", inner)

        # \mathrm{x} or \mathrm{m m} or \mathrm x → x (collapse spaces inside braces)
        def _mathrm_replace(m: re.Match[str]) -> str:
            content = re.sub(r"\s+", "", m.group(1))
            return content

        inner = re.sub(r"\\mathrm\s*\{\s*([\w\s]+?)\s*\}", _mathrm_replace, inner)
        inner = re.sub(r"\\mathrm\s+(\w+)", r"\1", inner)
        # Superscripts: ^{2} or ^2 → ²; ^{3} or ^3 → ³
        inner = re.sub(r"\^\s*\{\s*2\s*\}", "²", inner)
        inner = re.sub(r"\^2\b", "²", inner)
        inner = re.sub(r"\^\s*\{\s*3\s*\}", "³", inner)
        inner = re.sub(r"\^3\b", "³", inner)
        # Drop any remaining LaTeX commands
        inner = re.sub(r"\\[a-zA-Z]+\s*\{[^}]*\}", "", inner)
        inner = re.sub(r"\\[a-zA-Z]+", "", inner)
        # Collapse spaces between digit/unit tokens (e.g. "0.04 m ²" → "0.04m²")
        inner = re.sub(r"(\d)\s+([a-zA-Z])", r"\1\2", inner)
        inner = re.sub(r"([a-zA-Z])\s+(²)", r"\1\2", inner)
        # Collapse any remaining multiple spaces
        inner = re.sub(r"\s+", " ", inner)
        return inner.strip()

    return re.sub(r"\$([^$]+)\$", _replace, text)


def clean_text(text: str) -> str:
    """Light cleanup for text and step chunks.

    - Replaces full-width spaces (U+3000) with regular spaces.
    - Merges short dangling lines caused by OCR mid-word breaks.
      A dangling line is ≤ 6 characters and follows a line that does
      not end with Chinese sentence-final punctuation.
    """
    text = text.replace("　", " ")
    sentence_endings = ("。", "；", "：", "！", "？", ".", ";", ":")
    lines = text.split("\n")
    merged: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        while (
            i + 1 < len(lines)
            and lines[i + 1].strip()
            and len(lines[i + 1].strip()) <= 6
            and not any(line.endswith(p) for p in sentence_endings)
        ):
            i += 1
            line = line + lines[i].strip()
        merged.append(line)
        i += 1
    return "\n".join(merged).strip()


def _cell_is_image_only(td: Any) -> bool:
    """Return True if a table cell contains only an image with no visible text."""
    return bool(td.find("img")) and not td.get_text(strip=True)


def _find_image_only_columns(all_rows: list[list[Any]]) -> set[int]:
    """Return column indices where ALL data rows (non-header) have image-only cells.

    A column is considered image-only if every data row has an image-only cell
    at that position. Header rows (first row) are excluded from this check.
    """
    if len(all_rows) <= 1:
        return set()
    data_rows = all_rows[1:]
    if not data_rows:
        return set()
    max_cols = max(len(r) for r in data_rows)
    image_cols: set[int] = set()
    for col_idx in range(max_cols):
        if all(col_idx < len(row) and _cell_is_image_only(row[col_idx]) for row in data_rows):
            image_cols.add(col_idx)
    return image_cols


def html_table_to_markdown(html: str) -> str:
    """Convert an HTML table string to a Markdown pipe table.

    - Skips image-only columns (detected from data rows, applied to all rows).
    - Cleans LaTeX expressions in cell text.
    - Escapes pipe characters inside cell content.
    - Returns empty string if the table has no usable rows.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return ""

    # Collect raw td/th elements per row first (to detect image columns).
    # Use tbody if present so recursive=False sees the <tr> children directly,
    # avoiding rows pulled from nested <table> elements inside cells.
    body = table.find("tbody") or table
    raw_rows: list[list[Any]] = []
    for tr in body.find_all("tr", recursive=False):
        tds = tr.find_all(["td", "th"])
        if tds:
            raw_rows.append(list(tds))

    if not raw_rows:
        return ""

    # Detect which columns are image-only across all data rows
    image_cols = _find_image_only_columns(raw_rows)

    rows: list[list[str]] = []
    for raw_row in raw_rows:
        cells: list[str] = []
        for col_idx, td in enumerate(raw_row):
            if col_idx in image_cols:
                continue
            raw = td.get_text(separator=" ", strip=True)
            raw = clean_latex(raw)
            raw = raw.replace("|", "｜")
            cells.append(raw)
        if any(cells):
            rows.append(cells)

    if not rows:
        return ""

    max_cols = max(len(r) for r in rows)

    def pad(row: list[str]) -> list[str]:
        return row + [""] * (max_cols - len(row))

    header = pad(rows[0])
    separator = ["---"] * max_cols
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(separator) + "|",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(pad(row)) + " |")

    return "\n".join(lines)


def classify_chunk(chunk: dict[str, Any]) -> str:
    """Return 'step', 'table', or 'text' for a RawIngestionChunk dict."""
    if chunk.get("sequential_group_id"):
        return "step"
    if chunk.get("table_raw"):
        return "table"
    return "text"


def _extract_intro_text(chunk_text: str, caption: str) -> str:
    """Extract the prose that appears before a table caption in chunk text."""
    if not caption:
        return ""
    idx = chunk_text.find(caption)
    if idx == -1:
        return ""
    before = chunk_text[:idx].strip()
    # Drop the last line if it is just the heading (already in heading_path)
    lines = [ln for ln in before.split("\n") if ln.strip()]
    if not lines:
        return ""
    # Heuristic: if the last line looks like a clause heading (short, no verb),
    # skip it — it duplicates heading_path info.
    last = lines[-1].strip()
    if len(last) < 20 and re.match(r"^[\d.]+\s", last):
        lines = lines[:-1]
    return clean_text("\n".join(lines))


def build_rawtext(chunk: dict[str, Any]) -> str:
    """Return the cleaned rawtext for a chunk, ready for RAG embedding.

    - text / step chunks: light text cleanup.
    - table chunks: intro prose + caption + Markdown pipe table(s).
    """
    chunk_type = classify_chunk(chunk)

    if chunk_type in ("text", "step"):
        return clean_latex(clean_text(chunk.get("text") or ""))

    # table chunk
    table_raw = chunk.get("table_raw") or []
    if not isinstance(table_raw, list):
        table_raw = [table_raw]

    chunk_text = chunk.get("text") or ""
    parts: list[str] = []

    for entry in table_raw:
        caption = (entry.get("caption") or "").strip()
        html = entry.get("text") or ""

        intro = _extract_intro_text(chunk_text, caption)
        if intro:
            parts.append(intro)
        if caption:
            parts.append(caption)
        md = html_table_to_markdown(html)
        if md:
            parts.append(md)

    return "\n\n".join(parts).strip()
