"""Detect appendix-style headings that are not numeric clauses."""

from __future__ import annotations

import re
from collections.abc import Callable

from road_distress_agent.ingestion.models import HeadingCandidate, ParsedLine

APPENDIX_RE = re.compile(r"^\s*附录\s*(?P<letter>[A-ZＡ-Ｚ])\s*(?P<title>.+)?\s*$")
APPENDIX_CLAUSE_RE = re.compile(r"^\s*(?P<clause>[A-ZＡ-Ｚ](?:\.\d+){1,5})\s*(?P<title>.+)?\s*$")
SPECIAL_SECTION_TITLES = {"本标准用词说明", "引用标准名录"}


def appendix_candidate_from_line(
    line: ParsedLine,
    clean_title: Callable[[str | None], str],
) -> HeadingCandidate | None:
    text = line.normalized_text()
    appendix = APPENDIX_RE.match(text)
    if appendix:
        letter = _normalize_latin_letter(appendix.group("letter"))
        title = clean_title(appendix.group("title"))
        return _plain_heading_candidate(line, f"附录{letter}", [f"附录{letter}", letter], 1, title)
    clause = APPENDIX_CLAUSE_RE.match(text)
    if clause:
        raw = _normalize_appendix_clause(clause.group("clause"))
        title = clean_title(clause.group("title"))
        return _plain_heading_candidate(line, raw, [raw], raw.count(".") + 1, title)
    if line.layout_type == "title" and text in SPECIAL_SECTION_TITLES:
        return _plain_heading_candidate(line, text, [text], 1, "")
    return None


def _normalize_latin_letter(letter: str) -> str:
    char = letter.strip().upper()
    if "Ａ" <= char <= "Ｚ":
        return chr(ord(char) - ord("Ａ") + ord("A"))
    return char


def _normalize_appendix_clause(raw: str) -> str:
    head, *tail = raw.split(".")
    return ".".join([_normalize_latin_letter(head), *tail])


def _plain_heading_candidate(
    line: ParsedLine,
    clause_id: str,
    aliases: list[str],
    level: int,
    title: str,
) -> HeadingCandidate:
    return HeadingCandidate(
        line_id=line.line_id,
        block_id=line.block_id,
        source_doc_id=line.source_doc_id,
        page_number=line.page_number,
        order_key=line.order_key,
        raw_clause_id=aliases[0],
        canonical_clause_id=clause_id,
        clause_aliases=aliases,
        level=level,
        title=title,
        raw_text=line.normalized_text(),
    )
