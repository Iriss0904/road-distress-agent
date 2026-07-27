"""Shared inline-citation helpers for final-answer rendering."""

from __future__ import annotations

import re
from collections.abc import Iterable

from road_distress_agent.state import ReferenceItem

CLAUSE_ID_PATTERN = r"\d+\.\d+\.\d+(?:\([^)]+\))?"
CLAUSE_ID_RE = re.compile(rf"(?<!\d)({CLAUSE_ID_PATTERN})(?!\d)")
CLAUSE_SEPARATOR_RE = re.compile(r"[;；,，、\s]+")
CLAUSE_REFERENCE_BLOCK_RE = re.compile(
    rf"[（(]\s*(?:依据|reference|references|source|sources|clause|clauses)\s*"
    rf"[:：]\s*({CLAUSE_ID_PATTERN}(?:[;；,，、\s]+{CLAUSE_ID_PATTERN})*)\s*[)）]",
    re.IGNORECASE,
)


def clause_ids(value: str) -> list[str]:
    return CLAUSE_ID_RE.findall(value)


def split_clause_ids(value: str) -> list[str]:
    return [item for item in CLAUSE_SEPARATOR_RE.split(value) if item]


def reference_ids_by_clause(references: list[ReferenceItem]) -> dict[str, str]:
    result: dict[str, str] = {}
    for reference in references:
        for clause in split_clause_ids(reference.source_clause or ""):
            result.setdefault(clause, reference.ref_id)
    return result


def with_clause_citation_tokens(
    items: Iterable[str],
    references: list[ReferenceItem],
) -> list[str]:
    ref_by_clause = reference_ids_by_clause(references)
    return [_with_clause_citation_token(item, ref_by_clause) for item in items]


def _with_clause_citation_token(item: str, ref_by_clause: dict[str, str]) -> str:
    ref_ids = _unique_reference_ids(clause_ids(item), ref_by_clause)
    if not ref_ids:
        return item
    text = CLAUSE_REFERENCE_BLOCK_RE.sub("", item).strip()
    return f"{text} {' '.join(f'[[{ref_id}]]' for ref_id in ref_ids)}"


def _unique_reference_ids(
    clauses: Iterable[str],
    ref_by_clause: dict[str, str],
) -> list[str]:
    result: list[str] = []
    for clause in clauses:
        ref_id = ref_by_clause.get(clause)
        if ref_id and ref_id not in result:
            result.append(ref_id)
    return result
