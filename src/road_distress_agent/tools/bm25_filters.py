"""Metadata matching for the in-memory BM25 index."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from road_distress_agent.state import Citation

FILTER_KEYS = ("clause_id", "sequential_group_id", "semantic_role", "source_doc_id")


def matches_filters(chunk: Citation, filters: Mapping[str, Any]) -> bool:
    return all(_filter_matches(chunk, key, filters.get(key)) for key in FILTER_KEYS)


def _filter_matches(chunk: Citation, key: str, expected: Any) -> bool:
    if expected is None:
        return True
    actual = _filter_value(chunk, key)
    return bool(_string_values(actual) & _string_values(expected))


def _string_values(value: Any) -> frozenset[str]:
    if isinstance(value, Collection) and not isinstance(value, (str, bytes, Mapping)):
        return frozenset(str(item) for item in value)
    return frozenset({str(value)})


def _filter_value(chunk: Citation, key: str) -> Any:
    if key == "semantic_role":
        return (chunk.annotations or {}).get("semantic_role")
    if key == "source_doc_id":
        return chunk.source_doc or chunk.document_id or (chunk.annotations or {}).get(key)
    return getattr(chunk, key, None)
