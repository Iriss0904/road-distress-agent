"""Build a deduplicated reference index for user-facing citations."""

from __future__ import annotations

import re

from road_distress_agent.state import Citation, ReferenceItem

DEFAULT_MAX_REFERENCES = 12
SNIPPET_SEPARATOR = "\n\n"


def build_reference_index(
    citations: list[Citation],
    max_items: int = DEFAULT_MAX_REFERENCES,
) -> list[ReferenceItem]:
    index: list[ReferenceItem] = []
    positions: dict[tuple[str, str, str], int] = {}
    for citation in citations:
        key = _dedupe_key(citation)
        if key in positions:
            index[positions[key]] = _merge(index[positions[key]], citation)
            continue
        if len(index) >= max_items:
            continue
        item = _item_from_citation(citation, len(index) + 1)
        positions[key] = len(index)
        index.append(item)
    return index


def reference_ids(references: list[ReferenceItem]) -> list[str]:
    return [item.ref_id for item in references]


def _item_from_citation(citation: Citation, number: int) -> ReferenceItem:
    return ReferenceItem(
        ref_id=f"R{number}",
        title=_title(citation),
        source_doc=citation.source_doc or citation.document_id,
        source_standard=citation.source_standard,
        source_clause=citation.source_clause or citation.clause_id,
        source_pages=citation.source_pages,
        snippet=_snippet(citation),
        chunk_ids=_chunk_ids(citation),
        evidence_quality=_evidence_quality(citation),
    )


def _merge(item: ReferenceItem, citation: Citation) -> ReferenceItem:
    chunk_ids = [*item.chunk_ids]
    for chunk_id in _chunk_ids(citation):
        if chunk_id not in chunk_ids:
            chunk_ids.append(chunk_id)
    snippet = _merge_snippets(item.snippet, _snippet(citation))
    quality = _best_quality(item.evidence_quality, _evidence_quality(citation))
    return item.model_copy(
        update={"chunk_ids": chunk_ids, "snippet": snippet, "evidence_quality": quality}
    )


def _merge_snippets(existing: str, incoming: str) -> str:
    snippets = [part for part in existing.split(SNIPPET_SEPARATOR) if part]
    incoming_identity = _snippet_identity(incoming)
    identities = {_snippet_identity(part) for part in snippets}
    if incoming_identity and incoming_identity not in identities:
        snippets.append(incoming)
    return SNIPPET_SEPARATOR.join(snippets)


def _snippet_identity(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _dedupe_key(citation: Citation) -> tuple[str, str, str]:
    title = _normalize(_title(citation))
    source = _normalize(citation.source_doc or citation.document_id or "")
    clause = _normalize(citation.source_clause or citation.clause_id or "")
    if not title and not source and not clause:
        return (_normalize(_snippet(citation)), "", "")
    return (title, source, clause)


def _title(citation: Citation) -> str:
    return (
        citation.title
        or citation.source_standard
        or citation.source_doc
        or citation.document_id
        or "未命名资料"
    )


def _snippet(citation: Citation) -> str:
    text = citation.snippet or citation.text or ""
    return re.sub(r"\s+", " ", text).strip()


def _chunk_ids(citation: Citation) -> list[str]:
    chunk_ids: list[str] = []
    for item in (citation.chunk_id, citation.citation_id):
        if item and item not in chunk_ids:
            chunk_ids.append(item)
    return chunk_ids


def _evidence_quality(citation: Citation) -> float | None:
    quality = (citation.annotations or {}).get("evidence_quality")
    return float(quality) if isinstance(quality, (int, float)) else None


def _best_quality(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).strip().lower()
