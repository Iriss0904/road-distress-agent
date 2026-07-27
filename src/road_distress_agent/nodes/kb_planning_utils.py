"""Shared helpers for KB query planning nodes."""

from __future__ import annotations

import json
from dataclasses import dataclass

from road_distress_agent.reference_index import build_reference_index
from road_distress_agent.state import Citation, KbHop, KbHopResult, ReferenceItem, RetrievedChunk

PLANNED_REFERENCE_BUDGET = 10
EVIDENCE_TEXT_LIMIT = 800
EVIDENCE_CHUNK_LIMIT = 16
CHUNK_KEY_TEXT_LIMIT = 80


@dataclass(frozen=True)
class BudgetedEvidence:
    chunks: list[RetrievedChunk]
    references: list[ReferenceItem]


def budgeted_evidence(
    plan_type: str,
    hops: list[KbHop],
    hop_results: list[KbHopResult],
    max_refs: int = PLANNED_REFERENCE_BUDGET,
) -> BudgetedEvidence:
    selected = _select_budgeted_chunks(plan_type, hops, hop_results, max_refs)
    references = build_reference_index([_citation(chunk) for chunk in selected], max_items=max_refs)
    return BudgetedEvidence(chunks=_citable_chunks(selected, references), references=references)


def references_json(references: list[ReferenceItem]) -> str:
    items = [
        {
            "ref_id": item.ref_id,
            "chunk_ids": item.chunk_ids,
            "title": item.title,
            "source_clause": item.source_clause,
        }
        for item in references
    ]
    return json.dumps(items, ensure_ascii=False)


def evidence_text(
    chunks: list[RetrievedChunk],
    references: list[ReferenceItem],
) -> str:
    if not chunks:
        return "（未检索到可引用证据）"
    refs_by_chunk = refs_by_chunk_id(references)
    lines: list[str] = []
    for index, chunk in enumerate(chunks[:EVIDENCE_CHUNK_LIMIT], 1):
        chunk_id = chunk.chunk_id or chunk.citation_id or f"chunk-{index}"
        ref_id = refs_by_chunk.get(chunk_id, "")
        clause = chunk.clause_id or chunk.source_clause or ""
        prefix = chunk.context_prefix or ""
        body = (chunk.text or chunk.snippet or "").strip()[:EVIDENCE_TEXT_LIMIT]
        lines.append(
            f"[{index}] ref_id={ref_id} chunk_id={chunk_id} clause={clause}\n{prefix}\n{body}"
        )
    return "\n\n".join(lines)


def refs_by_chunk_id(references: list[ReferenceItem]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for item in references:
        for chunk_id in item.chunk_ids:
            lookup[chunk_id] = item.ref_id
    return lookup


def _select_budgeted_chunks(
    plan_type: str,
    hops: list[KbHop],
    hop_results: list[KbHopResult],
    budget: int,
) -> list[RetrievedChunk]:
    if plan_type == "compare":
        return _round_robin_chunks(hops, hop_results, budget)
    if plan_type == "evidence_composition":
        return _slot_budget_chunks(hops, hop_results, budget)
    return _top_scored_chunks(_all_chunks(hop_results), budget)


def _round_robin_chunks(
    hops: list[KbHop],
    hop_results: list[KbHopResult],
    budget: int,
) -> list[RetrievedChunk]:
    groups = [_sorted_chunks(result.chunks) for result in _ordered_results(hops, hop_results)]
    selected: list[RetrievedChunk] = []
    seen: set[str] = set()
    while len(selected) < budget and any(groups):
        for group in groups:
            while group and len(selected) < budget:
                chunk = group.pop(0)
                key = _chunk_key(chunk)
                if key not in seen:
                    seen.add(key)
                    selected.append(chunk)
                    break
    return selected


def _slot_budget_chunks(
    hops: list[KbHop],
    hop_results: list[KbHopResult],
    budget: int,
) -> list[RetrievedChunk]:
    groups = [_sorted_chunks(result.chunks) for result in _ordered_results(hops, hop_results)]
    first_pass = [group.pop(0) for group in groups if group]
    remaining = _top_scored_chunks([chunk for group in groups for chunk in group], budget)
    return _dedupe_chunks([*first_pass, *remaining])[:budget]


def _ordered_results(hops: list[KbHop], results: list[KbHopResult]) -> list[KbHopResult]:
    by_hop = {result.hop_id: result for result in results}
    ordered = [by_hop[hop.hop_id] for hop in hops if hop.hop_id in by_hop]
    return [*ordered, *[result for result in results if result.hop_id not in by_hop]]


def _all_chunks(hop_results: list[KbHopResult]) -> list[RetrievedChunk]:
    return [chunk for result in hop_results for chunk in result.chunks]


def _top_scored_chunks(chunks: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
    return _dedupe_chunks(_sorted_chunks(chunks))[:limit]


def _sorted_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return sorted(chunks, key=_chunk_score, reverse=True)


def _dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    unique: list[RetrievedChunk] = []
    for chunk in chunks:
        key = _chunk_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def _citable_chunks(
    chunks: list[RetrievedChunk],
    references: list[ReferenceItem],
) -> list[RetrievedChunk]:
    citable_ids = set(refs_by_chunk_id(references))
    return [chunk for chunk in chunks if _chunk_id(chunk) in citable_ids]


def _chunk_score(chunk: RetrievedChunk) -> float:
    annotations = chunk.annotations or {}
    score = annotations.get("evidence_quality") or annotations.get("rerank_score")
    if isinstance(score, (int, float)):
        return float(score)
    return chunk.score if chunk.score is not None else chunk.similarity or 0.0


def _chunk_key(chunk: RetrievedChunk) -> str:
    return _chunk_id(chunk) or "|".join(
        [
            chunk.title or "",
            chunk.source_doc or chunk.document_id or "",
            chunk.clause_id or chunk.source_clause or "",
            (chunk.text or chunk.snippet or "")[:CHUNK_KEY_TEXT_LIMIT],
        ]
    )


def _chunk_id(chunk: RetrievedChunk) -> str | None:
    return chunk.chunk_id or chunk.citation_id


def _citation(chunk: RetrievedChunk) -> Citation:
    return Citation.model_validate(chunk.model_dump(mode="json"))
