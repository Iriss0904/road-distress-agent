"""State projections used by the planned KB composer prompt."""

from __future__ import annotations

from typing import Any

from road_distress_agent.state import (
    AgentState,
    KbHopResult,
    KbQueryPlan,
    ReferenceItem,
    RetrievedChunk,
)


def plan(state: AgentState) -> KbQueryPlan:
    raw = state.get("kb_query_plan_v2")
    if isinstance(raw, KbQueryPlan):
        return raw
    if isinstance(raw, dict):
        return KbQueryPlan.model_validate(raw)
    raise ValueError("kb_planned_answer_composer requires kb_query_plan_v2.")


def chunks(state: AgentState) -> list[RetrievedChunk]:
    return [
        item if isinstance(item, RetrievedChunk) else RetrievedChunk.model_validate(item)
        for item in (state.get("kb_retrieved_chunks") or [])
    ]


def references(state: AgentState) -> list[ReferenceItem]:
    return [
        item if isinstance(item, ReferenceItem) else ReferenceItem.model_validate(item)
        for item in (state.get("reference_index") or [])
    ]


def evidence_by_hop(
    state: AgentState,
    refs_by_chunk: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "hop_id": result.hop_id,
            "query": result.query,
            "chunk_ids": _chunk_ids(result.chunks, refs_by_chunk),
        }
        for result in _hop_results(state)
    ]


def evidence_groups(
    state: AgentState,
    plan_type: str,
    refs_by_chunk: dict[str, str],
) -> dict[str, list[str]]:
    if plan_type not in {"compare", "evidence_composition"}:
        return {}
    return {
        key: _chunk_ids(values, refs_by_chunk) for key, values in _evidence_slots(state).items()
    }


def _hop_results(state: AgentState) -> list[KbHopResult]:
    return [
        item if isinstance(item, KbHopResult) else KbHopResult.model_validate(item)
        for item in (state.get("kb_hop_results") or [])
    ]


def _evidence_slots(state: AgentState) -> dict[str, list[RetrievedChunk]]:
    raw_slots = state.get("kb_evidence_slots") or {}
    return {
        str(key): [
            item if isinstance(item, RetrievedChunk) else RetrievedChunk.model_validate(item)
            for item in values
        ]
        for key, values in raw_slots.items()
    }


def _chunk_ids(
    values: list[RetrievedChunk],
    refs_by_chunk: dict[str, str],
) -> list[str]:
    return [chunk_id for item in values if (chunk_id := _chunk_id(item) or "") in refs_by_chunk]


def _chunk_id(chunk: RetrievedChunk) -> str | None:
    return chunk.chunk_id or chunk.citation_id
