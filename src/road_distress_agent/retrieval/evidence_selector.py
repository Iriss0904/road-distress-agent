"""Frozen selector tiers for the O-A evidence-selection ablation."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from road_distress_agent.retrieval.evidence_options import EvidenceRetrievalOptions
from road_distress_agent.retrieval.llm_reranker import (
    EvidenceRerankOutput,
    select_evidence_with_llm,
)
from road_distress_agent.retrieval.protection import merge_protected_chunks
from road_distress_agent.state import AuditEvent, RetrievedChunk
from road_distress_agent.tracing import trace_event

SELECTOR_TIER_ENV = "ROAD_DISTRESS_EVIDENCE_SELECTOR_TIER"
CD_SELECTOR_TIER_ENV = "ROAD_DISTRESS_EVIDENCE_SELECTOR_CD_TIER"
FLASH_MODEL_ENV = "ROAD_DISTRESS_EVIDENCE_SELECTOR_FLASH_MODEL"
DEFAULT_FLASH_MODEL = "deepseek-v4-flash"
R2_BGE_SCORE_THRESHOLD = 3.0
R3_LOW_MARGIN_THRESHOLD = 0.1
B_SELECTOR_NODES = frozenset({"eval", "eval_baseline", "kb_hop_retriever", "kb_retriever"})
CD_SELECTOR_NODES = frozenset({"detail_retriever_v2", "disease_retriever", "method_retriever"})
SELECTOR_NODES = B_SELECTOR_NODES | CD_SELECTOR_NODES


class SelectorTier(str, Enum):
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"


@dataclass(frozen=True)
class SelectorPolicy:
    tier: SelectorTier
    bge_score_threshold: float = R2_BGE_SCORE_THRESHOLD
    low_margin_threshold: float = R3_LOW_MARGIN_THRESHOLD
    flash_model: str = DEFAULT_FLASH_MODEL


def selector_policy(node_name: str | None = None) -> SelectorPolicy:
    configured = _configured_tier(node_name)
    raw_tier = (configured or _default_tier(node_name).value).strip()
    try:
        tier = SelectorTier(raw_tier.upper())
    except ValueError as exc:
        allowed = ", ".join(tier.value for tier in SelectorTier)
        raise ValueError(
            f"{SELECTOR_TIER_ENV} must be one of {allowed}; got {raw_tier!r}."
        ) from exc
    flash_model = (os.environ.get(FLASH_MODEL_ENV) or DEFAULT_FLASH_MODEL).strip()
    if not flash_model:
        raise ValueError(f"{FLASH_MODEL_ENV} must be non-empty.")
    return SelectorPolicy(tier=tier, flash_model=flash_model)


def default_selector_tiers() -> dict[str, Any]:
    return {
        "R1_nodes": sorted(B_SELECTOR_NODES),
        "R0_nodes": sorted(CD_SELECTOR_NODES),
        "global_environment_override": SELECTOR_TIER_ENV,
        "cd_environment_override": CD_SELECTOR_TIER_ENV,
    }


def selector_policy_manifest(
    policy: SelectorPolicy,
    node_name: str | None = None,
) -> dict[str, Any]:
    return {
        "tier": policy.tier.value,
        "tier_source": _tier_source(node_name),
        "r2_bge_score_threshold": policy.bge_score_threshold,
        "r3_low_margin_threshold": policy.low_margin_threshold,
        "r3_comparison_gate": True,
        "r4_selector_model": policy.flash_model,
    }


def selector_policies_manifest(node_names: Iterable[str]) -> dict[str, Any]:
    return {
        node_name: selector_policy_manifest(selector_policy(node_name), node_name)
        for node_name in sorted(node_names)
    }


def select_evidence(
    options: EvidenceRetrievalOptions,
    chunks: list[RetrievedChunk],
    supplied: EvidenceRerankOutput | None,
) -> tuple[list[RetrievedChunk], AuditEvent]:
    policy = selector_policy(options.node_name)
    if _should_invoke_llm(policy, options, chunks):
        return _llm_selection(policy, options, chunks, supplied)
    selected = _deterministic_selection(policy, options, chunks)
    return selected, _deterministic_trace(policy, options, chunks, selected)


def _should_invoke_llm(
    policy: SelectorPolicy,
    options: EvidenceRetrievalOptions,
    chunks: list[RetrievedChunk],
) -> bool:
    if policy.tier in {SelectorTier.R0, SelectorTier.R4}:
        return True
    if policy.tier != SelectorTier.R3:
        return False
    return (
        options.comparison_query
        or _bge_margin(chunks, options.final_top_k) <= policy.low_margin_threshold
    )


def _llm_selection(
    policy: SelectorPolicy,
    options: EvidenceRetrievalOptions,
    chunks: list[RetrievedChunk],
    supplied: EvidenceRerankOutput | None,
) -> tuple[list[RetrievedChunk], AuditEvent]:
    model_name = policy.flash_model if policy.tier == SelectorTier.R4 else None
    return select_evidence_with_llm(
        node_name=options.node_name,
        query=options.query,
        stage_goal=options.stage_goal,
        chunks=chunks,
        final_top_k=options.final_top_k,
        protected_exact_top_k=options.protected_exact_top_k,
        supplied=supplied,
        selector_tier=policy.tier.value,
        model_name=model_name,
        decision_metadata=_decision_metadata(policy, options, chunks, True),
    )


def _deterministic_selection(
    policy: SelectorPolicy,
    options: EvidenceRetrievalOptions,
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    if policy.tier != SelectorTier.R2:
        return chunks[: options.final_top_k]
    above_threshold = [chunk for chunk in chunks if _score(chunk) >= policy.bge_score_threshold]
    primary = above_threshold or chunks[:1]
    return merge_protected_chunks(
        primary,
        chunks,
        options.final_top_k,
        options.protected_exact_top_k,
    )


def _deterministic_trace(
    policy: SelectorPolicy,
    options: EvidenceRetrievalOptions,
    chunks: list[RetrievedChunk],
    selected: list[RetrievedChunk],
) -> AuditEvent:
    metadata = {
        **_decision_metadata(policy, options, chunks, False),
        "selected_chunk_ids": [_chunk_key(chunk) for chunk in selected],
        "selected_clause_ids": _clause_ids(selected),
    }
    return trace_event(
        node_name=options.node_name,
        kind="evidence_selector",
        title="Deterministic evidence selection",
        inputs={"query": options.query, "stage_goal": options.stage_goal},
        output={"chunks": selected},
        metadata=metadata,
    )


def _decision_metadata(
    policy: SelectorPolicy,
    options: EvidenceRetrievalOptions,
    chunks: list[RetrievedChunk],
    llm_invoked: bool,
) -> dict[str, Any]:
    return {
        "selector_tier": policy.tier.value,
        "selector_policy_source": _tier_source(options.node_name),
        "selector_llm_invoked": llm_invoked,
        "comparison_query": options.comparison_query,
        "bge_score_threshold": policy.bge_score_threshold,
        "bge_margin": _bge_margin(chunks, options.final_top_k),
        "bge_margin_cutoff_rank": options.final_top_k,
        "final_top_k": options.final_top_k,
        "reranked_chunk_ids": [_chunk_key(chunk) for chunk in chunks],
        "reranked_clause_ids": _clause_ids(chunks),
        "reranked_scores": [_score(chunk) for chunk in chunks],
    }


def _bge_margin(chunks: list[RetrievedChunk], cutoff_rank: int) -> float:
    scores = sorted((_score(chunk) for chunk in chunks), reverse=True)
    if len(scores) <= cutoff_rank:
        return float("inf")
    return scores[cutoff_rank - 1] - scores[cutoff_rank]


def _score(chunk: RetrievedChunk) -> float:
    return chunk.score if chunk.score is not None else chunk.similarity or 0.0


def _default_tier(node_name: str | None) -> SelectorTier:
    if node_name not in SELECTOR_NODES:
        raise ValueError(f"Unknown evidence selector node: {node_name!r}.")
    return SelectorTier.R1 if node_name in B_SELECTOR_NODES else SelectorTier.R0


def _chunk_key(chunk: RetrievedChunk) -> str:
    return chunk.chunk_id or chunk.citation_id or f"{chunk.source_doc}:{chunk.clause_id}"


def _clause_ids(chunks: list[RetrievedChunk]) -> list[str]:
    return [str(chunk.clause_id) for chunk in chunks if chunk.clause_id]


def _configured_tier(node_name: str | None) -> str | None:
    global_tier = os.environ.get(SELECTOR_TIER_ENV)
    cd_tier = os.environ.get(CD_SELECTOR_TIER_ENV)
    if global_tier and cd_tier:
        raise ValueError(f"{SELECTOR_TIER_ENV} and {CD_SELECTOR_TIER_ENV} cannot both be set.")
    if global_tier:
        return global_tier
    return cd_tier if node_name in CD_SELECTOR_NODES else None


def _tier_source(node_name: str | None) -> str:
    if os.environ.get(SELECTOR_TIER_ENV):
        return "global_environment_override"
    if os.environ.get(CD_SELECTOR_TIER_ENV) and node_name in CD_SELECTOR_NODES:
        return "cd_environment_override"
    return "node_default"
