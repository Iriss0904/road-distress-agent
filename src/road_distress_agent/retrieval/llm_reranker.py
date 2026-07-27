"""LLM evidence selection after local reranking."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from road_distress_agent.llm_deepseek import deepseek_chat
from road_distress_agent.llm_runtime import invoke_llm_call, llm_timeout_seconds
from road_distress_agent.retrieval.protection import merge_protected_chunks
from road_distress_agent.state import AuditEvent, RetrievedChunk
from road_distress_agent.tracing import trace_event

PROMPT_BODY_LIMIT = 900


class EvidencePick(BaseModel):
    candidate_id: str = Field(description="The exact candidate_id selected from candidates.")
    reason: str = Field(description="One sentence explaining why this evidence supports.")


class EvidenceRerankOutput(BaseModel):
    selected: list[EvidencePick] = Field(default_factory=list)


def select_evidence_with_llm(
    *,
    node_name: str,
    query: str,
    stage_goal: str,
    chunks: list[RetrievedChunk],
    final_top_k: int,
    protected_exact_top_k: int,
    supplied: EvidenceRerankOutput | None,
    selector_tier: str = "R0",
    model_name: str | None = None,
    decision_metadata: dict[str, Any] | None = None,
) -> tuple[list[RetrievedChunk], AuditEvent]:
    messages = _build_messages(query, stage_goal, chunks, final_top_k)
    output = supplied or _invoke_llm(
        messages,
        usage_correlation_name=node_name,
        model_name=model_name,
    )
    selected = merge_protected_chunks(
        _selected_chunks(chunks, output),
        chunks,
        final_top_k,
        protected_exact_top_k,
    )
    trace = trace_event(
        node_name=node_name,
        kind="llm_call",
        title="Intent-aware evidence rerank",
        inputs={"query": query, "stage_goal": stage_goal},
        prompt=messages,
        output=output,
        metadata=_trace_metadata(
            output,
            selected,
            selector_tier=selector_tier,
            model_name=model_name,
            decision_metadata=decision_metadata,
        ),
    )
    return selected, trace


def _trace_metadata(
    output: EvidenceRerankOutput,
    selected: list[RetrievedChunk],
    *,
    selector_tier: str,
    model_name: str | None,
    decision_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        **(decision_metadata or {}),
        "selector_tier": selector_tier,
        "selector_llm_invoked": True,
        "selector_model_override": model_name,
        "selected_candidate_ids": [pick.candidate_id for pick in output.selected],
        "selected_chunk_ids": [_chunk_key(chunk) for chunk in selected],
        "selected_clause_ids": [str(chunk.clause_id) for chunk in selected if chunk.clause_id],
        "rerank_reason": [
            {"candidate_id": pick.candidate_id, "reason": pick.reason} for pick in output.selected
        ],
    }


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    path = Path(__file__).resolve().parents[1] / "prompts" / "evidence_reranker.txt"
    return path.read_text(encoding="utf-8")


def _build_messages(
    query: str,
    stage_goal: str,
    chunks: list[RetrievedChunk],
    final_top_k: int,
) -> list[Any]:
    return [
        SystemMessage(content=_prompt_text()),
        HumanMessage(
            content="\n".join(
                [
                    f"stage_goal = {stage_goal!r}",
                    f"query = {query!r}",
                    f"max_selected = {final_top_k}",
                    "candidate_chunks =",
                    _chunks_text(chunks),
                    "Return the EvidenceRerankOutput JSON now.",
                ]
            )
        ),
    ]


def _invoke_llm(
    messages: list[Any],
    *,
    usage_correlation_name: str,
    model_name: str | None,
) -> EvidenceRerankOutput:
    timeout = llm_timeout_seconds("evidence_reranker")
    llm = deepseek_chat(
        timeout=timeout,
        model=model_name,
        correlation_name=usage_correlation_name,
    )
    method = os.environ.get("DEEPSEEK_STRUCTURED_METHOD") or "json_mode"
    kwargs: dict[str, Any] = {"method": method}
    if method != "json_mode":
        kwargs["strict"] = True
    structured_llm = llm.with_structured_output(EvidenceRerankOutput, **kwargs)
    result = invoke_llm_call(
        error_context_name="evidence_reranker",
        usage_correlation_name=usage_correlation_name,
        timeout_seconds=timeout,
        call=lambda: structured_llm.invoke(messages),
    )
    if isinstance(result, EvidenceRerankOutput):
        return result
    return EvidenceRerankOutput.model_validate(result)


def _chunks_text(chunks: list[RetrievedChunk]) -> str:
    lines = [_chunk_text(chunk) for chunk in chunks]
    return "\n\n".join(lines)


def _chunk_text(chunk: RetrievedChunk) -> str:
    body = (chunk.text or "").strip()[:PROMPT_BODY_LIMIT]
    heading = " → ".join(chunk.heading_path or [])
    prefix = chunk.context_prefix or ""
    return "\n".join(
        [
            f"candidate_id: {_candidate_key(chunk)}",
            f"chunk_id: {_chunk_key(chunk)}",
            f"rank: {chunk.rank}; rerank_score: {_score(chunk):.4f}",
            f"heading_path: {heading}",
            f"context_prefix: {prefix}",
            f"rawtext: {body}",
        ]
    )


def _selected_chunks(
    chunks: list[RetrievedChunk],
    output: EvidenceRerankOutput,
) -> list[RetrievedChunk]:
    by_id = {_candidate_key(chunk): chunk for chunk in chunks}
    selected: list[RetrievedChunk] = []
    for pick in output.selected:
        if pick.candidate_id not in by_id:
            raise ValueError(
                f"LLM evidence reranker selected unknown candidate_id: {pick.candidate_id}"
            )
        selected.append(_with_reason(by_id[pick.candidate_id], pick.reason))
    return selected


def _with_reason(chunk: RetrievedChunk, reason: str) -> RetrievedChunk:
    annotations = {**(chunk.annotations or {}), "llm_rerank_reason": reason}
    return chunk.model_copy(update={"annotations": annotations})


def _chunk_key(chunk: RetrievedChunk) -> str:
    return chunk.chunk_id or chunk.citation_id or f"{chunk.source_doc}:{chunk.clause_id}"


def _candidate_key(chunk: RetrievedChunk) -> str:
    if chunk.rank is None:
        raise ValueError("Evidence candidate rank is required before LLM rerank.")
    return f"C{chunk.rank}"


def _score(chunk: RetrievedChunk) -> float:
    return chunk.score if chunk.score is not None else chunk.similarity or 0.0
