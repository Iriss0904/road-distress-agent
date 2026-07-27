"""Step A query rewriting for the road distress RAG pipeline.

Two modes:
  disease_query_rewriter  → "identify_disease"
  query_rewriter          → "select_method" (treatment plan retrieval)
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.llm_deepseek import deepseek_chat
from road_distress_agent.llm_runtime import invoke_llm_call, llm_timeout_seconds
from road_distress_agent.localization import query_language_instruction
from road_distress_agent.nodes.method_retrieval_context import (
    enrich_method_query,
    method_context_from_state,
)
from road_distress_agent.state import AgentState, AuditEvent, QueryPlan
from road_distress_agent.tracing import trace_event


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    return (Path(__file__).resolve().parents[1] / "prompts" / "query_rewriter.txt").read_text(
        encoding="utf-8"
    )


def _structured_method() -> str:
    return os.environ.get("DEEPSEEK_STRUCTURED_METHOD") or "json_mode"


def _resolve_distress_fields(
    state: AgentState,
) -> tuple[str | None, str | None, dict[str, Any], str | None]:
    """Return (material, defect, known_features, raw_text).

    Top-level state fields take precedence over the legacy state.distress bridge.
    """
    material: str | None = state.get("material")
    defect: str | None = state.get("defect_category")
    features: dict[str, Any] = state.get("known_features") or {}
    raw_text: str | None = state.get("latest_user_text") or state.get("raw_user_text")

    distress = state.get("distress")
    if distress:
        if material is None:
            material = distress.material
        if defect is None:
            defect = distress.defect_category
        if not features:
            features = distress.known_features or {}
        if raw_text is None:
            raw_text = distress.raw_user_text

    return material, defect, features, raw_text


def _query_inputs(state: AgentState) -> dict[str, Any]:
    material, defect, features, raw_text = _resolve_distress_fields(state)
    return {
        "material": material,
        "defect_category": defect,
        "known_features": features,
        "latest_user_text": raw_text,
    }


def _build_messages(state: AgentState, query_mode: str) -> list[Any]:
    query_inputs = _query_inputs(state)
    distress_json = json.dumps(
        {
            "material": query_inputs["material"],
            "defect_category": query_inputs["defect_category"],
            "known_features": query_inputs["known_features"],
        },
        ensure_ascii=False,
    )
    return [
        SystemMessage(content=_prompt_text()),
        HumanMessage(
            content="\n".join(
                [
                    f"query_mode = {json.dumps(query_mode, ensure_ascii=False)}",
                    f"target_locale = {json.dumps(state.get('locale') or 'zh-CN')}",
                    query_language_instruction(),
                    f"distress = {distress_json}",
                    f"latest_user_text = {(query_inputs['latest_user_text'] or '')!r}",
                    "Return the QueryPlan JSON now.",
                ]
            )
        ),
    ]


def _invoke_llm(messages: list[Any], node_name: str) -> QueryPlan:
    timeout = llm_timeout_seconds(node_name)
    llm = deepseek_chat(correlation_name=node_name, timeout=timeout)
    method = _structured_method()
    kwargs: dict[str, Any] = {"method": method}
    if method != "json_mode":
        kwargs["strict"] = True
    structured_llm = llm.with_structured_output(QueryPlan, **kwargs)
    result = invoke_llm_call(
        error_context_name=node_name,
        usage_correlation_name=node_name,
        timeout_seconds=timeout,
        call=lambda: structured_llm.invoke(messages),
    )
    return result if isinstance(result, QueryPlan) else QueryPlan.model_validate(result)


def _effective_plan(plan: QueryPlan, query_mode: str, state: AgentState) -> QueryPlan:
    if not plan.queries:
        raise ValueError(f"query_rewriter LLM returned no queries for {query_mode}.")
    queries = plan.queries[:1]
    if query_mode == "select_method":
        context = method_context_from_state(state)
        queries = [enrich_method_query(queries[0], context)]
    return plan.model_copy(update={"queries": queries, "filters": {}})


def _run_query_rewriter(state: AgentState, query_mode: str, node_name: str) -> AgentState:
    messages = _build_messages(state, query_mode)
    raw_plan = _invoke_llm(messages, node_name)
    plan = _effective_plan(raw_plan, query_mode, state)
    rewritten_query = plan.queries[0]
    next_action = (
        "retrieve_disease_definition"
        if query_mode == "identify_disease"
        else "retrieve_method_evidence"
    )
    return {
        "rewritten_query": rewritten_query,
        "query_plan": plan,
        "phase": WorkflowPhase.EVIDENCE,
        "next_action": next_action,
        "audit_log": [
            trace_event(
                node_name=node_name,
                kind="llm_call",
                title=f"Query rewrite ({query_mode})",
                inputs=_query_inputs(state),
                prompt=messages,
                output={"llm_output": raw_plan, "effective_plan": plan},
            ),
            AuditEvent(
                node_name=node_name,
                message=f"Step A query rewrite (mode={query_mode}).",
                metadata={
                    "query_mode": query_mode,
                    "rewritten_query": rewritten_query,
                    "filters": plan.filters,
                },
            ),
        ],
    }


def query_rewriter(state: AgentState) -> AgentState:
    """Step A — treatment mode: generate a broad treatment-plan retrieval query."""
    return _run_query_rewriter(state, query_mode="select_method", node_name="method_query_rewriter")


def disease_query_rewriter(state: AgentState) -> AgentState:
    """Step A — disease identification mode."""
    return _run_query_rewriter(
        state,
        query_mode="identify_disease",
        node_name="disease_query_rewriter",
    )
