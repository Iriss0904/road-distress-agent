"""Top-level per-turn router for diagnosis, KB QA, and off-topic input."""

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
from road_distress_agent.nodes.deterministic_confirmation_route import (
    SHORTCUT_SOURCE,
    deterministic_confirmation_route,
)
from road_distress_agent.nodes.top_router_context import (
    TopRouteResult,
    router_payload,
)
from road_distress_agent.nodes.top_router_invariants import apply_route_invariants
from road_distress_agent.settings import deterministic_confirm_route_enabled
from road_distress_agent.speculative_prefetch import discard_prefetch_for_route
from road_distress_agent.state import AgentState, AuditEvent, QueryPlan
from road_distress_agent.tracing import trace_event


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    return (Path(__file__).resolve().parents[1] / "prompts" / "top_router.txt").read_text(
        encoding="utf-8"
    )


def _structured_method() -> str:
    return os.environ.get("DEEPSEEK_STRUCTURED_METHOD") or "json_mode"


def _build_messages(state: AgentState) -> list[Any]:
    return [
        SystemMessage(content=_prompt_text()),
        HumanMessage(
            content="\n".join(
                [
                    "router_input = " + json.dumps(router_payload(state), ensure_ascii=False),
                    "Return the TopRouteResult JSON now.",
                ]
            )
        ),
    ]


def _invoke_llm(messages: list[Any]) -> TopRouteResult:
    timeout = llm_timeout_seconds("top_router")
    llm = deepseek_chat(correlation_name="top_router", timeout=timeout)
    method = _structured_method()
    kwargs: dict[str, Any] = {"method": method}
    if method != "json_mode":
        kwargs["strict"] = True
    structured_llm = llm.with_structured_output(TopRouteResult, **kwargs)
    result = invoke_llm_call(
        error_context_name="top_router",
        usage_correlation_name="top_router",
        timeout_seconds=timeout,
        call=lambda: structured_llm.invoke(messages),
    )
    if isinstance(result, TopRouteResult):
        return result
    return TopRouteResult.model_validate(result)


def _missing_text(value: str | None) -> bool:
    return not value or not value.strip()


def _validate_kb_contract(route: TopRouteResult) -> None:
    if route.action != "kb_aside":
        return
    if route.rag_tier is None:
        raise ValueError("top_router adaptive kb_aside route requires rag_tier.")
    if route.rag_tier == "single_hop" and _missing_text(route.rewritten_query):
        raise ValueError("top_router single_hop kb_aside route requires rewritten_query.")
    if route.rag_tier == "no_retrieval" and _missing_text(route.direct_message):
        raise ValueError("top_router no_retrieval kb_aside route requires direct_message.")


def _route_trace_output(
    llm_route: TopRouteResult,
    effective_route: TopRouteResult,
    conflicts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "llm_route": llm_route.model_dump(mode="json"),
        "effective_route": effective_route.model_dump(mode="json"),
        "conflicts": conflicts,
    }


def _single_hop_bridge(route: TopRouteResult) -> dict[str, Any]:
    if route.action != "kb_aside" or route.rag_tier != "single_hop":
        return {}
    if route.rewritten_query is None:
        raise ValueError("top_router single_hop kb_aside route requires rewritten_query.")
    query = route.rewritten_query.strip()
    return {
        "kb_query_plan": QueryPlan(queries=[query], filters={}),
        "kb_rewritten_query": query,
        "kb_plan_type": "single_hop",
        "rag_route": "single_hop",
    }


def _kb_state_bridge(route: TopRouteResult) -> dict[str, Any]:
    if route.action != "kb_aside":
        return {}
    bridge = _single_hop_bridge(route)
    if route.rag_tier != "single_hop":
        bridge["rag_route"] = route.rag_tier
    if route.rag_tier == "no_retrieval":
        bridge["direct_message"] = route.direct_message
    return bridge


def _route_delta(result: TopRouteResult, audit_log: list[AuditEvent]) -> AgentState:
    return {
        **_kb_state_bridge(result),
        "top_route": result.model_dump(mode="json"),
        "user_intent": result.diagnosis_intent or result.action,
        "phase": WorkflowPhase.INPUT,
        "next_action": f"route_{result.action}",
        "audit_log": audit_log,
    }


def _shortcut_delta(state: AgentState, result: TopRouteResult) -> AgentState:
    payload = router_payload(state)
    route = result.model_dump(mode="json")
    trace = trace_event(
        node_name="top_router",
        kind="route_decision",
        title="Deterministic confirmation route",
        inputs=payload,
        output={"effective_route": route, "source": SHORTCUT_SOURCE},
    )
    audit = AuditEvent(
        node_name="top_router",
        message="Classified an unambiguous HITL confirmation deterministically.",
        metadata={"effective_route": route, "source": SHORTCUT_SOURCE},
    )
    return _route_delta(result, [trace, audit])


def top_router(state: AgentState) -> AgentState:
    """Classify the current user turn without mutating diagnostic facts."""
    if deterministic_confirm_route_enabled():
        shortcut = deterministic_confirmation_route(state)
        if shortcut is not None:
            return _shortcut_delta(state, shortcut)
    messages = _build_messages(state)
    raw_result = _invoke_llm(messages)
    invariant_result = apply_route_invariants(state, raw_result)
    result = invariant_result.route
    conflicts = invariant_result.conflicts
    _validate_kb_contract(result)
    payload = router_payload(state)
    trace_output = _route_trace_output(raw_result, result, conflicts)
    source = "llm+invariants" if conflicts else "llm"
    audit_log = [
        trace_event(
            node_name="top_router",
            kind="llm_call",
            title="Top-level route decision",
            inputs=payload,
            prompt=messages,
            output=trace_output,
        ),
    ]
    if conflicts:
        audit_log.append(
            AuditEvent(
                node_name="top_router",
                message="Applied explicit top-router invariant adjustment.",
                metadata=trace_output,
            )
        )
    audit_log.append(
        AuditEvent(
            node_name="top_router",
            message="Classified the current turn at the top level.",
            metadata={**trace_output, "source": source},
        )
    )
    prefetch_patch, prefetch_events = discard_prefetch_for_route(
        state,
        action=result.action,
        diagnosis_intent=result.diagnosis_intent,
    )
    return {
        **_route_delta(result, [*audit_log, *prefetch_events]),
        **prefetch_patch,
    }
