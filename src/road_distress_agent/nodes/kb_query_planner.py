"""Plan KB QA retrieval topology before executing evidence search."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.evidence_observation import (
    evidence_observation_event,
    observe_kb_plan,
)
from road_distress_agent.llm_deepseek import deepseek_chat
from road_distress_agent.llm_runtime import invoke_llm_call, llm_timeout_seconds
from road_distress_agent.nodes.top_router_context import diagnosis_context, recent_dialogue
from road_distress_agent.state import (
    AgentState,
    AuditEvent,
    KbPlanType,
    KbQueryPlan,
    QueryPlan,
)
from road_distress_agent.tracing import trace_event

# Confidence gate (fix ②): when the planner is not confident about a multi-hop
# topology, route conservatively to the single-hop path instead of committing to
# a possibly-wrong chain/compare/evidence_composition plan. Explicit + logged
# routing policy on a first-class signal, not a silent fallback. Disable by
# setting KB_PLANNER_MIN_CONFIDENCE=0.
LOW_CONFIDENCE_DEFAULT = 0.55
PLANNER_STRUCTURED_METHOD_ENV = "KB_QUERY_PLANNER_STRUCTURED_METHOD"
PLANNER_STRUCTURED_METHOD_DEFAULT = "function_calling"
_DOWNGRADEABLE_PLAN_TYPES = frozenset({"chain", "compare", "evidence_composition"})


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    return (Path(__file__).resolve().parents[1] / "prompts" / "kb_query_planner.txt").read_text(
        encoding="utf-8"
    )


def _structured_method() -> str:
    return os.environ.get(PLANNER_STRUCTURED_METHOD_ENV) or PLANNER_STRUCTURED_METHOD_DEFAULT


def _min_confidence() -> float:
    raw = os.environ.get("KB_PLANNER_MIN_CONFIDENCE")
    if raw is None:
        return LOW_CONFIDENCE_DEFAULT
    try:
        return float(raw)
    except ValueError:
        return LOW_CONFIDENCE_DEFAULT


def _gated_plan_type(plan: KbQueryPlan) -> tuple[KbPlanType, bool]:
    """Downgrade a low-confidence multi-hop plan to single_hop; return (type, gated)."""
    if plan.plan_type in _DOWNGRADEABLE_PLAN_TYPES and plan.confidence < _min_confidence():
        return "single_hop", True
    return plan.plan_type, False


def _coref_anchors(diagnosis_context: dict[str, Any]) -> dict[str, Any]:
    """Minimal, classification-relevant coreference hints (names only)."""
    candidates = diagnosis_context.get("last_candidates") or []
    return {
        "confirmed_defect": diagnosis_context.get("confirmed_defect"),
        "confirmed_method": diagnosis_context.get("confirmed_method"),
        "last_candidates": [c.get("name") for c in candidates if isinstance(c, dict)],
    }


def _planner_payload(state: AgentState) -> dict[str, Any]:
    return {
        "planner_scope": "kb_qa_only",
        "mode_assumption": "用户已选择知识问答入口；不要判断是否进入诊断链。",
        "target_locale": state.get("locale") or "zh-CN",
        "original_question": state.get("latest_user_text") or "",
        "recent_dialogue": recent_dialogue(state),
        "coref_anchors": _coref_anchors(diagnosis_context(state)),
    }


def _build_messages(state: AgentState) -> list[Any]:
    return _messages_for_payload(_planner_payload(state))


def _messages_for_payload(payload: dict[str, Any]) -> list[Any]:
    question = payload["original_question"]
    payload = {
        **payload,
        "latest_user_text": question,
    }
    return [
        SystemMessage(content=_prompt_text()),
        HumanMessage(
            content="\n".join(
                [
                    "planner_input = " + json.dumps(payload, ensure_ascii=False),
                    "Return one complete KbQueryPlan JSON object now. "
                    "Do not omit any schema field.",
                ]
            )
        ),
    ]


def _invoke_llm(messages: list[Any]) -> KbQueryPlan:
    timeout = llm_timeout_seconds("kb_query_planner")
    llm = deepseek_chat(timeout=timeout)
    method = _structured_method()
    kwargs: dict[str, Any] = {"method": method}
    if method != "json_mode":
        kwargs["strict"] = True
    structured_llm = llm.with_structured_output(KbQueryPlan, **kwargs)
    result = invoke_llm_call(
        error_context_name="kb_query_planner",
        usage_correlation_name="kb_query_planner",
        timeout_seconds=timeout,
        call=lambda: structured_llm.invoke(messages),
    )
    if isinstance(result, KbQueryPlan):
        return result
    return KbQueryPlan.model_validate(result)


def kb_query_planner(state: AgentState) -> AgentState:
    """Classify KB question topology and write only KB-scoped planning fields."""
    payload = _planner_payload(state)
    messages = _messages_for_payload(payload)
    plan = _invoke_llm(messages)
    effective_type, gated = _gated_plan_type(plan)
    trace = _planning_trace(plan, effective_type, gated)
    audit = _planner_audit(payload, messages, plan, trace, gated)
    return _planner_update(plan, effective_type, trace, audit)


def _planner_audit(
    payload: dict[str, Any],
    messages: list[Any],
    plan: KbQueryPlan,
    trace: dict[str, Any],
    gated: bool,
) -> list[Any]:
    audit: list[Any] = [
        trace_event(
            node_name="kb_query_planner",
            kind="llm_call",
            title="KB query plan",
            inputs=payload,
            prompt=messages,
            output=plan,
        ),
    ]
    if gated:
        audit.append(
            AuditEvent(
                node_name="kb_query_planner",
                message="Downgraded low-confidence KB plan to single_hop.",
                metadata={
                    "original_plan_type": plan.plan_type,
                    "confidence": plan.confidence,
                    "threshold": _min_confidence(),
                },
            )
        )
    audit.append(
        AuditEvent(
            node_name="kb_query_planner",
            message="Planned the knowledge-base retrieval topology.",
            metadata=trace,
        )
    )
    return audit


def _planner_update(
    plan: KbQueryPlan,
    effective_type: str,
    trace: dict[str, Any],
    audit: list[Any],
) -> AgentState:
    update: AgentState = {
        "kb_query_plan_v2": plan,
        "kb_plan_type": effective_type,
        "kb_subquestions": plan.subquestions,
        "kb_hops": plan.hops,
        "kb_missing_slots": plan.missing_user_slots,
        "kb_clarification_request": plan.clarification,
        "kb_planning_trace": trace,
        "phase": WorkflowPhase.EVIDENCE,
        "next_action": "route_kb_plan",
        "audit_log": audit,
    }
    if effective_type == "single_hop":
        query_plan = _single_hop_query_plan(plan)
        update["kb_query_plan"] = query_plan
        update["kb_rewritten_query"] = query_plan.queries[0]
    if _has_explicit_user_gap(plan):
        observation = observe_kb_plan(plan)
        update["evidence_assessment"] = observation.assessment
        update["audit_log"] = [
            *audit,
            evidence_observation_event(
                node_name="kb_query_planner",
                observation=observation,
            ),
        ]
    return update


def _has_explicit_user_gap(plan: KbQueryPlan) -> bool:
    clarification = plan.clarification
    return bool(
        plan.missing_user_slots or (clarification is not None and clarification.missing_user_slots)
    )


def _single_hop_query_plan(plan: KbQueryPlan) -> QueryPlan:
    return QueryPlan(queries=[plan.rewritten_main_query], filters={})


def _planning_trace(
    plan: KbQueryPlan,
    effective_type: str | None = None,
    gated: bool = False,
) -> dict[str, Any]:
    return {
        "plan_type": plan.plan_type,
        "effective_plan_type": effective_type or plan.plan_type,
        "confidence_gated": gated,
        "answer_mode": plan.answer_mode,
        "hop_count": len(plan.hops),
        "subquestion_count": len(plan.subquestions),
        "confidence": plan.confidence,
        "decision_reason": plan.decision_reason,
    }
