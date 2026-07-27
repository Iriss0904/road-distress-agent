"""Rewrite a safety-norm retrieval query for construction arrangement advice."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.llm_deepseek import deepseek_chat
from road_distress_agent.llm_runtime import invoke_llm_call, llm_timeout_seconds
from road_distress_agent.localization import query_language_instruction
from road_distress_agent.state import AgentState, AuditEvent, RetrievedChunk
from road_distress_agent.tracing import trace_event


class SafetyNormQueryPlan(BaseModel):
    inferred_work_duration: Literal["移动", "短时", "短期", "长期"]
    safety_query: str = Field(min_length=1)


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    path = Path(__file__).resolve().parents[1] / "prompts" / "safety_norm_rewriter.txt"
    return path.read_text(encoding="utf-8")


def _structured_method() -> str:
    return os.environ.get("DEEPSEEK_STRUCTURED_METHOD") or "json_mode"


def _method(state: AgentState) -> str:
    if chosen := state.get("chosen_method"):
        return str(chosen)
    selection = state.get("candidate_selection")
    if selection and selection.selected_name:
        return selection.selected_name
    return "未确认处治方法"


def _format_chunks(chunks: list[RetrievedChunk], max_body: int) -> str:
    if not chunks:
        return "（无证据）"
    lines: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        clause = f"[{chunk.clause_id}]" if chunk.clause_id else ""
        body = (chunk.text or "").strip()[:max_body]
        lines.append(f"[{index}]{clause} {chunk.context_prefix or ''}\n{body}")
    return "\n\n".join(lines)


def _build_messages(state: AgentState) -> list[Any]:
    road_class = state.get("road_class")
    if not road_class:
        raise ValueError("safety_norm_rewriter requires road_class.")
    return [
        SystemMessage(content=_prompt_text()),
        HumanMessage(
            content="\n".join(
                [
                    query_language_instruction(),
                    f"road_class = {road_class!r}",
                    f"material = {(state.get('material') or '未知')!r}",
                    f"defect_category = {(state.get('defect_category') or '未知')!r}",
                    f"chosen_method = {_method(state)!r}",
                    f"known_features = {state.get('known_features') or {}}",
                    "construction_steps =",
                    _format_chunks(state.get("procedure_chunks") or [], 600),
                    "acceptance_criteria =",
                    _format_chunks(state.get("acceptance_chunks") or [], 600),
                    "Return the SafetyNormQueryPlan JSON now.",
                ]
            )
        ),
    ]


def _invoke_llm(messages: list[Any]) -> SafetyNormQueryPlan:
    timeout = llm_timeout_seconds("safety_norm_rewriter")
    llm = deepseek_chat(timeout=timeout)
    method = _structured_method()
    kwargs: dict[str, Any] = {"method": method}
    if method != "json_mode":
        kwargs["strict"] = True
    structured_llm = llm.with_structured_output(SafetyNormQueryPlan, **kwargs)
    result = invoke_llm_call(
        error_context_name="safety_norm_rewriter",
        usage_correlation_name="safety_norm_rewriter",
        timeout_seconds=timeout,
        call=lambda: structured_llm.invoke(messages),
    )
    if isinstance(result, SafetyNormQueryPlan):
        return result
    return SafetyNormQueryPlan.model_validate(result)


def safety_norm_rewriter(state: AgentState) -> AgentState:
    messages = _build_messages(state)
    plan = _invoke_llm(messages)
    return {
        "inferred_work_duration": plan.inferred_work_duration,
        "safety_query": plan.safety_query,
        "phase": WorkflowPhase.EVIDENCE,
        "next_action": "retrieve_safety_norms",
        "audit_log": [
            trace_event(
                node_name="safety_norm_rewriter",
                kind="llm_call",
                title="Safety norm query rewrite",
                inputs={"road_class": state.get("road_class"), "method": _method(state)},
                prompt=messages,
                output=plan,
            ),
            AuditEvent(
                node_name="safety_norm_rewriter",
                message="Rewrote the safety-norm retrieval query.",
                metadata={
                    "inferred_work_duration": plan.inferred_work_duration,
                    "safety_query": plan.safety_query,
                },
            ),
        ],
    }
