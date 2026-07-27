"""Knowledge-base QA query rewriting."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.llm_deepseek import deepseek_chat
from road_distress_agent.llm_runtime import invoke_llm_call, llm_timeout_seconds
from road_distress_agent.localization import query_language_instruction
from road_distress_agent.state import AgentState, AuditEvent, QueryPlan
from road_distress_agent.tracing import trace_event


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    return (Path(__file__).resolve().parents[1] / "prompts" / "kb_query_rewriter.txt").read_text(
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
                    f"target_locale = {(state.get('locale') or 'zh-CN')!r}",
                    query_language_instruction(),
                    f"user_question = {(state.get('latest_user_text') or '')!r}",
                    "Return the QueryPlan JSON now.",
                ]
            )
        ),
    ]


def _invoke_llm(messages: list[Any]) -> QueryPlan:
    timeout = llm_timeout_seconds("kb_query_rewriter")
    llm = deepseek_chat(correlation_name="kb_query_rewriter", timeout=timeout)
    method = _structured_method()
    kwargs: dict[str, Any] = {"method": method}
    if method != "json_mode":
        kwargs["strict"] = True
    structured_llm = llm.with_structured_output(QueryPlan, **kwargs)
    result = invoke_llm_call(
        error_context_name="kb_query_rewriter",
        usage_correlation_name="kb_query_rewriter",
        timeout_seconds=timeout,
        call=lambda: structured_llm.invoke(messages),
    )
    if isinstance(result, QueryPlan):
        return result
    return QueryPlan.model_validate(result)


def _effective_plan(plan: QueryPlan) -> QueryPlan:
    if not plan.queries:
        raise ValueError("kb_query_rewriter LLM returned no queries.")
    return plan.model_copy(update={"queries": [plan.queries[0]], "filters": {}})


def kb_query_rewriter(state: AgentState) -> AgentState:
    """Rewrite a pure knowledge question without touching diagnosis state."""
    messages = _build_messages(state)
    plan = _effective_plan(_invoke_llm(messages))
    return {
        "kb_query_plan": plan,
        "kb_rewritten_query": plan.queries[0],
        "phase": WorkflowPhase.EVIDENCE,
        "next_action": "retrieve_kb_evidence",
        "audit_log": [
            trace_event(
                node_name="kb_query_rewriter",
                kind="llm_call",
                title="KB query rewrite",
                inputs={"latest_user_text": state.get("latest_user_text")},
                prompt=messages,
                output=plan,
            ),
            AuditEvent(
                node_name="kb_query_rewriter",
                message="Rewrote the knowledge-base question.",
                metadata={"query": plan.queries[0], "filters": plan.filters},
            ),
        ],
    }
