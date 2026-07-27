"""Extract and persist long-term memory after terminal workflow paths."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.llm_deepseek import deepseek_chat
from road_distress_agent.llm_runtime import invoke_llm_call, llm_timeout_seconds
from road_distress_agent.memory_eligibility import (
    KBMemoryEligibility,
    assess_kb_memory_eligibility,
)
from road_distress_agent.state import AgentState, AuditEvent, FinalAnswer
from road_distress_agent.tools.memory import make_memory_tool
from road_distress_agent.tools.memory_idempotency import MemoryWriteCommand, MemoryWriteResult
from road_distress_agent.tracing import trace_event


class MemoryPatch(BaseModel):
    user_preferences: dict[str, str] = Field(default_factory=dict)
    regional_context: dict[str, str] = Field(default_factory=dict)
    resource_constraints: dict[str, str] = Field(default_factory=dict)
    case_summary: str | None = None

    @field_validator("user_preferences", "regional_context", "resource_constraints", mode="before")
    @classmethod
    def _none_to_dict(cls, value: Any) -> Any:
        return {} if value is None else value


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    return (Path(__file__).resolve().parents[1] / "prompts" / "memory_writer.txt").read_text(
        encoding="utf-8"
    )


def _structured_method() -> str:
    return os.environ.get("DEEPSEEK_STRUCTURED_METHOD") or "json_mode"


def _route_action(state: AgentState) -> str | None:
    route = state.get("top_route") or {}
    action = route.get("action")
    return str(action) if action else None


def _should_extract(action: str | None) -> bool:
    return action in {"diagnosis_proceed", "kb_aside"}


def _case_summary_allowed(state: AgentState, action: str | None) -> bool:
    review = state.get("safety_review")
    return (
        action == "diagnosis_proceed"
        and bool(review and review.passed)
        and isinstance(state.get("final_answer"), FinalAnswer)
    )


def _final_answer_payload(answer: FinalAnswer | str | None) -> Any:
    if isinstance(answer, FinalAnswer):
        return {
            "summary": answer.summary,
            "diagnosis": answer.diagnosis,
            "method_selection": answer.method_selection,
            "evidence_gaps": answer.evidence_gaps,
            "need_human_review": answer.need_human_review,
            "weather_advice": (
                answer.weather_advice.model_dump(mode="json") if answer.weather_advice else None
            ),
        }
    return answer


def _build_messages(state: AgentState, action: str | None) -> list[Any]:
    payload = {
        "route_action": action,
        "case_summary_allowed": _case_summary_allowed(state, action),
        "latest_user_text": state.get("latest_user_text"),
        "direct_message": state.get("direct_message"),
        "kb_answer": state.get("kb_answer"),
        "final_answer": _final_answer_payload(state.get("final_answer")),
        "thread_id": state.get("thread_id"),
    }
    return [
        SystemMessage(content=_prompt_text()),
        HumanMessage(content=f"memory_extraction_input = {payload}"),
    ]


def _invoke_llm(messages: list[Any]) -> MemoryPatch:
    timeout = llm_timeout_seconds("memory_writer")
    llm = deepseek_chat(timeout=timeout)
    method = _structured_method()
    kwargs: dict[str, Any] = {"method": method}
    if method != "json_mode":
        kwargs["strict"] = True
    structured_llm = llm.with_structured_output(MemoryPatch, **kwargs)
    result = invoke_llm_call(
        error_context_name="memory_writer",
        usage_correlation_name="memory_writer",
        timeout_seconds=timeout,
        call=lambda: structured_llm.invoke(messages),
    )
    if isinstance(result, MemoryPatch):
        return result
    return MemoryPatch.model_validate(result)


def _skip_result(message: str) -> AgentState:
    return {
        "phase": WorkflowPhase.DONE,
        "next_action": "return_final",
        "awaiting_user_input": False,
        "audit_log": [AuditEvent(node_name="memory_writer", message=message)],
    }


def _kb_skip_result(decision: KBMemoryEligibility) -> AgentState:
    return {
        "phase": WorkflowPhase.DONE,
        "next_action": "return_final",
        "awaiting_user_input": False,
        "audit_log": [
            AuditEvent(
                node_name="memory_writer",
                message="Skipped KB memory extraction by deterministic eligibility.",
                metadata=_eligibility_metadata(decision),
            )
        ],
    }


def _eligibility_metadata(decision: KBMemoryEligibility) -> dict[str, Any]:
    return {
        "eligibility": decision.eligible,
        "eligibility_reason": decision.reason,
        "eligibility_category_signals": list(decision.category_signals),
    }


def _result_with_patch(
    *,
    state: AgentState,
    action: str | None,
    messages: list[Any],
    patch: MemoryPatch,
    write_result: MemoryWriteResult,
    eligibility: KBMemoryEligibility | None,
) -> AgentState:
    metadata = {
        **(_eligibility_metadata(eligibility) if eligibility else {}),
        "wrote_updates": bool(write_result.applied_types),
        "user_preference_count": len(patch.user_preferences),
        "regional_context_count": len(patch.regional_context),
        "resource_constraint_count": len(patch.resource_constraints),
        "case_summary_written": bool(patch.case_summary),
        "request_id": state["request_id"],
        "side_effect_skipped": bool(write_result.skipped_types),
        "applied_memory_types": list(write_result.applied_types),
        "skipped_memory_types": list(write_result.skipped_types),
    }
    if write_result.skipped_types:
        metadata["resume_point"] = "after_memory_write"
    return {
        "loaded_memory": write_result.memory,
        "phase": WorkflowPhase.DONE,
        "next_action": "return_final",
        "awaiting_user_input": False,
        "audit_log": [
            trace_event(
                node_name="memory_writer",
                kind="llm_call",
                title="Long-term memory extraction",
                inputs={
                    "request_id": state["request_id"],
                    "route_action": action,
                    "case_summary_allowed": _case_summary_allowed(state, action),
                },
                prompt=messages,
                output=patch,
                metadata={
                    "request_id": state["request_id"],
                    "resume_point": metadata.get("resume_point"),
                    "side_effect_skipped": metadata["side_effect_skipped"],
                },
            ),
            AuditEvent(
                node_name="memory_writer",
                message="Extracted and persisted long-term memory updates.",
                metadata=metadata,
            ),
        ],
    }


def memory_writer(state: AgentState) -> AgentState:
    request_id = state.get("request_id")
    if not request_id or not request_id.strip():
        raise ValueError("request_id is required by memory_writer.")
    action = _route_action(state)
    if not _should_extract(action):
        return _skip_result("Skipped memory extraction for this route.")
    eligibility = None
    if action == "kb_aside":
        eligibility = assess_kb_memory_eligibility(state.get("latest_user_text"))
        if not eligibility.eligible:
            return _kb_skip_result(eligibility)

    messages = _build_messages(state, action)
    patch = _invoke_llm(messages)
    if not _case_summary_allowed(state, action):
        patch = patch.model_copy(update={"case_summary": None})

    write_result = make_memory_tool().apply_once(
        MemoryWriteCommand(
            user_id=state["user_id"],
            request_id=request_id,
            thread_id=state.get("thread_id"),
            user_preferences=patch.user_preferences,
            regional_context=patch.regional_context,
            resource_constraints=patch.resource_constraints,
            case_summary=patch.case_summary,
        )
    )
    result = _result_with_patch(
        state=state,
        action=action,
        messages=messages,
        patch=patch,
        write_result=write_result,
        eligibility=eligibility,
    )
    if write_result.applied_types:
        _after_memory_commit(request_id)
    return result


def _after_memory_commit(request_id: str) -> None:
    """Explicit fault-injection seam after commit and before node checkpoint."""
