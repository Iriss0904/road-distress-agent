"""Compose a diagnosis answer from procedure and optional acceptance evidence."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.evidence_gate import (
    RUNTIME_R1_GATE_POLICY,
    DiagnosisGateInput,
    diagnosis_evidence_gate,
)
from road_distress_agent.llm_deepseek import deepseek_chat
from road_distress_agent.llm_runtime import invoke_llm_call, llm_timeout_seconds
from road_distress_agent.nodes.diagnosis_gate_boundary import (
    DIAGNOSIS_NO_PROCEDURE_TEMPLATE_VERSION,
    active_diagnosis_state,
    no_procedure_outcome,
)
from road_distress_agent.nodes.evidence_gate_runtime import (
    assessment,
    gate_trace,
    raise_on_error,
)
from road_distress_agent.nodes.evidence_gate_runtime import (
    chunks as runtime_chunks,
)
from road_distress_agent.state import (
    AgentState,
    AuditEvent,
    Citation,
    FinalAnswer,
    RetrievedChunk,
)
from road_distress_agent.tracing import trace_event

NODE_NAME = "answer_composer"


class AcceptanceItem(BaseModel):
    item: str  # 检验项目名称
    requirement: str  # 规定值及允许偏差（精确摘录）
    check_method: str  # 检验方法（精确摘录）


class ComposedAnswer(BaseModel):
    disease_definition: str
    method_rationale: str
    construction_steps: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    acceptance_items: list[AcceptanceItem] = Field(default_factory=list)
    need_human_review: bool = False
    review_reason: str | None = None

    @field_validator("construction_steps", "materials", mode="before")
    @classmethod
    def _coerce_lists(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("acceptance_items", mode="before")
    @classmethod
    def _coerce_items(cls, value: Any) -> Any:
        return [] if value is None else value


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    return (Path(__file__).resolve().parents[1] / "prompts" / "answer_composer.txt").read_text(
        encoding="utf-8"
    )


def _structured_method() -> str:
    return os.environ.get("DEEPSEEK_STRUCTURED_METHOD") or "json_mode"


def _format_chunks(chunks: list[RetrievedChunk], max_body: int | None = 800) -> str:
    if not chunks:
        return "（无证据）"
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        prefix = c.context_prefix or ""
        body = (c.text or "").strip()
        if max_body is not None:
            body = body[:max_body]
        clause = f"[{c.clause_id}]" if c.clause_id else ""
        step_tag = f" step={c.step_index}" if c.step_index is not None else ""
        parts.append(f"[{i}]{clause}{step_tag} {prefix}\n{body}")
    return "\n\n".join(parts)


def _get_method(state: AgentState) -> str:
    if m := state.get("chosen_method"):
        return m
    selection = state.get("candidate_selection")
    if selection and selection.selected_name:
        return selection.selected_name
    return "未确认的处治方法"


def _composer_messages(state: AgentState) -> list[Any]:
    material = state.get("material") or "未知"
    defect = state.get("defect_category") or "未知"
    method = _get_method(state)
    known = state.get("known_features") or {}
    turn_plan = state.get("turn_plan")
    requested = getattr(turn_plan, "requested_outputs", [])
    if isinstance(turn_plan, dict):
        requested = turn_plan.get("requested_outputs", requested)

    procedure_chunks: list[RetrievedChunk] = state.get("procedure_chunks") or []
    acceptance_chunks: list[RetrievedChunk] = state.get("acceptance_chunks") or []
    disease_chunks: list[RetrievedChunk] = state.get("disease_evidence_chunks") or []
    method_chunks: list[RetrievedChunk] = state.get("method_evidence_chunks") or []

    return [
        SystemMessage(content=_prompt_text()),
        HumanMessage(
            content="\n".join(
                [
                    f"路面材质：{material}",
                    f"病害类型：{defect}",
                    f"已选施工方法：{method}",
                    f"已知特征：{known}",
                    f"用户请求输出：{requested}",
                    "",
                    f"【病害定义证据】\n{_format_chunks(disease_chunks, max_body=600)}",
                    "",
                    f"【方法选择证据】\n{_format_chunks(method_chunks, max_body=800)}",
                    "",
                    "【施工步骤证据（已按 step_index 排序）】\n"
                    f"{_format_chunks(procedure_chunks, max_body=None)}",
                    *_acceptance_prompt(acceptance_chunks),
                    "",
                    "按格式输出 JSON。",
                ]
            )
        ),
    ]


def _acceptance_prompt(chunks: list[RetrievedChunk]) -> list[str]:
    if not chunks:
        return ["acceptance evidence is absent; return acceptance_items = []."]
    return [
        "",
        f"【验收标准证据（含验收总表，请按步骤筛选）】\n{_format_chunks(chunks, max_body=1200)}",
    ]


def _invoke_llm(messages: list[Any]) -> ComposedAnswer:
    timeout = llm_timeout_seconds("answer_composer")
    llm = deepseek_chat(timeout=timeout)
    method_str = _structured_method()
    kwargs: dict[str, Any] = {"method": method_str}
    if method_str != "json_mode":
        kwargs["strict"] = True
    structured_llm = llm.with_structured_output(ComposedAnswer, **kwargs)
    result = invoke_llm_call(
        error_context_name="answer_composer",
        usage_correlation_name="answer_composer",
        timeout_seconds=timeout,
        call=lambda: structured_llm.invoke(messages),
    )
    if isinstance(result, ComposedAnswer):
        return result
    return ComposedAnswer.model_validate(result)


def _compose_with_llm(state: AgentState) -> ComposedAnswer:
    return _invoke_llm(_composer_messages(state))


def _all_citations(state: AgentState) -> list[Citation]:
    all_chunks: list[RetrievedChunk] = [
        *(state.get("method_evidence_chunks") or []),
        *(state.get("procedure_chunks") or []),
        *(state.get("acceptance_chunks") or []),
        *(state.get("disease_evidence_chunks") or []),
    ]
    seen: set[str] = set()
    unique: list[Citation] = []
    for c in all_chunks:
        key = c.chunk_id or f"{c.source_doc}:{c.clause_id}"
        if key not in seen:
            seen.add(key)
            unique.append(Citation.model_validate(c.model_dump()))
    return unique[:12]


def _to_final_answer(draft: ComposedAnswer, citations: list[Citation]) -> FinalAnswer:
    ac_lines = [
        f"{a.item}：{a.requirement}（检验方法：{a.check_method}）" for a in draft.acceptance_items
    ]
    summary = _answer_summary(draft)
    return FinalAnswer(
        summary=summary,
        diagnosis=draft.disease_definition,
        method_selection=draft.method_rationale,
        steps=draft.construction_steps,
        materials=draft.materials,
        acceptance_criteria=ac_lines,
        citations=citations,
        need_human_review=draft.need_human_review,
        evidence_gaps=[draft.review_reason] if draft.review_reason else [],
    )


def _answer_summary(draft: ComposedAnswer) -> str:
    definition = draft.disease_definition.rstrip("。.!！?？ ")
    rationale = draft.method_rationale.rstrip("。.!！?？ ")
    return f"{definition}。 采用{rationale}。"


def answer_composer(state: AgentState) -> AgentState:
    """Step E: compose a structured final answer with step-filtered acceptance criteria."""
    policy = RUNTIME_R1_GATE_POLICY
    procedure = runtime_chunks(state.get("procedure_chunks") or ())
    acceptance = runtime_chunks(state.get("acceptance_chunks") or ())
    current_assessment = assessment(state)
    decision = diagnosis_evidence_gate(
        DiagnosisGateInput(current_assessment, procedure, acceptance), policy
    )
    template_version = (
        DIAGNOSIS_NO_PROCEDURE_TEMPLATE_VERSION if decision.decision == "REFUSE" else None
    )
    decision_trace = gate_trace(NODE_NAME, decision, policy, template_version=template_version)
    raise_on_error(decision, current_assessment)
    if decision.decision == "REFUSE":
        return no_procedure_outcome(state, decision_trace)
    active_state = active_diagnosis_state(state, decision.usable_chunks)
    messages = _composer_messages(active_state)
    draft = _invoke_llm(messages)
    if not active_state.get("acceptance_chunks") and draft.acceptance_items:
        raise ValueError("answer_composer returned acceptance_items without acceptance evidence")
    return _answer_patch(active_state, decision_trace, messages=messages, draft=draft)


def _answer_patch(
    state: AgentState,
    decision_trace: AuditEvent,
    *,
    messages: list[Any],
    draft: ComposedAnswer,
) -> AgentState:
    citations = _all_citations(state)
    final_answer = _to_final_answer(draft, citations)
    return {
        "final_answer": final_answer,
        "phase": WorkflowPhase.SAFETY,
        "errors": [],
        # Clear interrupt so route_after_synthesis goes to safety_critic directly.
        "awaiting_user_input": False,
        "interrupt": None,
        "audit_log": [
            decision_trace,
            trace_event(
                node_name=NODE_NAME,
                kind="llm_call",
                title="Answer composer LLM",
                inputs={
                    "material": state.get("material"),
                    "defect_category": state.get("defect_category"),
                    "method": _get_method(state),
                    "known_features": state.get("known_features") or {},
                    "memory_present": state.get("loaded_memory") is not None,
                },
                prompt=messages,
                output={"composed_answer": draft, "final_answer": final_answer},
            ),
            AuditEvent(
                node_name="answer_composer",
                message="Step E: composed final answer with step-filtered acceptance criteria.",
                metadata={
                    "method": _get_method(state),
                    "step_count": len(draft.construction_steps),
                    "material_count": len(draft.materials),
                    "acceptance_item_count": len(draft.acceptance_items),
                    "citation_count": len(citations),
                    "need_human_review": final_answer.need_human_review,
                },
            ),
        ],
    }
