"""Compose answers for planned KB retrieval paths."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.evidence_assessment import EvidenceStatus
from road_distress_agent.evidence_gate import (
    RUNTIME_R1_GATE_POLICY,
    PlannedKbGateInput,
    planned_kb_evidence_gate,
)
from road_distress_agent.llm_deepseek import deepseek_chat
from road_distress_agent.llm_runtime import invoke_llm_call, llm_timeout_seconds
from road_distress_agent.localization import user_language_instruction
from road_distress_agent.memory_context import expression_memory_block
from road_distress_agent.nodes.evidence_gate_runtime import (
    assessment,
    gate_trace,
    planned_slot_inputs,
    raise_on_error,
    unsupported_slots,
)
from road_distress_agent.nodes.kb_answer_composer import KbAnswer
from road_distress_agent.nodes.kb_evidence_boundary import (
    ComposerEvidence,
    UnsupportedSlot,
    composer_evidence,
    ensure_citations_allowed,
)
from road_distress_agent.nodes.kb_fixed_outcome import fixed_kb_outcome_for_status
from road_distress_agent.nodes.kb_planned_composer_projection import (
    chunks as _chunks,
)
from road_distress_agent.nodes.kb_planned_composer_projection import (
    evidence_by_hop as _evidence_by_hop,
)
from road_distress_agent.nodes.kb_planned_composer_projection import (
    evidence_groups as _evidence_groups,
)
from road_distress_agent.nodes.kb_planned_composer_projection import (
    plan as _plan,
)
from road_distress_agent.nodes.kb_planned_composer_projection import (
    references as _references,
)
from road_distress_agent.nodes.kb_planning_utils import (
    evidence_text,
    references_json,
    refs_by_chunk_id,
)
from road_distress_agent.reference_index import build_reference_index
from road_distress_agent.state import (
    AgentState,
    AuditEvent,
    Citation,
    ReferenceItem,
    RetrievedChunk,
)
from road_distress_agent.tracing import trace_event

NODE_NAME = "kb_planned_answer_composer"


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    return (
        Path(__file__).resolve().parents[1] / "prompts" / "kb_planned_answer_composer.txt"
    ).read_text(encoding="utf-8")


def _structured_method() -> str:
    return os.environ.get("DEEPSEEK_STRUCTURED_METHOD") or "function_calling"


def _build_messages(
    state: AgentState,
    boundary: ComposerEvidence | None = None,
) -> list[Any]:
    plan = _plan(state)
    active = boundary or _composer_evidence(state)
    references = list(active.references)
    chunks = list(active.chunks)
    refs_by_chunk = refs_by_chunk_id(references)
    payload = {
        "plan_type": plan.plan_type,
        "answer_mode": plan.answer_mode,
        "subquestions": [item.model_dump(mode="json") for item in plan.subquestions],
        "hops": [item.model_dump(mode="json") for item in plan.hops],
        "missing_slots": state.get("kb_missing_slots") or [],
        "evidence_by_hop": _evidence_by_hop(state, refs_by_chunk),
        "evidence_groups": _evidence_groups(state, plan.plan_type, refs_by_chunk),
    }
    if active.unsupported_slots:
        payload["unsupported_slots"] = active.unsupported_payload()
    return [
        SystemMessage(content=_prompt_text()),
        HumanMessage(content=_human_prompt(state, payload, chunks, references=references)),
    ]


def _human_prompt(
    state: AgentState,
    payload: dict[str, Any],
    chunks: list[RetrievedChunk],
    *,
    references: list[ReferenceItem],
) -> str:
    return "\n".join(
        [
            f"target_locale = {(state.get('locale') or 'zh-CN')!r}",
            user_language_instruction(state.get("locale")),
            f"original_question = {(state.get('latest_user_text') or '')!r}",
            expression_memory_block(state.get("loaded_memory")),
            "plan = " + json.dumps(payload, ensure_ascii=False),
            f"reference_index = {references_json(references)}",
            f"evidence =\n{evidence_text(chunks, references)}",
            "Every cited_chunk_ids entry must have its matching [[R...]] token in answer.",
            "Return the KbAnswer JSON now.",
        ]
    )


def _invoke_llm(messages: list[Any]) -> KbAnswer:
    timeout = llm_timeout_seconds("kb_planned_answer_composer")
    llm = deepseek_chat(timeout=timeout)
    method = _structured_method()
    kwargs: dict[str, Any] = {"method": method}
    if method != "json_mode":
        kwargs["strict"] = True
    structured_llm = llm.with_structured_output(KbAnswer, **kwargs)
    result = invoke_llm_call(
        error_context_name="kb_planned_answer_composer",
        usage_correlation_name="kb_planned_answer_composer",
        timeout_seconds=timeout,
        call=lambda: structured_llm.invoke(messages),
    )
    if isinstance(result, KbAnswer):
        return result
    return KbAnswer.model_validate(result)


def kb_planned_answer_composer(state: AgentState) -> AgentState:
    """Generate a planned KB answer without modifying diagnosis state."""
    policy = RUNTIME_R1_GATE_POLICY
    current_assessment = assessment(state)
    hard_statuses = {
        EvidenceStatus.DEPENDENCY_FAILURE,
        EvidenceStatus.OUT_OF_SCOPE,
        EvidenceStatus.MISSING_USER_CONTEXT,
        EvidenceStatus.CONFLICTING,
        EvidenceStatus.NO_KB_EVIDENCE,
    }
    slots = (
        ()
        if current_assessment is not None and current_assessment.status in hard_statuses
        else planned_slot_inputs(state, _plan(state))
    )
    decision = planned_kb_evidence_gate(
        PlannedKbGateInput(assessment=current_assessment, slots=slots), policy
    )
    decision_trace = gate_trace(NODE_NAME, decision, policy)
    raise_on_error(decision, current_assessment)
    if decision.decision == "REFUSE":
        fixed = fixed_kb_outcome_for_status(state, NODE_NAME, decision.reason)
        if fixed is None:
            raise ValueError(f"unsupported KB gate refusal: {decision.reason.value}")
        fixed["audit_log"] = [decision_trace, *(fixed.get("audit_log") or [])]
        return fixed
    boundary = _gate_boundary(decision.usable_chunks, slots, decision.passed_slots)
    chunks = list(boundary.chunks)
    references = list(boundary.references)
    messages = _build_messages(state, boundary)
    answer = _invoke_llm(messages)
    llm_trace = _llm_trace(state, messages, answer, reference_count=len(references))
    _ensure_citations(answer, chunks, references)
    return {
        "direct_message": answer.answer,
        "refusal_type": None,
        "kb_answer": answer.model_dump(mode="json"),
        "reference_index": references,
        "kb_final_cited_chunk_ids": answer.cited_chunk_ids,
        "awaiting_user_input": state.get("interrupt") is not None,
        "phase": WorkflowPhase.DONE,
        "next_action": "return_kb_answer",
        "audit_log": _audit_log(
            answer,
            llm_trace,
            decision_trace=decision_trace,
            references=references,
            unsupported_slots=boundary.unsupported_payload(),
        ),
    }


def _gate_boundary(
    usable: tuple[RetrievedChunk, ...],
    slots,
    passed_slots: tuple[str, ...],
) -> ComposerEvidence:
    references = build_reference_index(
        [Citation.model_validate(chunk.model_dump(mode="json")) for chunk in usable]
    )
    missing = unsupported_slots(slots, passed_slots)
    unsupported = tuple(
        UnsupportedSlot(slot_id=slot_id, reason_code="evidence_gate_slot_failed")
        for slot_id in missing
    )
    return ComposerEvidence(usable, tuple(references), unsupported)


def _composer_evidence(state: AgentState) -> ComposerEvidence:
    return composer_evidence(state, _chunks(state), _references(state))


def _ensure_citations(
    answer: KbAnswer,
    chunks: list[RetrievedChunk],
    references: list[ReferenceItem],
) -> None:
    if chunks and not answer.cited_chunk_ids:
        raise ValueError("kb_planned_answer_composer returned no cited_chunk_ids.")
    ensure_citations_allowed(answer.cited_chunk_ids, references, NODE_NAME)
    missing_tokens = _missing_inline_tokens(answer, references)
    if missing_tokens:
        raise ValueError(
            f"kb_planned_answer_composer omitted inline ref tokens: {missing_tokens!r}."
        )


def _missing_inline_tokens(answer: KbAnswer, references: list[ReferenceItem]) -> list[str]:
    refs_by_chunk = refs_by_chunk_id(references)
    ref_ids = {
        refs_by_chunk[chunk_id] for chunk_id in answer.cited_chunk_ids if chunk_id in refs_by_chunk
    }
    return [ref_id for ref_id in sorted(ref_ids) if f"[[{ref_id}]]" not in answer.answer]


def _llm_trace(
    state: AgentState,
    messages: list[Any],
    answer: KbAnswer,
    *,
    reference_count: int,
) -> AuditEvent:
    return trace_event(
        node_name=NODE_NAME,
        kind="llm_call",
        title="KB planned answer composer LLM",
        inputs={
            "latest_user_text": state.get("latest_user_text"),
            "kb_plan_type": state.get("kb_plan_type"),
            "reference_count": reference_count,
        },
        prompt=messages,
        output=answer,
    )


def _audit_log(
    answer: KbAnswer,
    llm_trace: AuditEvent,
    *,
    decision_trace: AuditEvent,
    references: list[ReferenceItem],
    unsupported_slots: list[dict[str, str]],
) -> list[AuditEvent]:
    metadata: dict[str, Any] = {
        "cited_chunk_ids": answer.cited_chunk_ids,
        "reference_count": len(references),
    }
    if unsupported_slots:
        metadata["unsupported_slots"] = unsupported_slots
    return [
        decision_trace,
        llm_trace,
        AuditEvent(
            node_name=NODE_NAME,
            message="Composed planned knowledge-base answer.",
            metadata=metadata,
        ),
    ]
