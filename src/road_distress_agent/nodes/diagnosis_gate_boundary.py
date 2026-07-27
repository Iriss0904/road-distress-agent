"""Deterministic diagnosis composer boundary outcomes and evidence projection."""

from __future__ import annotations

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.state import AgentState, AuditEvent, FinalAnswer, RetrievedChunk

DIAGNOSIS_NO_PROCEDURE_TEMPLATE_VERSION = "a-stage.diagnosis.no_procedure.v1"


def active_diagnosis_state(
    state: AgentState,
    usable: tuple[RetrievedChunk, ...],
) -> AgentState:
    return {
        **state,
        "procedure_chunks": _allowed_group(state.get("procedure_chunks") or [], usable),
        "acceptance_chunks": _allowed_group(state.get("acceptance_chunks") or [], usable),
    }


def no_procedure_outcome(state: AgentState, decision_trace: AuditEvent) -> AgentState:
    message = (
        "No directly supporting construction-procedure evidence was found, so a procedure "
        "cannot be generated."
        if state.get("locale") == "en-US"
        else "未找到可直接支持施工步骤的知识库证据，因此无法生成处治流程。"
    )
    return {
        "final_answer": FinalAnswer(summary=message, need_human_review=True),
        "phase": WorkflowPhase.SAFETY,
        "errors": [],
        "awaiting_user_input": False,
        "interrupt": None,
        "audit_log": [
            decision_trace,
            AuditEvent(
                node_name="answer_composer",
                message="Returned fixed missing-procedure evidence outcome.",
                metadata={"template_version": DIAGNOSIS_NO_PROCEDURE_TEMPLATE_VERSION},
            ),
        ],
    }


def _allowed_group(
    raw: list[object],
    usable: tuple[RetrievedChunk, ...],
) -> list[RetrievedChunk]:
    values = [
        item if isinstance(item, RetrievedChunk) else RetrievedChunk.model_validate(item)
        for item in raw
    ]
    usable_ids = {_required_chunk_id(item) for item in usable}
    return [item for item in values if _required_chunk_id(item) in usable_ids]


def _required_chunk_id(chunk: RetrievedChunk) -> str:
    chunk_id = chunk.chunk_id or chunk.citation_id
    if not chunk_id:
        raise ValueError("diagnosis gate requires stable chunk ids")
    return chunk_id
