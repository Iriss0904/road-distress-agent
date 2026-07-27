"""Deterministic KB outcomes for evidence boundaries."""

from __future__ import annotations

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.evidence_assessment import EvidenceAssessment, EvidenceStatus
from road_distress_agent.retrieval.query_features import detect_query_features
from road_distress_agent.state import AgentState, AuditEvent
from road_distress_agent.tracing import trace_event

NO_KB_TEMPLATE_VERSION = "a1.no_kb.v1"
CONFLICT_TEMPLATE_VERSION = "a2.conflict.v1"


def fixed_kb_outcome(state: AgentState, node_name: str) -> AgentState | None:
    """Return a fixed evidence-boundary result, or continue composing."""
    assessment = _assessment(state.get("evidence_assessment"))
    if assessment is None:
        return None
    if assessment.status is EvidenceStatus.DEPENDENCY_FAILURE:
        raise RuntimeError("dependency_failure cannot enter a KB answer composer")
    return fixed_kb_outcome_for_status(state, node_name, assessment.status)


def fixed_kb_outcome_for_status(
    state: AgentState,
    node_name: str,
    status: EvidenceStatus,
) -> AgentState | None:
    """Return the existing deterministic outcome for a gate refusal status."""
    if status is EvidenceStatus.CONFLICTING:
        return _conflicting_outcome(state, node_name)
    if status is not EvidenceStatus.NO_KB_EVIDENCE:
        return None
    return _no_kb_outcome(state, node_name)


def _no_kb_outcome(state: AgentState, node_name: str) -> AgentState:
    message = _no_kb_message(state)
    output = {
        "decision": "fixed_no_kb",
        "evidence_status": EvidenceStatus.NO_KB_EVIDENCE.value,
        "refusal_type": "no_kb_evidence",
        "cited_chunk_ids": [],
    }
    return {
        "direct_message": message,
        "refusal_type": "no_kb_evidence",
        "kb_answer": {"answer": message, "cited_chunk_ids": []},
        "reference_index": [],
        "kb_final_cited_chunk_ids": [],
        "awaiting_user_input": state.get("interrupt") is not None,
        "phase": WorkflowPhase.DONE,
        "next_action": "return_kb_answer",
        "audit_log": [
            trace_event(
                node_name=node_name,
                kind="evidence_boundary",
                title="Fixed no-KB evidence outcome",
                output=output,
                metadata={
                    "affects_runtime_behavior": True,
                    "template_version": NO_KB_TEMPLATE_VERSION,
                },
            ),
            AuditEvent(
                node_name=node_name,
                message="Returned fixed no-KB evidence outcome.",
                metadata=output,
            ),
        ],
    }


def _conflicting_outcome(state: AgentState, node_name: str) -> AgentState:
    message = _conflicting_message(state)
    output = {
        "decision": "fixed_conflicting_evidence",
        "evidence_status": EvidenceStatus.CONFLICTING.value,
        "refusal_type": None,
        "cited_chunk_ids": [],
    }
    return {
        "direct_message": message,
        "refusal_type": None,
        "kb_answer": {"answer": message, "cited_chunk_ids": []},
        "reference_index": [],
        "kb_final_cited_chunk_ids": [],
        "awaiting_user_input": state.get("interrupt") is not None,
        "phase": WorkflowPhase.DONE,
        "next_action": "return_kb_answer",
        "audit_log": [
            trace_event(
                node_name=node_name,
                kind="evidence_boundary",
                title="Fixed conflicting-evidence outcome",
                output=output,
                metadata={
                    "affects_runtime_behavior": True,
                    "template_version": CONFLICT_TEMPLATE_VERSION,
                },
            ),
            AuditEvent(
                node_name=node_name,
                message="Returned fixed conflicting-evidence outcome.",
                metadata=output,
            ),
        ],
    }


def _assessment(value: object) -> EvidenceAssessment | None:
    if value is None:
        return None
    if isinstance(value, EvidenceAssessment):
        return value
    return EvidenceAssessment.model_validate(value)


def _no_kb_message(state: AgentState) -> str:
    clause_id = _single_clause_id(state.get("latest_user_text"))
    if state.get("locale") == "en-US":
        if clause_id:
            return (
                f"The knowledge base has no direct evidence for Clause {clause_id}, so I "
                "cannot answer from that clause or substitute a neighboring clause."
            )
        return (
            "The knowledge-base search completed but found no direct supporting evidence, "
            "so I cannot answer or substitute unrelated evidence."
        )
    if clause_id:
        return (
            f"当前知识库中未找到第{clause_id}条的直接证据，因此不能根据该条回答，"
            "也不会使用邻近条款替代。"
        )
    return "知识库检索已完成，但未找到能直接支持该问题的证据，因此无法作答。"


def _conflicting_message(state: AgentState) -> str:
    if state.get("locale") == "en-US":
        return (
            "The available evidence has conflicting applicability conditions. "
            "I cannot select a single conclusion automatically; please request human review."
        )
    return "当前证据存在适用条件差异，无法自动选择唯一结论，请由人工复核。"


def _single_clause_id(text: str | None) -> str | None:
    clause_ids = detect_query_features(text or "", {}).clause_ids
    return clause_ids[0] if len(clause_ids) == 1 else None
