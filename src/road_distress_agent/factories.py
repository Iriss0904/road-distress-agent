"""Factory helpers for initial state and user-turn patches."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import AnyMessage, HumanMessage

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.localization import DEFAULT_LOCALE, Locale, normalize_locale
from road_distress_agent.state import AgentState, AttachmentRef, AuditEvent, CandidateSelection
from road_distress_agent.tracing import trace_event


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_input_trace(
    text: str | None,
    attachments: list[AttachmentRef],
    request_id: str | None,
) -> AuditEvent:
    return trace_event(
        node_name="user_input",
        kind="user_input",
        title="User input",
        inputs={"text": text, "attachments": attachments, "request_id": request_id},
    )


def make_initial_state(
    *,
    user_id: str,
    latest_user_text: str | None = None,
    thread_id: str | None = None,
    session_id: str | None = None,
    locale: str | None = DEFAULT_LOCALE,
    latest_attachments: list[AttachmentRef] | None = None,
    request_id: str | None = None,
) -> AgentState:
    now = utc_now_iso()
    active_request_id = request_id or f"request-{uuid4().hex}"
    active_locale: Locale = normalize_locale(locale)
    attachments = latest_attachments or []
    messages: list[AnyMessage] = (
        [HumanMessage(content=latest_user_text)] if latest_user_text else []
    )
    return {
        "request_id": active_request_id,
        "thread_id": thread_id or f"thread-{uuid4().hex[:8]}",
        "user_id": user_id,
        "session_id": session_id,
        "locale": active_locale,
        "created_at": now,
        "updated_at": now,
        "messages": messages,
        "latest_user_text": latest_user_text,
        "latest_attachments": attachments,
        "new_user_input_pending": latest_user_text is not None or bool(attachments),
        "scene_description": None,
        "scene_context": None,
        "material": None,
        "defect_category": None,
        "defect_subtype": None,
        "known_features": {},
        "raw_user_text": latest_user_text,
        "distress": None,
        "loaded_memory": None,
        "address_context": None,
        "weather_context": None,
        "weather_advice": None,
        "weather_route": None,
        "construction_tip_offered": False,
        "road_class": None,
        "inferred_work_duration": None,
        "safety_query": None,
        "safety_norm_chunks": [],
        "query_plan": None,
        "method_evidence_bundles": {},
        "discriminator_output": None,
        "needed_fields": [],
        "clarification_attempts": 0,
        "user_intent": None,
        "guard_decision": None,
        "standalone_query_plan": None,
        "retrieval_targets": [],
        "retrieval_strategy": None,
        "speculative_prefetch": None,
        "top_route": None,
        "direct_message": None,
        "refusal_type": None,
        "reconcile_result": None,
        "kb_query_plan": None,
        "kb_query_plan_v2": None,
        "kb_plan_type": None,
        "kb_subquestions": [],
        "kb_hops": [],
        "kb_hop_results": [],
        "kb_evidence_slots": {},
        "kb_missing_slots": [],
        "kb_clarification_request": None,
        "kb_planning_trace": None,
        "kb_final_cited_chunk_ids": [],
        "kb_rewritten_query": None,
        "kb_retrieved_chunks": [],
        "kb_answer": None,
        "evidence_assessment": None,
        "solution_candidates": [],
        "chosen_method": None,
        "candidate_selection": CandidateSelection(),
        "detail_query_plan": None,
        "detail_evidence_bundles": {},
        "final_answer": None,
        "final_answer_message": None,
        "final_answer_display": None,
        "reference_index": [],
        "procedure_chunks": [],
        "acceptance_chunks": [],
        "disease_evidence_chunks": [],
        "method_evidence_chunks": [],
        "retrieved_chunks": [],
        "expanded_chunks": [],
        "safety_review": None,
        "phase": WorkflowPhase.INPUT,
        "awaiting_user_input": False,
        "next_action": None,
        "interrupt": None,
        "interrupts": [],
        "rag_route": None,
        "critic_retry_count": 0,
        "field_history": [],
        "audit_log": (
            [_user_input_trace(latest_user_text, attachments, active_request_id)]
            if latest_user_text is not None or attachments
            else []
        ),
        "retrieval_attempts": [],
        "errors": [],
    }


def begin_user_turn_patch(
    *,
    text: str | None = None,
    locale: str | None = DEFAULT_LOCALE,
    attachments: list[AttachmentRef] | None = None,
    request_id: str | None = None,
) -> AgentState:
    active_locale: Locale = normalize_locale(locale)
    active_request_id = request_id or f"request-{uuid4().hex}"
    active_attachments = attachments or []
    messages: list[AnyMessage] = [HumanMessage(content=text)] if text else []
    return {
        "request_id": active_request_id,
        "updated_at": utc_now_iso(),
        "messages": messages,
        "locale": active_locale,
        "latest_user_text": text,
        "latest_attachments": active_attachments,
        "scene_context": None,
        "scene_description": None,
        "direct_message": None,
        "refusal_type": None,
        "top_route": None,
        "reconcile_result": None,
        "kb_query_plan": None,
        "kb_query_plan_v2": None,
        "kb_plan_type": None,
        "kb_subquestions": [],
        "kb_hops": [],
        "kb_hop_results": [],
        "kb_evidence_slots": {},
        "kb_missing_slots": [],
        "kb_clarification_request": None,
        "kb_planning_trace": None,
        "kb_final_cited_chunk_ids": [],
        "kb_rewritten_query": None,
        "kb_retrieved_chunks": [],
        "kb_answer": None,
        "evidence_assessment": None,
        "final_answer_display": None,
        "new_user_input_pending": text is not None or bool(active_attachments),
        "awaiting_user_input": False,
        "guard_decision": None,
        "standalone_query_plan": None,
        "retrieval_targets": [],
        "retrieval_strategy": None,
        "audit_log": (
            [_user_input_trace(text, active_attachments, active_request_id)]
            if text is not None or active_attachments
            else []
        ),
    }
