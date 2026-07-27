"""Execute bounded KB query-plan hops with shared retrieval."""

from __future__ import annotations

from functools import partial

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.evidence_assessment import EvidenceStatus
from road_distress_agent.evidence_observation import (
    evidence_observation_event,
    observe_planned_kb,
)
from road_distress_agent.kb_hop_execution import (
    kb_hop_parallel_enabled,
    run_kb_hop_calls,
)
from road_distress_agent.nodes.kb_planning_utils import budgeted_evidence
from road_distress_agent.retrieval.evidence import EvidenceRetrievalOptions, retrieve_evidence
from road_distress_agent.retrieval.query_features import detect_query_features
from road_distress_agent.state import (
    MAX_KB_PLAN_HOPS,
    AgentState,
    AuditEvent,
    KbHop,
    KbHopResult,
    KbQueryPlan,
    RetrievalAttempt,
    RetrievedChunk,
)

COMPOSER_BOUNDARY_STATUSES = frozenset(
    {
        EvidenceStatus.NO_KB_EVIDENCE,
        EvidenceStatus.PARTIAL,
        EvidenceStatus.CONFLICTING,
    }
)


def kb_hop_retriever(state: AgentState) -> AgentState:
    """Retrieve planned KB evidence; failures from retrieval are intentionally exposed."""
    plan = _plan(state)
    original_user_text = state.get("latest_user_text")
    explicit_clause_ids = detect_query_features(original_user_text or "", {}).clause_ids
    results, attempts, audit_log = _retrieve_hops(plan, explicit_clause_ids)
    budgeted = budgeted_evidence(plan.plan_type, plan.hops, results)
    missing = _missing_slots(plan, results)
    evidence_slots = _evidence_slots(plan, results, budgeted.chunks)
    observation = observe_planned_kb(
        plan,
        budgeted.chunks,
        evidence_slots,
        original_user_text=original_user_text,
        retrieval_attempts=attempts,
    )
    return {
        "kb_hop_results": results,
        "kb_evidence_slots": evidence_slots,
        "kb_missing_slots": missing,
        "kb_retrieved_chunks": budgeted.chunks,
        "evidence_assessment": observation.assessment,
        "reference_index": budgeted.references,
        "retrieval_attempts": attempts,
        "kb_planning_trace": _planning_trace(state, results, missing),
        "phase": WorkflowPhase.EVIDENCE,
        "next_action": "compose_planned_answer",
        "audit_log": [
            *audit_log,
            evidence_observation_event(
                node_name="kb_hop_retriever",
                observation=observation,
                affects_runtime_behavior=_affects_composer(observation.assessment),
            ),
        ],
    }


def _affects_composer(assessment: object | None) -> bool:
    return getattr(assessment, "status", None) in COMPOSER_BOUNDARY_STATUSES


def _plan(state: AgentState) -> KbQueryPlan:
    raw = state.get("kb_query_plan_v2")
    if isinstance(raw, KbQueryPlan):
        return raw
    if isinstance(raw, dict):
        return KbQueryPlan.model_validate(raw)
    raise ValueError("kb_hop_retriever requires kb_query_plan_v2.")


def _retrieve_hops(
    plan: KbQueryPlan,
    explicit_clause_ids: tuple[str, ...],
) -> tuple[list[KbHopResult], list[RetrievalAttempt], list[AuditEvent]]:
    executed = plan.hops[:MAX_KB_PLAN_HOPS]
    audit_log = _truncation_events(plan, executed)
    calls = [
        partial(
            _retrieve_hop,
            plan,
            hop,
            index=index,
            explicit_clause_ids=explicit_clause_ids,
        )
        for index, hop in enumerate(executed)
    ]
    parallel = kb_hop_parallel_enabled() and all(not hop.depends_on for hop in executed)
    hop_outputs = run_kb_hop_calls(calls, parallel=parallel)
    results = [output[0] for output in hop_outputs]
    attempts = [attempt for output in hop_outputs for attempt in output[1]]
    audit_log.extend(event for output in hop_outputs for event in output[2])
    return results, attempts, audit_log


def _truncation_events(plan: KbQueryPlan, executed: list[KbHop]) -> list[AuditEvent]:
    if len(plan.hops) > len(executed):
        return [
            AuditEvent(
                node_name="kb_hop_retriever",
                message="Truncated KB plan hops to retrieval budget.",
                metadata={
                    "planned_hops": len(plan.hops),
                    "executed_hops": len(executed),
                    "budget": MAX_KB_PLAN_HOPS,
                },
            )
        ]
    return []


def _retrieve_hop(
    plan: KbQueryPlan,
    hop: KbHop,
    *,
    index: int,
    explicit_clause_ids: tuple[str, ...],
) -> tuple[KbHopResult, list[RetrievalAttempt], list[AuditEvent]]:
    chunks, attempts, traces = retrieve_evidence(
        _options(
            hop,
            comparison_query=plan.plan_type == "compare",
            explicit_clause_ids=explicit_clause_ids if index == 0 else (),
        )
    )
    result = KbHopResult(hop_id=hop.hop_id, query=hop.query, chunks=chunks)
    return result, attempts, [*traces, _hop_event(hop, len(chunks))]


def _options(
    hop: KbHop,
    *,
    comparison_query: bool,
    explicit_clause_ids: tuple[str, ...],
) -> EvidenceRetrievalOptions:
    return EvidenceRetrievalOptions(
        node_name="kb_hop_retriever",
        title=f"KB hop {hop.hop_id}",
        query=hop.query,
        filters={},
        stage_goal=hop.stage_goal,
        final_top_k=hop.final_top_k,
        comparison_query=comparison_query,
        explicit_clause_ids=explicit_clause_ids,
    )


def _hop_event(hop: KbHop, chunk_count: int) -> AuditEvent:
    return AuditEvent(
        node_name="kb_hop_retriever",
        message="Retrieved one planned KB hop.",
        metadata={
            "hop_id": hop.hop_id,
            "query": hop.query,
            "object_label": hop.object_label,
            "slot": hop.slot,
            "chunk_count": chunk_count,
        },
    )


def _missing_slots(plan: KbQueryPlan, results: list[KbHopResult]) -> list[str]:
    missing = list(plan.missing_user_slots)
    hop_by_id = {hop.hop_id: hop for hop in plan.hops}
    for result in results:
        if result.chunks:
            continue
        hop = hop_by_id.get(result.hop_id)
        label = hop.slot or hop.object_label or result.hop_id if hop else result.hop_id
        if label not in missing:
            missing.append(label)
    return missing


def _evidence_slots(
    plan: KbQueryPlan,
    results: list[KbHopResult],
    citable_chunks: list[RetrievedChunk],
) -> dict[str, list[RetrievedChunk]]:
    citable_ids = {chunk.chunk_id or chunk.citation_id for chunk in citable_chunks}
    result_by_hop = {result.hop_id: result for result in results}
    slots: dict[str, list[RetrievedChunk]] = {}
    for hop in plan.hops:
        key = _evidence_group_key(plan.plan_type, hop)
        if not key or hop.hop_id not in result_by_hop:
            continue
        chunks = result_by_hop[hop.hop_id].chunks
        slots[key] = [
            chunk for chunk in chunks if (chunk.chunk_id or chunk.citation_id) in citable_ids
        ]
    return slots


def _evidence_group_key(plan_type: str, hop: KbHop) -> str | None:
    del plan_type
    return hop.slot or hop.object_label or hop.hop_id


def _planning_trace(
    state: AgentState,
    results: list[KbHopResult],
    missing: list[str],
) -> dict[str, object]:
    prior = dict(state.get("kb_planning_trace") or {})
    return {
        **prior,
        "executed_hops": [
            {"hop_id": result.hop_id, "query": result.query, "chunk_count": len(result.chunks)}
            for result in results
        ],
        "missing_slots": missing,
    }
