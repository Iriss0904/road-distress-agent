"""Adapters that project existing producer outputs into A0 observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from road_distress_agent.evidence_assessment import (
    EvidenceAssessmentInput,
    EvidenceObservation,
    EvidenceRiskFlag,
    EvidenceSlotCoverage,
    assess_evidence,
)
from road_distress_agent.evidence_clause_boundary import observe_single_clause_absence
from road_distress_agent.state import (
    AgentState,
    AuditEvent,
    InterruptState,
    KbQueryPlan,
    RetrievalAttempt,
    RetrievedChunk,
)
from road_distress_agent.tracing import trace_event

EVIDENCE_TRACE_KIND = "evidence_assessment"


def observe_off_topic() -> EvidenceObservation:
    return assess_evidence(EvidenceAssessmentInput(route_action="off_topic"))


def observe_simple_kb(
    chunks: Sequence[RetrievedChunk],
    plan: KbQueryPlan | None = None,
    *,
    original_user_text: str | None = None,
    retrieval_attempts: Sequence[RetrievalAttempt] = (),
) -> EvidenceObservation:
    unavailable = _producer_input_unavailable(chunks)
    if unavailable is not None:
        return unavailable
    missing_reason = _missing_fields_unavailable(plan)
    if missing_reason is not None:
        return EvidenceObservation(field_unavailable_reason=missing_reason)
    clause_absence = observe_single_clause_absence(original_user_text, retrieval_attempts, chunks)
    if clause_absence is not None:
        return clause_absence
    return assess_evidence(
        EvidenceAssessmentInput(
            retrieval_completed=True,
            selected_chunk_ids=_chunk_ids(chunks),
            missing_user_fields=_missing_user_fields(plan) if plan else (),
            risk_flags=_risk_flags(chunks),
        )
    )


def observe_kb_plan(plan: KbQueryPlan) -> EvidenceObservation:
    missing_reason = _missing_fields_unavailable(plan)
    if missing_reason is not None:
        return EvidenceObservation(field_unavailable_reason=missing_reason)
    return assess_evidence(EvidenceAssessmentInput(missing_user_fields=_missing_user_fields(plan)))


def observe_clarification_interrupt(
    interrupt: InterruptState,
) -> tuple[AgentState, list[AuditEvent]]:
    if not interrupt.required_fields:
        return {}, []
    fields = tuple(interrupt.required_fields)
    if _has_blank(fields):
        observation = EvidenceObservation(field_unavailable_reason="missing_user_fields_invalid")
    else:
        observation = assess_evidence(EvidenceAssessmentInput(missing_user_fields=_unique(fields)))
    return (
        {"evidence_assessment": observation.assessment},
        [
            evidence_observation_event(
                node_name=interrupt.node_name,
                observation=observation,
            )
        ],
    )


def observe_planned_kb(
    plan: KbQueryPlan,
    chunks: Sequence[RetrievedChunk],
    slot_chunks: Mapping[str, Sequence[RetrievedChunk]],
    *,
    original_user_text: str | None = None,
    retrieval_attempts: Sequence[RetrievalAttempt] = (),
) -> EvidenceObservation:
    slot_evidence = [chunk for values in slot_chunks.values() for chunk in values]
    unavailable = _producer_input_unavailable([*chunks, *slot_evidence])
    if unavailable is not None:
        return unavailable
    missing_reason = _missing_fields_unavailable(plan)
    if missing_reason is not None:
        return EvidenceObservation(field_unavailable_reason=missing_reason)
    clause_absence = observe_single_clause_absence(original_user_text, retrieval_attempts, chunks)
    if clause_absence is not None:
        return clause_absence
    slot_input_reason = _plan_slot_input_unavailable(plan, slot_chunks)
    if slot_input_reason is not None:
        return EvidenceObservation(field_unavailable_reason=slot_input_reason)
    selected_ids = _chunk_ids(chunks)
    required_slots = _required_slots(plan)
    mapping_reason = _slot_mapping_unavailable(slot_chunks, required_slots, selected_ids)
    if mapping_reason is not None:
        return EvidenceObservation(field_unavailable_reason=mapping_reason)
    return assess_evidence(
        EvidenceAssessmentInput(
            retrieval_completed=True,
            selected_chunk_ids=selected_ids,
            required_slots=required_slots,
            slot_coverage=_slot_coverage(slot_chunks),
            missing_user_fields=_missing_user_fields(plan),
            risk_flags=_risk_flags(chunks),
        )
    )


def observe_detail_evidence(
    procedure_chunks: Sequence[RetrievedChunk],
    acceptance_chunks: Sequence[RetrievedChunk],
) -> EvidenceObservation:
    all_chunks = [*procedure_chunks, *acceptance_chunks]
    unavailable = _producer_input_unavailable(all_chunks)
    if unavailable is not None:
        return unavailable
    selected_ids = _chunk_ids(all_chunks)
    slot_chunks = {
        "construction_steps": procedure_chunks,
        "acceptance_criteria": acceptance_chunks,
    }
    return assess_evidence(
        EvidenceAssessmentInput(
            retrieval_completed=True,
            selected_chunk_ids=selected_ids,
            required_slots=("construction_steps", "acceptance_criteria"),
            slot_coverage=_slot_coverage(slot_chunks),
            risk_flags=_risk_flags(all_chunks),
        )
    )


def evidence_observation_event(
    *,
    node_name: str,
    observation: EvidenceObservation,
    affects_runtime_behavior: bool = False,
) -> AuditEvent:
    return trace_event(
        node_name=node_name,
        kind=EVIDENCE_TRACE_KIND,
        title="Evidence assessment",
        output={
            "evidence_assessment": observation.assessment,
            "field_unavailable_reason": observation.field_unavailable_reason,
        },
        metadata={"affects_runtime_behavior": affects_runtime_behavior},
    )


def _required_slots(plan: KbQueryPlan) -> tuple[str, ...]:
    return _unique(_raw_required_slots(plan))


def _missing_user_fields(plan: KbQueryPlan) -> tuple[str, ...]:
    clarification = plan.clarification
    explicit = tuple(clarification.missing_user_slots) if clarification else ()
    return _unique((*plan.missing_user_slots, *explicit))


def _missing_fields_unavailable(plan: KbQueryPlan | None) -> str | None:
    if plan is None:
        return None
    clarification = plan.clarification
    explicit = tuple(clarification.missing_user_slots) if clarification else ()
    values = (*plan.missing_user_slots, *explicit)
    return "missing_user_fields_invalid" if _has_blank(values) else None


def _plan_slot_input_unavailable(
    plan: KbQueryPlan,
    slot_chunks: Mapping[str, Sequence[RetrievedChunk]],
) -> str | None:
    if _has_blank(_raw_required_slots(plan)):
        return "required_slots_invalid"
    if _has_blank(tuple(slot_chunks)):
        return "slot_mapping_keys_invalid"
    return None


def _raw_required_slots(plan: KbQueryPlan) -> tuple[str, ...]:
    if plan.evidence_slots:
        return tuple(plan.evidence_slots)
    values: list[str] = []
    for hop in plan.hops:
        values.append(hop.slot or hop.object_label or hop.hop_id)
    return tuple(values)


def _slot_coverage(
    slot_chunks: Mapping[str, Sequence[RetrievedChunk]],
) -> tuple[EvidenceSlotCoverage, ...]:
    rows: list[EvidenceSlotCoverage] = []
    for slot_id, chunks in slot_chunks.items():
        allowed = _chunk_ids(chunks)
        reason = None if allowed else "required_slot_uncovered"
        rows.append(
            EvidenceSlotCoverage(
                slot_id=slot_id,
                allowed_chunk_ids=allowed,
                reason_code=reason,
            )
        )
    return tuple(rows)


def _chunk_ids(chunks: Sequence[RetrievedChunk]) -> tuple[str, ...]:
    values = tuple(chunk.chunk_id or chunk.citation_id or "" for chunk in chunks)
    return _unique(tuple(value for value in values if value))


def _risk_flags(chunks: Sequence[RetrievedChunk]) -> tuple[EvidenceRiskFlag, ...]:
    values = _risk_flag_values(chunks)
    known = {flag.value: flag for flag in EvidenceRiskFlag}
    return tuple(known[value] for value in _unique(values))


def _risk_flag_values(chunks: Sequence[RetrievedChunk]) -> tuple[str, ...]:
    values: list[str] = []
    for chunk in chunks:
        raw = chunk.annotations.get("risk_flags") if chunk.annotations else None
        if isinstance(raw, (list, tuple)):
            values.extend(str(value) for value in raw)
    return tuple(values)


def _producer_input_unavailable(
    chunks: Sequence[RetrievedChunk],
) -> EvidenceObservation | None:
    if any(_chunk_id_unavailable(chunk) for chunk in chunks):
        return EvidenceObservation(field_unavailable_reason="selected_evidence_ids_unavailable")
    invalid_shape = any(
        chunk.annotations.get("risk_flags") is not None
        and not isinstance(chunk.annotations.get("risk_flags"), (list, tuple))
        for chunk in chunks
        if chunk.annotations
    )
    if invalid_shape:
        return EvidenceObservation(field_unavailable_reason="explicit_risk_flags_invalid_shape")
    known = {flag.value for flag in EvidenceRiskFlag}
    if any(value not in known for value in _risk_flag_values(chunks)):
        return EvidenceObservation(field_unavailable_reason="explicit_risk_flags_unrecognized")
    return None


def _chunk_id_unavailable(chunk: RetrievedChunk) -> bool:
    value = chunk.chunk_id or chunk.citation_id
    return value is None or not value.strip()


def _slot_mapping_unavailable(
    slot_chunks: Mapping[str, Sequence[RetrievedChunk]],
    required_slots: tuple[str, ...],
    selected_ids: tuple[str, ...],
) -> str | None:
    if not set(slot_chunks).issubset(required_slots):
        return "slot_mapping_not_in_required_slots"
    linked_ids = {chunk_id for chunks in slot_chunks.values() for chunk_id in _chunk_ids(chunks)}
    if not linked_ids.issubset(selected_ids):
        return "slot_evidence_not_selected"
    return None


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _has_blank(values: Sequence[str]) -> bool:
    return any(not value.strip() for value in values)
