"""Runtime adapters between pure evidence gates and existing composers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from road_distress_agent.evidence_assessment import EvidenceAssessment
from road_distress_agent.evidence_gate import (
    EvidenceGateContractError,
    EvidenceGateErrorCode,
    GateDecision,
    GatePolicy,
    KbSlotInput,
)
from road_distress_agent.state import (
    AgentState,
    AuditEvent,
    KbHopResult,
    KbQueryPlan,
    RetrievedChunk,
)
from road_distress_agent.tracing import trace_event


class EvidenceGateDecisionError(RuntimeError):
    """Loud dependency error returned by the pure gate."""

    def __init__(
        self,
        decision: GateDecision,
        value: EvidenceAssessment | None,
    ) -> None:
        self.decision = decision.decision
        self.reason = decision.reason
        failure = value.dependency_failure if value is not None else None
        self.dependency_failure = failure
        self.category = failure.category if failure is not None else None
        self.code = failure.code if failure is not None else None
        self.step = failure.step if failure is not None else None
        self.retriable = failure.retriable if failure is not None else None
        super().__init__(
            f"evidence gate decision={decision.decision} reason={decision.reason.value}"
        )


def assessment(state: AgentState) -> EvidenceAssessment | None:
    raw = state.get("evidence_assessment")
    if raw is None or isinstance(raw, EvidenceAssessment):
        return raw
    return EvidenceAssessment.model_validate(raw)


def chunks(raw: Sequence[object]) -> tuple[RetrievedChunk, ...]:
    return tuple(
        item if isinstance(item, RetrievedChunk) else RetrievedChunk.model_validate(item)
        for item in raw
    )


def gate_trace(
    node_name: str,
    decision: GateDecision,
    policy: GatePolicy,
    *,
    template_version: str | None = None,
) -> AuditEvent:
    metadata: dict[str, object] = {"affects_runtime_behavior": True}
    if template_version is not None:
        metadata["template_version"] = template_version
    return trace_event(
        node_name=node_name,
        kind="evidence_gate",
        title="Evidence gate decision",
        output={
            "decision": decision.decision,
            "reason": decision.reason.value,
            "usable_chunk_ids": [_chunk_id(item) for item in decision.usable_chunks],
            "passed_slots": list(decision.passed_slots),
            "policy_tier": policy.tier.value,
        },
        metadata=metadata,
    )


def raise_on_error(
    decision: GateDecision,
    value: EvidenceAssessment | None = None,
) -> None:
    if decision.decision == "ERROR":
        raise EvidenceGateDecisionError(decision, value)


def planned_slot_inputs(state: AgentState, plan: KbQueryPlan) -> tuple[KbSlotInput, ...]:
    mapping = _slot_mapping(state)
    descriptors = _slot_descriptors(plan)
    if not {slot_id for slot_id, _, _ in descriptors}.issubset(mapping):
        raise EvidenceGateContractError(EvidenceGateErrorCode.INVALID_SLOT, None)
    results = _hop_results(state)
    _validate_slot_evidence(descriptors, mapping, results)
    return tuple(
        KbSlotInput(slot_id=slot_id, query=query, chunks=mapping[slot_id])
        for slot_id, query, _ in descriptors
    )


def unsupported_slots(
    slots: tuple[KbSlotInput, ...],
    passed_slots: tuple[str, ...],
) -> tuple[str, ...]:
    passed = frozenset(passed_slots)
    return tuple(slot.slot_id for slot in slots if slot.slot_id not in passed)


def _slot_mapping(state: AgentState) -> dict[str, tuple[RetrievedChunk, ...]]:
    raw = state.get("kb_evidence_slots") or {}
    if not isinstance(raw, Mapping):
        raise TypeError("kb_evidence_slots must be a mapping")
    return {str(key): chunks(value) for key, value in raw.items()}


def _slot_descriptors(plan: KbQueryPlan) -> tuple[tuple[str, str, str], ...]:
    hop_rows = tuple((_hop_key(hop), hop.query, hop.hop_id) for hop in plan.hops)
    if len({hop.hop_id for hop in plan.hops}) != len(plan.hops):
        raise EvidenceGateContractError(EvidenceGateErrorCode.INVALID_SLOT, None)
    if plan.evidence_slots:
        by_key = {key: (query, hop_id) for key, query, hop_id in hop_rows}
        if len(by_key) != len(hop_rows) or any(slot not in by_key for slot in plan.evidence_slots):
            raise ValueError("planned evidence_slots require one stable matching hop")
        return tuple((slot, *by_key[slot]) for slot in plan.evidence_slots)
    if len({key for key, _, _ in hop_rows}) != len(hop_rows):
        raise ValueError("planned hops require unique stable slot keys")
    return hop_rows


def _hop_results(state: AgentState) -> dict[str, KbHopResult]:
    raw = state.get("kb_hop_results") or []
    values = [
        item if isinstance(item, KbHopResult) else KbHopResult.model_validate(item) for item in raw
    ]
    if len({item.hop_id for item in values}) != len(values):
        raise EvidenceGateContractError(EvidenceGateErrorCode.INVALID_SLOT, None)
    return {item.hop_id: item for item in values}


def _validate_slot_evidence(
    descriptors: tuple[tuple[str, str, str], ...],
    mapping: dict[str, tuple[RetrievedChunk, ...]],
    results: dict[str, KbHopResult],
) -> None:
    for slot_id, query, hop_id in descriptors:
        result = results.get(hop_id)
        if result is None or result.query != query:
            raise EvidenceGateContractError(EvidenceGateErrorCode.INVALID_SLOT, None)
        mapped_ids = {_required_chunk_id(item) for item in mapping[slot_id]}
        result_ids = {_required_chunk_id(item) for item in result.chunks}
        if not mapped_ids.issubset(result_ids):
            raise EvidenceGateContractError(EvidenceGateErrorCode.INVALID_SLOT, None)


def _hop_key(hop: object) -> str:
    slot = getattr(hop, "slot", None)
    label = getattr(hop, "object_label", None)
    hop_id = hop.hop_id
    return str(slot or label or hop_id)


def _chunk_id(chunk: RetrievedChunk) -> str | None:
    return chunk.chunk_id or chunk.citation_id


def _required_chunk_id(chunk: RetrievedChunk) -> str:
    chunk_id = _chunk_id(chunk)
    if not chunk_id:
        raise EvidenceGateContractError(EvidenceGateErrorCode.INVALID_SLOT, None)
    return chunk_id
