"""Pure, path-aware evidence decisions before answer composition."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, cast

from road_distress_agent.evidence_anchors import query_anchor_coverage
from road_distress_agent.evidence_assessment import EvidenceAssessment, EvidenceStatus
from road_distress_agent.state import RetrievedChunk

MIN_COVERAGE = 0.0
MAX_COVERAGE = 1.0


class GateTier(str, Enum):
    R1 = "R1"
    R2 = "R2"


class EvidenceGateErrorCode(str, Enum):
    SUFFICIENT_WITHOUT_CHUNKS = "sufficient_without_chunks"
    OUT_OF_SCOPE_AT_COMPOSER = "out_of_scope_at_composer"
    MISSING_CHUNK_SCORE = "missing_chunk_score"
    NONFINITE_CHUNK_SCORE = "nonfinite_chunk_score"
    QUERY_WITHOUT_ANCHORS = "query_without_anchors"
    INVALID_SLOT = "invalid_slot"


class EvidenceGateContractError(RuntimeError):
    """A structured, loud failure for an invalid gate input contract."""

    def __init__(
        self,
        code: EvidenceGateErrorCode,
        status: EvidenceStatus | None,
    ) -> None:
        self.code = code
        self.status = status
        super().__init__(f"evidence gate contract violation: {code.value}")


@dataclass(frozen=True)
class GatePolicy:
    tier: GateTier
    semantic_score_threshold: float | None = None
    anchor_coverage_threshold: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tier, GateTier):
            raise TypeError("tier must be a GateTier")
        thresholds = (self.semantic_score_threshold, self.anchor_coverage_threshold)
        if self.tier is GateTier.R1 and any(value is not None for value in thresholds):
            raise ValueError("R1 policy does not accept signal thresholds")
        if self.tier is GateTier.R2 and any(value is None for value in thresholds):
            raise ValueError("R2 policy requires semantic and coverage thresholds")
        if any(value is not None and not math.isfinite(value) for value in thresholds):
            raise ValueError("gate thresholds must be finite")
        if self.anchor_coverage_threshold is not None and not (
            MIN_COVERAGE <= self.anchor_coverage_threshold <= MAX_COVERAGE
        ):
            raise ValueError("coverage threshold must be between zero and one")


RUNTIME_R1_GATE_POLICY = GatePolicy(tier=GateTier.R1)


@dataclass(frozen=True)
class GateDecision:
    decision: Literal["ANSWER", "REFUSE", "ERROR"]
    reason: EvidenceStatus
    usable_chunks: tuple[RetrievedChunk, ...]
    passed_slots: tuple[str, ...] = ()


@dataclass(frozen=True)
class SimpleKbGateInput:
    assessment: EvidenceAssessment | None
    query: str
    chunks: tuple[RetrievedChunk, ...]
    explicit_clause_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class KbSlotInput:
    slot_id: str
    query: str
    chunks: tuple[RetrievedChunk, ...]

    def __post_init__(self) -> None:
        if not self.slot_id.strip():
            raise EvidenceGateContractError(EvidenceGateErrorCode.INVALID_SLOT, None)


@dataclass(frozen=True)
class PlannedKbGateInput:
    assessment: EvidenceAssessment | None
    slots: tuple[KbSlotInput, ...]

    def __post_init__(self) -> None:
        slot_ids = tuple(slot.slot_id for slot in self.slots)
        if len(set(slot_ids)) != len(slot_ids):
            raise EvidenceGateContractError(EvidenceGateErrorCode.INVALID_SLOT, None)


@dataclass(frozen=True)
class DiagnosisGateInput:
    assessment: EvidenceAssessment | None
    procedure_chunks: tuple[RetrievedChunk, ...]
    acceptance_chunks: tuple[RetrievedChunk, ...]


def evidence_gate(options: SimpleKbGateInput, policy: GatePolicy) -> GateDecision:
    """Evaluate a simple KB retrieval over the union of selected chunks."""
    mapped = _mapped_assessment(options.assessment, options.chunks)
    if mapped is not None:
        return mapped
    chunks = options.chunks
    if not chunks:
        return _empty_after_assessment(options.assessment)
    if len(options.explicit_clause_ids) == 1 and not any(
        chunk.clause_id == options.explicit_clause_ids[0] for chunk in chunks
    ):
        return GateDecision("REFUSE", EvidenceStatus.NO_KB_EVIDENCE, ())
    if not _kb_signals_pass(options.query, chunks, policy):
        return GateDecision("REFUSE", EvidenceStatus.NO_KB_EVIDENCE, ())
    return GateDecision("ANSWER", EvidenceStatus.SUFFICIENT, chunks)


def planned_kb_evidence_gate(options: PlannedKbGateInput, policy: GatePolicy) -> GateDecision:
    """Evaluate each planned slot independently, then expose only passing evidence."""
    all_chunks = tuple(chunk for slot in options.slots for chunk in slot.chunks)
    mapped = _mapped_assessment(options.assessment, all_chunks)
    if mapped is not None:
        return mapped
    passing_slots = tuple(
        (slot.slot_id, _passing_slot_chunks(slot, policy)) for slot in options.slots
    )
    passed_ids = tuple(slot_id for slot_id, chunks in passing_slots if chunks)
    passing = tuple(chunk for _, chunks in passing_slots for chunk in chunks)
    usable = _dedupe_chunks(passing)
    if not usable:
        return GateDecision("REFUSE", EvidenceStatus.NO_KB_EVIDENCE, ())
    reason = _planned_answer_reason(len(passed_ids), len(options.slots))
    return GateDecision("ANSWER", reason, usable, passed_ids)


def diagnosis_evidence_gate(options: DiagnosisGateInput, policy: GatePolicy) -> GateDecision:
    """Require procedure evidence; acceptance evidence remains optional."""
    del policy
    mapped = _mapped_assessment(options.assessment, options.procedure_chunks)
    if mapped is not None:
        return mapped
    procedure = options.procedure_chunks
    if not procedure:
        return GateDecision("REFUSE", _empty_reason(options.assessment), ())
    acceptance = options.acceptance_chunks
    usable = _dedupe_chunks((*procedure, *acceptance))
    return GateDecision("ANSWER", EvidenceStatus.SUFFICIENT, usable)


def _mapped_assessment(
    assessment: EvidenceAssessment | None,
    chunks: Sequence[RetrievedChunk],
) -> GateDecision | None:
    if assessment is None:
        return None
    status = assessment.status
    if status is EvidenceStatus.OUT_OF_SCOPE:
        raise EvidenceGateContractError(EvidenceGateErrorCode.OUT_OF_SCOPE_AT_COMPOSER, status)
    if status is EvidenceStatus.SUFFICIENT and not chunks:
        raise EvidenceGateContractError(EvidenceGateErrorCode.SUFFICIENT_WITHOUT_CHUNKS, status)
    if status is EvidenceStatus.DEPENDENCY_FAILURE:
        return GateDecision("ERROR", status, ())
    if status in _hard_refusal_statuses():
        return GateDecision("REFUSE", status, ())
    return None


def _hard_refusal_statuses() -> frozenset[EvidenceStatus]:
    return frozenset(
        {
            EvidenceStatus.NO_KB_EVIDENCE,
            EvidenceStatus.CONFLICTING,
            EvidenceStatus.MISSING_USER_CONTEXT,
        }
    )


def _kb_signals_pass(
    query: str,
    chunks: tuple[RetrievedChunk, ...],
    policy: GatePolicy,
) -> bool:
    if policy.tier is GateTier.R1:
        return True
    semantic_threshold = cast(float, policy.semantic_score_threshold)
    coverage_threshold = cast(float, policy.anchor_coverage_threshold)
    if _top_score(chunks) < semantic_threshold:
        return False
    try:
        coverage = query_anchor_coverage(query, chunks)
    except ValueError as exc:
        raise EvidenceGateContractError(EvidenceGateErrorCode.QUERY_WITHOUT_ANCHORS, None) from exc
    return coverage >= coverage_threshold


def _top_score(chunks: tuple[RetrievedChunk, ...]) -> float:
    scores = tuple(chunk.score for chunk in chunks)
    if any(score is None for score in scores):
        raise EvidenceGateContractError(EvidenceGateErrorCode.MISSING_CHUNK_SCORE, None)
    if any(not math.isfinite(cast(float, score)) for score in scores):
        raise EvidenceGateContractError(EvidenceGateErrorCode.NONFINITE_CHUNK_SCORE, None)
    return max(cast(float, score) for score in scores)


def _passing_slot_chunks(
    slot: KbSlotInput,
    policy: GatePolicy,
) -> tuple[RetrievedChunk, ...]:
    chunks = slot.chunks
    if not chunks or not _kb_signals_pass(slot.query, chunks, policy):
        return ()
    return chunks


def _dedupe_chunks(chunks: Sequence[RetrievedChunk]) -> tuple[RetrievedChunk, ...]:
    unique: list[RetrievedChunk] = []
    seen_ids: set[str] = set()
    for chunk in chunks:
        chunk_id = _chunk_id(chunk)
        if chunk_id is not None and chunk_id in seen_ids:
            continue
        if chunk_id is None and chunk in unique:
            continue
        unique.append(chunk)
        if chunk_id is not None:
            seen_ids.add(chunk_id)
    return tuple(unique)


def _planned_answer_reason(
    passed_count: int,
    slot_count: int,
) -> EvidenceStatus:
    if passed_count < slot_count:
        return EvidenceStatus.PARTIAL
    return EvidenceStatus.SUFFICIENT


def _empty_after_assessment(assessment: EvidenceAssessment | None) -> GateDecision:
    return GateDecision("REFUSE", _empty_reason(assessment), ())


def _empty_reason(assessment: EvidenceAssessment | None) -> EvidenceStatus:
    del assessment
    return EvidenceStatus.NO_KB_EVIDENCE


def _chunk_id(chunk: RetrievedChunk) -> str | None:
    return cast(str | None, chunk.chunk_id or chunk.citation_id)
