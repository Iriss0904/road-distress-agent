"""Deterministic evidence boundary shared by KB answer composers."""

from __future__ import annotations

from dataclasses import dataclass

from road_distress_agent.evidence_assessment import EvidenceAssessment, EvidenceStatus
from road_distress_agent.reference_index import build_reference_index
from road_distress_agent.state import AgentState, Citation, ReferenceItem, RetrievedChunk


@dataclass(frozen=True)
class UnsupportedSlot:
    slot_id: str
    reason_code: str

    def payload(self) -> dict[str, str]:
        return {"slot_id": self.slot_id, "reason_code": self.reason_code}


@dataclass(frozen=True)
class ComposerEvidence:
    chunks: tuple[RetrievedChunk, ...]
    references: tuple[ReferenceItem, ...]
    unsupported_slots: tuple[UnsupportedSlot, ...] = ()

    def unsupported_payload(self) -> list[dict[str, str]]:
        return [slot.payload() for slot in self.unsupported_slots]


def composer_evidence(
    state: AgentState,
    chunks: list[RetrievedChunk],
    references: list[ReferenceItem],
) -> ComposerEvidence:
    """Restrict partial answers to assessment-approved evidence."""
    assessment = _assessment(state.get("evidence_assessment"))
    if assessment is None or assessment.status is not EvidenceStatus.PARTIAL:
        return ComposerEvidence(tuple(chunks), tuple(references))
    allowed_ids = set(assessment.allowed_chunk_ids)
    allowed_chunks = tuple(chunk for chunk in chunks if _chunk_id(chunk) in allowed_ids)
    _validate_partial_boundary(assessment, allowed_chunks)
    return ComposerEvidence(
        chunks=allowed_chunks,
        references=tuple(_reference_index(allowed_chunks)),
        unsupported_slots=_unsupported_slots(assessment),
    )


def ensure_citations_allowed(
    cited_chunk_ids: list[str],
    references: tuple[ReferenceItem, ...] | list[ReferenceItem],
    node_name: str,
) -> None:
    allowed = {chunk_id for item in references for chunk_id in item.chunk_ids}
    invalid = [chunk_id for chunk_id in cited_chunk_ids if chunk_id not in allowed]
    if invalid:
        raise ValueError(f"{node_name} cited chunk_ids outside allowed evidence: {invalid!r}.")


def _assessment(value: object) -> EvidenceAssessment | None:
    if value is None:
        return None
    if isinstance(value, EvidenceAssessment):
        return value
    return EvidenceAssessment.model_validate(value)


def _validate_partial_boundary(
    assessment: EvidenceAssessment,
    chunks: tuple[RetrievedChunk, ...],
) -> None:
    available = {_chunk_id(chunk) for chunk in chunks}
    missing = set(assessment.allowed_chunk_ids) - available
    if missing:
        raise ValueError(
            f"partial assessment allowed chunk_ids are unavailable: {sorted(missing)!r}."
        )
    if not assessment.allowed_chunk_ids:
        raise ValueError("partial assessment requires at least one allowed chunk_id.")
    if not _unsupported_slots(assessment):
        raise ValueError("partial assessment requires at least one unsupported slot.")


def _unsupported_slots(assessment: EvidenceAssessment) -> tuple[UnsupportedSlot, ...]:
    return tuple(
        UnsupportedSlot(slot_id=item.slot_id, reason_code=item.reason_code or "")
        for item in assessment.slot_coverage
        if not item.allowed_chunk_ids
    )


def _reference_index(chunks: tuple[RetrievedChunk, ...]) -> list[ReferenceItem]:
    return build_reference_index(
        [Citation.model_validate(chunk.model_dump(mode="json")) for chunk in chunks]
    )


def _chunk_id(chunk: RetrievedChunk) -> str | None:
    return chunk.chunk_id or chunk.citation_id
