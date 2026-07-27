"""Deterministic evidence observation for explicit clause lookups."""

from __future__ import annotations

from collections.abc import Sequence

from road_distress_agent.evidence_assessment import (
    EvidenceAssessmentInput,
    EvidenceObservation,
    EvidenceRiskFlag,
    EvidenceSlotCoverage,
    assess_evidence,
)
from road_distress_agent.retrieval.query_features import detect_query_features
from road_distress_agent.state import RetrievalAttempt, RetrievedChunk

TARGET_CLAUSE_ABSENT_REASON = "target_clause_absent"
TARGET_CLAUSE_SLOT_PREFIX = "requested_clause:"


def observe_single_clause_absence(
    original_user_text: str | None,
    retrieval_attempts: Sequence[RetrievalAttempt],
    selected_chunks: Sequence[RetrievedChunk],
) -> EvidenceObservation | None:
    """Observe absence only for one explicit user-stated clause."""
    # Callers validate stable IDs and risk metadata before this boundary check.
    clause_ids = detect_query_features(original_user_text or "", {}).clause_ids
    if len(clause_ids) != 1:
        return None
    if any(chunk.clause_id == clause_ids[0] for chunk in selected_chunks):
        return None
    del retrieval_attempts
    coverage = EvidenceSlotCoverage(
        slot_id=f"{TARGET_CLAUSE_SLOT_PREFIX}{clause_ids[0]}",
        reason_code=TARGET_CLAUSE_ABSENT_REASON,
    )
    return assess_evidence(
        EvidenceAssessmentInput(
            retrieval_completed=True,
            selected_chunk_ids=_chunk_ids(selected_chunks),
            required_slots=(coverage.slot_id,),
            slot_coverage=(coverage,),
            risk_flags=_risk_flags(selected_chunks),
        )
    )


def _chunk_ids(chunks: Sequence[RetrievedChunk]) -> tuple[str, ...]:
    values = (chunk.chunk_id or chunk.citation_id or "" for chunk in chunks)
    return tuple(dict.fromkeys(value for value in values if value))


def _risk_flags(chunks: Sequence[RetrievedChunk]) -> tuple[EvidenceRiskFlag, ...]:
    values = (
        str(value)
        for chunk in chunks
        for value in ((chunk.annotations or {}).get("risk_flags") or ())
    )
    return tuple(EvidenceRiskFlag(value) for value in dict.fromkeys(values))
