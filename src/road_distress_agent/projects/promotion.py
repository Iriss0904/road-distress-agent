"""Promote a finished diagnosis thread's terminal state into a DefectRecord.

The diagnosis layer stays the source of truth. Promotion is a read-only
projection of the checkpoint terminal state, gated by the same safety invariant
``memory_writer`` uses for case summaries: a structured ``FinalAnswer`` that has
passed ``safety_critic``.
"""

from __future__ import annotations

from typing import Any

from road_distress_agent.projects.models import DefectPayload
from road_distress_agent.state import FinalAnswer, RetrievedChunk

PROMOTE_REQUIRES_PASS = "尚未通过安全审查，无法纳入台账。"
PROMOTE_REQUIRES_ANSWER = "尚无结构化诊断结论，无法纳入台账。"


def promotion_blocker(state: dict[str, Any]) -> str | None:
    """Return a human-readable reason the state cannot be promoted, else None."""
    review = state.get("safety_review")
    if not (review and getattr(review, "passed", False)):
        return PROMOTE_REQUIRES_PASS
    if not isinstance(state.get("final_answer"), FinalAnswer):
        return PROMOTE_REQUIRES_ANSWER
    return None


def extract_defect_payload(state: dict[str, Any]) -> DefectPayload:
    """Project a promotable terminal state into an immutable DefectPayload."""
    blocker = promotion_blocker(state)
    if blocker is not None:
        raise ValueError(blocker)

    answer: FinalAnswer = state["final_answer"]
    return DefectPayload(
        defect_category=state.get("defect_category"),
        defect_subtype=state.get("defect_subtype"),
        material=state.get("material"),
        known_features=dict(state.get("known_features") or {}),
        chosen_method=state.get("chosen_method"),
        summary=answer.summary,
        procedure_refs=_chunk_refs(state.get("procedure_chunks")),
        acceptance_refs=_chunk_refs(state.get("acceptance_chunks")),
        citations=[_citation_dict(c) for c in answer.citations],
        photo_refs=_photo_refs(state.get("latest_attachments")),
        final_answer=answer.model_dump(mode="json"),
    )


def _chunk_refs(chunks: Any) -> list[str]:
    refs: list[str] = []
    for chunk in chunks or []:
        ref = getattr(chunk, "clause_id", None) or getattr(chunk, "chunk_id", None)
        if ref:
            refs.append(str(ref))
    return refs


def _citation_dict(citation: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": citation.chunk_id,
        "clause_id": citation.clause_id,
        "source_standard": citation.source_standard,
        "source_clause": citation.source_clause,
        "snippet": citation.snippet or citation.text,
    }


def _photo_refs(attachments: Any) -> list[str]:
    return [a.uri for a in attachments or [] if getattr(a, "uri", None)]
