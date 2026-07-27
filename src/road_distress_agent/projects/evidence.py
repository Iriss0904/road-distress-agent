"""Pure evidence-completeness projection over a promoted defect payload."""

from __future__ import annotations

from typing import Literal

from road_distress_agent.projects.models import DefectPayload

EvidenceCompleteness = Literal["complete", "gap", "missing"]

EVIDENCE_COMPLETE = "complete"
EVIDENCE_GAP = "gap"
EVIDENCE_MISSING = "missing"
_SIZE_KEYS = ("crack_width_mm", "width_mm", "depth_mm", "area_m2")


def evidence_completeness(payload: DefectPayload) -> EvidenceCompleteness:
    if not payload.citations and not payload.chosen_method:
        return EVIDENCE_MISSING
    if payload.citations and not _has_size(payload.known_features):
        return EVIDENCE_GAP
    return EVIDENCE_COMPLETE


def _has_size(features: dict[str, object]) -> bool:
    return any(features.get(key) is not None for key in _SIZE_KEYS)
