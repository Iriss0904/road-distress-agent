"""Invalidation rules for diagnosis facts and derived state."""

from __future__ import annotations

from typing import Any, Literal

from road_distress_agent.state import CandidateSelection, FieldChange

NextStage = Literal["disease", "method", "detail", "done"]
TARGET_STAGE: dict[str, NextStage] = {
    "disease": "disease",
    "method": "method",
    "detail": "detail",
    "answer": "done",
}
STAGE_ORDER: dict[NextStage, int] = {"disease": 0, "method": 1, "detail": 2, "done": 3}

DISEASE_KEYS = {
    "crack_length_m",
    "crack_orientation",
    "crack_pattern",
    "rut_depth_mm",
    "depth_mm",
    "depth_cm",
    "area_m2",
}
METHOD_KEYS = {
    "crack_width_mm",
    "base_exposed",
    "base_condition",
    "crack_wall_condition",
    "water_condition",
    "traffic_load",
    "severity",
}
ADVISORY_KEYS = {"advisory_context"}
FEATURE_ALIASES = {
    "width": "crack_width_mm",
    "width_mm": "crack_width_mm",
    "crack_width": "crack_width_mm",
    "crack_width_mm": "crack_width_mm",
    "缝宽": "crack_width_mm",
    "depth": "depth_mm",
    "pothole_depth": "depth_mm",
    "pothole_depth_mm": "depth_mm",
    "rut_depth": "rut_depth_mm",
    "rut_depth_mm": "rut_depth_mm",
    "area": "area_m2",
    "area_m2": "area_m2",
    "severity": "severity",
    "severity_level": "severity",
    "severity_grade": "severity",
    "严重程度": "severity",
    "location": "advisory_context",
    "site_location": "advisory_context",
    "surrounding": "advisory_context",
    "loose_or_stable": "crack_wall_condition",
    "wall_condition": "crack_wall_condition",
    "crack_wall_condition": "crack_wall_condition",
    "crack_wall_stable": "crack_wall_condition",
}


def canonical_key(key: str) -> str:
    normalized = key.strip().lower()
    return FEATURE_ALIASES.get(normalized, normalized)


def highest_invalidated_stage(changes: list[FieldChange], has_defect: bool) -> NextStage:
    stages = [_field_stage(change.field_path) for change in changes]
    if "disease" in stages:
        return "disease"
    if "method" in stages:
        return "method" if has_defect else "disease"
    return "done"


def invalidations(stage: NextStage) -> dict[str, Any]:
    if stage == "disease":
        return {**_disease_invalidations(), **_method_invalidations()}
    if stage == "method":
        return _method_invalidations()
    if stage == "detail":
        return _detail_invalidations()
    return {}


def target_stage(route: dict[str, Any]) -> NextStage | None:
    if route.get("action") != "diagnosis_proceed":
        return None
    target = route.get("target")
    return TARGET_STAGE.get(str(target)) if target is not None else None


def earliest_stage(left: NextStage, right: NextStage) -> NextStage:
    return left if STAGE_ORDER[left] <= STAGE_ORDER[right] else right


def _field_stage(field_path: str) -> NextStage:
    if field_path == "material":
        return "disease"
    key = field_path.rsplit(".", 1)[-1]
    if key in ADVISORY_KEYS:
        return "done"
    if key in METHOD_KEYS:
        return "method"
    if key in DISEASE_KEYS:
        return "disease"
    return "disease"


def _disease_invalidations() -> dict[str, Any]:
    return {
        "defect_category": None,
        "disease_discriminator_output": None,
        "disease_evidence_chunks": [],
    }


def _method_invalidations() -> dict[str, Any]:
    return {
        "chosen_method": None,
        "method_discriminator_output": None,
        "method_evidence_chunks": [],
        "procedure_chunks": [],
        "acceptance_chunks": [],
        "final_answer": None,
        "final_answer_message": None,
        "reference_index": [],
        "query_plan": None,
        "rewritten_query": None,
        "detail_evidence_bundles": {},
        "method_evidence_bundles": {},
        "discriminator_output": None,
        "solution_candidates": [],
        "candidate_selection": CandidateSelection(),
        "interrupt": None,
        "awaiting_user_input": False,
    }


def _detail_invalidations() -> dict[str, Any]:
    return {
        "procedure_chunks": [],
        "acceptance_chunks": [],
        "final_answer": None,
        "final_answer_message": None,
        "reference_index": [],
        "detail_evidence_bundles": {},
    }
