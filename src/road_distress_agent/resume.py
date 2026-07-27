# ─────────────────────────────────────────────────────────────
# AUDIT NOTES (post-refactor)
# 保留：interrupt kind 常量 + 通用 state-path 取值 + missing_state_paths
#       + active_interrupt_required_fields；这些都是 HITL resume 的通用骨架。
# 删除：CRACK_REQUIRED_FIELD_PATHS = ("distress.width_mm", "distress.length_m")
#       —— 这是按裂缝病害硬编码的必填字段列表，正是新设计反对的枚举式约束。
#       新设计下"必填字段"由 Discriminator 在每一轮动态决定，
#       通过 InterruptState.required_fields 显式声明，
#       不再有默认表。
# 当 InterruptState.required_fields 为空时，
# missing_active_resume_requirements 默认返回空列表（即"由节点自行处理"）。
# ─────────────────────────────────────────────────────────────
"""HITL resume requirement helpers."""

from __future__ import annotations

from typing import Any

from road_distress_agent.state import AgentState

MISSING_REQUIRED_FIELDS = "missing_required_fields"
CANDIDATE_SELECTION = "candidate_selection"
EVIDENCE_REQUIRES_USER_FACT = "evidence_requires_user_fact"
WEATHER_LOCATION_REQUEST = "weather_location_request"
DISEASE_CLARIFICATION = "disease_clarification"
DISEASE_SELECTION = "disease_selection"
METHOD_CLARIFICATION = "method_clarification"
METHOD_SELECTION = "method_selection"

INTERRUPT_RESUME_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    # MISSING_REQUIRED_FIELDS: required fields are now declared per-interrupt
    # by the Discriminator (new design §四 Step C) via InterruptState.required_fields.
    MISSING_REQUIRED_FIELDS: (),
    CANDIDATE_SELECTION: (
        "candidate_selection.selected_candidate_id",
        "candidate_selection.selected_name",
    ),
    EVIDENCE_REQUIRES_USER_FACT: (),
    WEATHER_LOCATION_REQUEST: (),
}


def state_path_value(state: AgentState, path: str) -> Any:
    """Read a dotted State/model path such as ``distress.width_mm``."""
    current: Any = state
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current


def is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def missing_state_paths(state: AgentState, field_paths: tuple[str, ...]) -> list[str]:
    return [path for path in field_paths if not is_present(state_path_value(state, path))]


def active_interrupt_required_fields(state: AgentState) -> tuple[str, ...]:
    interrupt = state.get("interrupt")
    if interrupt is None or interrupt.kind is None:
        return ()
    if interrupt.required_fields:
        return tuple(interrupt.required_fields)
    return INTERRUPT_RESUME_REQUIREMENTS.get(interrupt.kind, ())


def missing_active_resume_requirements(state: AgentState) -> list[str]:
    return missing_state_paths(state, active_interrupt_required_fields(state))
