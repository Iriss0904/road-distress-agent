"""State-aware payload construction for the top-level router."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from road_distress_agent.nodes.diagnosis_dependencies import canonical_key
from road_distress_agent.nodes.payload_helpers import plain_model
from road_distress_agent.resume import (
    DISEASE_CLARIFICATION,
    DISEASE_SELECTION,
    METHOD_CLARIFICATION,
    METHOD_SELECTION,
)
from road_distress_agent.state import AgentState

RouteAction = Literal["diagnosis_proceed", "kb_aside", "off_topic"]
RouteTarget = Literal["disease", "method", "detail", "answer"]
DiagnosisIntent = Literal[
    "advance",
    "supplement",
    "correct",
    "answer_interrupt",
    "candidate_choice",
]
RagTier = Literal[
    "no_retrieval",
    "single_hop",
    "single_hop_needs_rewrite",
    "multi_hop",
]
TopRouterAnswerMode = Literal[
    "direct",
    "chain_reasoning",
    "compare_table",
    "procedure",
    "clarification",
]

RECENT_DIALOGUE_LIMIT = 6
MESSAGE_SNIPPET_LIMIT = 500
LAST_CANDIDATES_LIMIT = 3
FIELD_LABELS = {
    "crack_width_mm": "缝宽",
    "width_mm": "缝宽",
    "width": "缝宽",
    "crack_length_m": "裂缝长度",
    "crack_orientation": "裂缝方向",
    "crack_pattern": "裂缝形态",
    "rut_depth_mm": "车辙深度",
    "depth_mm": "深度",
    "area_m2": "影响面积",
    "base_exposed": "是否露基层",
    "base_condition": "基层状况",
    "crack_wall_condition": "缝壁状况",
    "water_condition": "水相关状况",
    "traffic_load": "交通荷载",
    "severity": "严重程度",
}


class TopRouteResult(BaseModel):
    action: RouteAction
    target: RouteTarget | None = None
    diagnosis_intent: DiagnosisIntent | None = None
    rag_tier: RagTier | None = None
    rewritten_query: str | None = None
    answer_mode: TopRouterAnswerMode | None = None
    direct_message: str | None = None
    reasoning: str | None = None


def router_payload(state: AgentState) -> dict[str, Any]:
    return {
        "target_locale": state.get("locale") or "zh-CN",
        "current_user_input": state.get("latest_user_text") or "",
        "visible_conversation_history": recent_dialogue(state),
    }


def diagnosis_context(state: AgentState) -> dict[str, Any]:
    phase = _diagnosis_phase(state)
    return {
        "phase": _workflow_phase(state, phase),
        "diagnosis_phase": phase,
        "pending": _pending_label(state),
        "pending_stage": _pending_stage(state),
        "pending_kind": _pending_kind(state),
        "pending_field": _pending_field(state),
        "pending_prompt": _pending_prompt(state),
        "confirmed_defect": state.get("defect_category"),
        "confirmed_method": state.get("chosen_method"),
        "last_candidates": _last_candidates(state),
        "known_features": {
            "material": state.get("material"),
            **(state.get("known_features") or {}),
        },
        "active_interrupt": plain_model(state.get("interrupt")),
    }


def recent_dialogue(state: AgentState) -> list[dict[str, str]]:
    entries = [_message_entry(item) for item in state.get("messages") or []]
    return [entry for entry in entries if entry][-RECENT_DIALOGUE_LIMIT:]


def _diagnosis_phase(state: AgentState) -> str:
    interrupt = state.get("interrupt")
    kind = interrupt.kind if interrupt else None
    if kind == DISEASE_SELECTION:
        return "awaiting_defect_choice"
    if kind == METHOD_SELECTION:
        return "awaiting_method_choice"
    if kind in (DISEASE_CLARIFICATION, METHOD_CLARIFICATION):
        return "awaiting_clarification"
    if _has_candidates(state.get("method_discriminator_output")):
        return "awaiting_method_choice"
    if _has_candidates(state.get("disease_discriminator_output")):
        return "awaiting_defect_choice"
    if not state.get("defect_category"):
        return "collecting"
    if not state.get("chosen_method"):
        return "method_needed"
    if state.get("final_answer") is None:
        return "detail_needed"
    return "done"


def _workflow_phase(state: AgentState, diagnosis_phase: str) -> str:
    if diagnosis_phase == "done":
        return "DONE"
    if _has_diagnosis_state(state):
        return "DIAGNOSING"
    return "IDLE"


def _pending_label(state: AgentState) -> str | None:
    interrupt = state.get("interrupt")
    if not interrupt:
        return None
    if interrupt.kind == DISEASE_SELECTION:
        return "待用户选择病害"
    if interrupt.kind == METHOD_SELECTION:
        return "待用户选择处治方法"
    field = _pending_field(state)
    return f"待补充{_field_label(field)}" if field else interrupt.prompt


def _pending_stage(state: AgentState) -> str | None:
    interrupt = state.get("interrupt")
    kind = interrupt.kind if interrupt else None
    if kind in (DISEASE_CLARIFICATION, DISEASE_SELECTION):
        return "disease"
    if kind in (METHOD_CLARIFICATION, METHOD_SELECTION):
        return "method"
    if not state.get("defect_category"):
        return "disease"
    if not state.get("chosen_method"):
        return "method"
    if state.get("final_answer") is None:
        return "detail"
    return None


def _pending_kind(state: AgentState) -> str | None:
    interrupt = state.get("interrupt")
    return interrupt.kind if interrupt else None


def _pending_prompt(state: AgentState) -> str | None:
    interrupt = state.get("interrupt")
    return interrupt.prompt if interrupt else None


def _pending_field(state: AgentState) -> str | None:
    interrupt = state.get("interrupt")
    if interrupt and interrupt.required_fields:
        return canonical_key(interrupt.required_fields[0])
    output = state.get("method_discriminator_output") or state.get("disease_discriminator_output")
    data = plain_model(output) if output else {}
    field = data.get("missing_feature") if isinstance(data, dict) else None
    return canonical_key(field) if field else None


def _field_label(field: str | None) -> str:
    if not field:
        return "现场信息"
    return FIELD_LABELS.get(field, field)


def _last_candidates(state: AgentState) -> list[dict[str, Any]]:
    interrupt = state.get("interrupt")
    kind = interrupt.kind if interrupt else None
    if kind == DISEASE_SELECTION:
        return _candidate_rows(state.get("disease_discriminator_output"))
    if kind == METHOD_SELECTION:
        return _candidate_rows(state.get("method_discriminator_output"))
    if interrupt:
        return []
    method = _candidate_rows(state.get("method_discriminator_output"))
    return method or _candidate_rows(state.get("disease_discriminator_output"))


def _candidate_rows(output: Any) -> list[dict[str, Any]]:
    data = plain_model(output) if output else {}
    candidates = data.get("candidates") if isinstance(data, dict) else None
    rows: list[dict[str, Any]] = []
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        rows.append(
            {
                "name": candidate.get("name"),
                "confidence": candidate.get("confidence"),
            }
        )
    return rows[:LAST_CANDIDATES_LIMIT]


def _has_candidates(output: Any) -> bool:
    return bool(_candidate_rows(output))


def _message_entry(message: Any) -> dict[str, str] | None:
    content = getattr(message, "content", None)
    role = getattr(message, "type", None)
    if isinstance(message, dict):
        content = message.get("content", content)
        role = message.get("type") or message.get("role") or role
    if content is None:
        return None
    return {"role": _role_label(role), "text": str(content)[:MESSAGE_SNIPPET_LIMIT]}


def _role_label(role: Any) -> str:
    if role in {"human", "user"}:
        return "user"
    if role in {"ai", "assistant"}:
        return "assistant"
    return str(role or "message")


def _has_diagnosis_state(state: AgentState) -> bool:
    return bool(
        state.get("known_features")
        or state.get("material")
        or state.get("defect_category")
        or state.get("chosen_method")
        or state.get("interrupt")
        or state.get("disease_discriminator_output")
        or state.get("method_discriminator_output")
    )
