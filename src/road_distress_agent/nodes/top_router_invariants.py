"""Explicit deterministic invariants for the top-level router."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from road_distress_agent.nodes.top_router_context import (
    DiagnosisIntent,
    RouteTarget,
    TopRouteResult,
    _has_diagnosis_state,
    _pending_field,
)
from road_distress_agent.nodes.top_router_kb_intent import (
    explicit_kb_aside_route,
    multi_hop_kb_route,
    single_hop_kb_route,
)
from road_distress_agent.nodes.top_router_treatment_intent import initial_treatment_route
from road_distress_agent.resume import (
    DISEASE_SELECTION,
    METHOD_CLARIFICATION,
    METHOD_SELECTION,
)
from road_distress_agent.state import AgentState

NUMERIC_FACT_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*(?:mm|毫米|cm|厘米|m|米|㎡|m2|平方)")
PENDING_KB_TERMS = (
    "怎么测",
    "如何测",
    "测量标准",
    "测量规范",
    "测量方法",
    "量测标准",
    "怎么确定",
    "如何确定",
    "怎么判断",
    "如何判断",
    "判断标准",
    "判定标准",
    "判定依据",
)
TREATMENT_TERMS = ("方案", "处治", "施工", "步骤", "流程", "验收", "维修", "修补")
DIAGNOSIS_TERMS = ("是什么病害", "哪种病害", "什么病害", "病害类型", "识别")
DISTRESS_TERMS = ("开裂", "裂缝", "裂隙", "龟裂", "坑槽", "车辙", "沉陷", "松散")
CORRECTION_TERMS = ("不是", "改成", "说错", "量错", "纠正", "其实")
INTERROGATIVE_TERMS = ("是不是", "是否", "吗", "呢", "有没有", "能不能", "可不可以", "？", "?")
CHOICE_TERMS = ("选择", "选", "就用", "按", "采用", "第一个", "第二个", "第三个")
POST_CHOICE_TERMS = ("检索", "方法", "方案", "处置", "处治", "处理", "维修", "修补")
KB_QUESTION_TERMS = ("多少", "是什么", "有什么", "一样吗", "区别", "分别", "是否", "吗", "？", "?")
SELECTION_KB_TERMS = (
    *KB_QUESTION_TERMS,
    "一样",
    "相同",
    "不同",
    "哪个",
    "为什么",
    "定义",
    "材质",
    "材料",
    "怎么确认",
    "如何确认",
)
PENDING_KB_RULE = "pending_clarification_kb"
SELECTION_RULE = "selection_candidate_kb"
EXPLICIT_RULE = "explicit_kb_boundary"
DIAGNOSIS_RULE = "must_continue_diagnosis"


@dataclass(frozen=True)
class RouteInvariantResult:
    route: TopRouteResult
    conflicts: list[dict[str, Any]]


def apply_route_invariants(
    state: AgentState,
    route: TopRouteResult,
) -> RouteInvariantResult:
    conflicts: list[dict[str, Any]] = []
    current = route
    aside = _pending_kb_aside_route(state)
    if aside:
        current, conflicts = _record_adjustment(current, aside, conflicts, PENDING_KB_RULE)
        return RouteInvariantResult(current, conflicts)
    selection_aside = _selection_kb_aside_route(state)
    if selection_aside:
        current, conflicts = _record_adjustment(current, selection_aside, conflicts, SELECTION_RULE)
        return RouteInvariantResult(current, conflicts)
    explicit_aside = explicit_kb_aside_route(state, current)
    if explicit_aside:
        current, conflicts = _record_adjustment(current, explicit_aside, conflicts, EXPLICIT_RULE)
        return RouteInvariantResult(current, conflicts)

    required = _required_diagnosis_route(state)
    if required:
        current, conflicts = _record_adjustment(current, required, conflicts, DIAGNOSIS_RULE)
    adjusted = _adjust_impossible_target(state, current)
    if adjusted:
        current, conflicts = _record_adjustment(
            current, adjusted, conflicts, "target_requires_prior_state"
        )
    return RouteInvariantResult(current, conflicts)


def _record_adjustment(
    current: TopRouteResult,
    replacement: TopRouteResult,
    conflicts: list[dict[str, Any]],
    rule: str,
) -> tuple[TopRouteResult, list[dict[str, Any]]]:
    if _same_routing_decision(current, replacement):
        return current, conflicts
    conflict = {
        "rule": rule,
        "llm_route": current.model_dump(mode="json"),
        "effective_route": replacement.model_dump(mode="json"),
    }
    return replacement, [*conflicts, conflict]


def _same_routing_decision(left: TopRouteResult, right: TopRouteResult) -> bool:
    same_route = (
        left.action == right.action
        and left.target == right.target
        and left.diagnosis_intent == right.diagnosis_intent
        and left.rag_tier == right.rag_tier
        and left.answer_mode == right.answer_mode
    )
    if not same_route:
        return False
    if left.action != "kb_aside":
        return True
    if left.rag_tier == "single_hop" and not (left.rewritten_query or "").strip():
        return False
    if left.rag_tier == "no_retrieval" and not (left.direct_message or "").strip():
        return False
    return True


def _pending_kb_aside_route(state: AgentState) -> TopRouteResult | None:
    text = state.get("latest_user_text") or ""
    if not _pending_field(state) or not _contains_any(text, PENDING_KB_TERMS):
        return None
    if NUMERIC_FACT_PATTERN.search(text) or _is_correction(text):
        return None
    return single_hop_kb_route(
        text,
        "当前待补充现场信息，但用户是在询问判断或测量标准，属于知识旁路。",
    )


def _selection_kb_aside_route(state: AgentState) -> TopRouteResult | None:
    text = state.get("latest_user_text") or ""
    if not _active_selection_interrupt(state):
        return None
    if _is_candidate_choice(state, text):
        return None
    if not _mentioned_candidate_names(state, text):
        return None
    if not _contains_any(text, SELECTION_KB_TERMS):
        return None
    return multi_hop_kb_route(
        "用户提到候选病害是在提问或比较知识，不是在选择候选。",
        "compare_table",
    )


def _required_diagnosis_route(state: AgentState) -> TopRouteResult | None:
    text = state.get("latest_user_text") or ""
    if _is_candidate_choice(state, text):
        return _diagnosis_route(
            _target_from_interrupt(state), "candidate_choice", "用户在选择候选。"
        )
    if explicit_kb_aside_route(state):
        return None
    if _submits_or_corrects_fact(state, text):
        return _diagnosis_route(
            _target_from_interrupt(state),
            _fact_intent(state, text),
            "用户提交或修正现场事实。",
        )
    if _active_treatment_request(state, text):
        return _treatment_route(state)
    if initial_route := initial_treatment_route(state):
        return initial_route
    if _initial_diagnosis_request(state, text):
        return _diagnosis_route("disease", "advance", "用户请求识别道路病害类型。")
    return None


def _diagnosis_route(
    target: RouteTarget,
    intent: DiagnosisIntent,
    reason: str,
) -> TopRouteResult:
    return TopRouteResult(
        action="diagnosis_proceed",
        target=target,
        diagnosis_intent=intent,
        reasoning=reason,
    )


def _treatment_route(state: AgentState) -> TopRouteResult:
    if state.get("chosen_method"):
        return _diagnosis_route("detail", "advance", "已确认处治方法，用户请求施工或验收详情。")
    return _diagnosis_route("method", "advance", "已确认病害，用户请求处治方案或施工验收。")


def _adjust_impossible_target(
    state: AgentState,
    route: TopRouteResult,
) -> TopRouteResult | None:
    if route.action != "diagnosis_proceed" or route.target is None:
        return None
    if route.target == "detail" and not state.get("chosen_method"):
        target: RouteTarget = "method" if state.get("defect_category") else "disease"
        return route.model_copy(update={"target": target})
    if route.target == "method" and not state.get("defect_category"):
        return route.model_copy(update={"target": "disease"})
    return None


def _target_from_interrupt(state: AgentState) -> RouteTarget:
    interrupt = state.get("interrupt")
    kind = interrupt.kind if interrupt else None
    if kind in (METHOD_CLARIFICATION, METHOD_SELECTION):
        return "method"
    return "disease" if not state.get("defect_category") else "method"


def _fact_intent(state: AgentState, text: str) -> DiagnosisIntent:
    if _is_correction(text):
        return "correct"
    return "answer_interrupt" if state.get("interrupt") else "supplement"


def _is_candidate_choice(state: AgentState, text: str) -> bool:
    if not _active_selection_interrupt(state):
        return False
    names = _mentioned_candidate_names(state, text)
    compacted = _compact(text)
    has_choice_action = _contains_any(text, CHOICE_TERMS)
    has_post_choice_action = _contains_any(text, POST_CHOICE_TERMS)
    has_candidate_confirmation = any(
        re.search(rf"确认(?:病害(?:类型)?)?(?:是|为)?{re.escape(name)}", compacted)
        for name in names
    )
    if not names:
        return has_choice_action
    if _looks_like_selection_kb_question(text) and not has_choice_action:
        return False
    if has_candidate_confirmation and not _is_correction(text):
        return True
    if has_choice_action or has_post_choice_action:
        return True
    return len(names) == 1 and _compact(text) == _compact(names[0])


def _active_selection_interrupt(state: AgentState) -> bool:
    interrupt = state.get("interrupt")
    kind = interrupt.kind if interrupt else None
    return kind in (DISEASE_SELECTION, METHOD_SELECTION)


def _mentioned_candidate_names(state: AgentState, text: str) -> tuple[str, ...]:
    interrupt = state.get("interrupt")
    names = tuple(interrupt.candidate_ids) if interrupt else ()
    return tuple(name for name in names if name and name in text)


def _looks_like_selection_kb_question(text: str) -> bool:
    return _contains_any(text, SELECTION_KB_TERMS)


def _compact(text: str) -> str:
    return re.sub(r"[\s，。？！、；：,.!?;:（）()【】\\[\\]{}《》<>“”\"'`]", "", text)


def _submits_or_corrects_fact(state: AgentState, text: str) -> bool:
    if NUMERIC_FACT_PATTERN.search(text):
        return True
    return bool(state.get("interrupt")) and _is_correction(text)


def _active_treatment_request(state: AgentState, text: str) -> bool:
    has_context = bool(state.get("defect_category") or state.get("chosen_method"))
    return has_context and _contains_any(text, TREATMENT_TERMS)


def _initial_diagnosis_request(state: AgentState, text: str) -> bool:
    if _has_diagnosis_state(state):
        return False
    return _contains_any(text, DIAGNOSIS_TERMS) and _contains_any(text, DISTRESS_TERMS)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term and term in text for term in terms)


def _is_correction(text: str) -> bool:
    """纠正是断言另一个事实，不是提问。"""
    if _contains_any(text, INTERROGATIVE_TERMS):
        return False
    return _contains_any(text, CORRECTION_TERMS)
