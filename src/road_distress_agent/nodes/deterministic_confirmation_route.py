"""High-precision B-04 shortcut for unambiguous HITL confirmations."""

from __future__ import annotations

import re

from road_distress_agent.nodes.top_router_context import TopRouteResult
from road_distress_agent.resume import DISEASE_SELECTION, METHOD_SELECTION
from road_distress_agent.state import AgentState

SHORTCUT_SOURCE = "deterministic_confirmation_shortcut"
SELECTION_NODES = {
    DISEASE_SELECTION: "disease_result_router",
    METHOD_SELECTION: "method_result_router",
}
QUESTION_FEATURES = (
    "？",
    "?",
    "如何",
    "怎么",
    "为什么",
    "什么",
    "是否",
    "是不是",
    "哪",
    "多少",
    "区别",
    "吗",
)
CORRECTION_FEATURES = (
    "不是",
    "不对",
    "改成",
    "更正",
    "纠正",
    "说错",
    "其实",
    "而是",
    "重新",
)
DECLARATIVE_ENDING = re.compile(r"[。！!]+$")
CHINESE_ORDINALS = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十")
PLAN_REQUEST_SEPARATORS = ("", "，", ",", "。", "；", ";", "：", ":", "！", "!")
ACTION_CONFIRM_PREFIXES = ("用", "就")
ACTION_CONFIRM_TEMPLATES = ("按{name}处置", "按{name}处治", "按{name}处理")


def deterministic_confirmation_route(state: AgentState) -> TopRouteResult | None:
    """Return a route only for an exact, non-question selection confirmation."""
    selection = _selection_context(state)
    if selection is None:
        return None
    kind, candidates = selection
    text = (state.get("latest_user_text") or "").strip()
    if not text or _contains_any(text, (*QUESTION_FEATURES, *CORRECTION_FEATURES)):
        return None
    candidate = _matched_candidate(text, candidates, kind)
    if candidate is None:
        return None
    target = "disease" if kind == DISEASE_SELECTION else "method"
    return TopRouteResult(
        action="diagnosis_proceed",
        target=target,
        diagnosis_intent="candidate_choice",
        reasoning=SHORTCUT_SOURCE,
    )


def _selection_context(state: AgentState) -> tuple[str, tuple[str, ...]] | None:
    interrupt = state.get("interrupt")
    if interrupt is None or SELECTION_NODES.get(interrupt.kind) != interrupt.node_name:
        return None
    candidates = tuple(name.strip() for name in interrupt.candidate_ids if name.strip())
    if not candidates or len(candidates) != len(set(candidates)):
        return None
    return interrupt.kind, candidates


def _matched_candidate(
    text: str,
    candidates: tuple[str, ...],
    kind: str,
) -> str | None:
    normalized = DECLARATIVE_ENDING.sub("", text).strip()
    exact = _exact_candidate(normalized, candidates)
    if exact is not None:
        return exact
    ordinal = _ordinal_candidate(normalized, candidates)
    if ordinal is not None:
        return ordinal
    confirmation = _confirmation_candidate(normalized, candidates, kind)
    if confirmation is not None:
        return confirmation
    action = _action_phrase_candidate(normalized, candidates)
    if action is not None:
        return action
    return _method_plan_candidate(normalized, candidates, kind)


def _exact_candidate(text: str, candidates: tuple[str, ...]) -> str | None:
    matches = [candidate for candidate in candidates if text == candidate]
    return matches[0] if len(matches) == 1 else None


def _ordinal_candidate(text: str, candidates: tuple[str, ...]) -> str | None:
    for index, candidate in enumerate(candidates, start=1):
        chinese = CHINESE_ORDINALS[index - 1] if index <= len(CHINESE_ORDINALS) else None
        forms = {str(index), f"第{index}个", f"第{index}项"}
        if chinese:
            forms.update({f"第{chinese}个", f"第{chinese}项"})
        if text in forms:
            return candidate
    return None


def _confirmation_candidate(
    text: str,
    candidates: tuple[str, ...],
    kind: str,
) -> str | None:
    labels = ("病害", "病害类型") if kind == DISEASE_SELECTION else ("方法", "处治方法", "方案")
    matches = [
        candidate
        for candidate in candidates
        if any(
            text in {f"确认{label}是{candidate}", f"确认{label}为{candidate}"} for label in labels
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _action_phrase_candidate(text: str, candidates: tuple[str, ...]) -> str | None:
    """Match natural 'proceed with {candidate}' confirmations by exact equality."""
    matches = [candidate for candidate in candidates if text in _action_confirm_forms(candidate)]
    return matches[0] if len(matches) == 1 else None


def _action_confirm_forms(candidate: str) -> set[str]:
    forms = {f"{prefix}{candidate}" for prefix in ACTION_CONFIRM_PREFIXES}
    forms.update(template.format(name=candidate) for template in ACTION_CONFIRM_TEMPLATES)
    return forms


def _method_plan_candidate(
    text: str,
    candidates: tuple[str, ...],
    kind: str,
) -> str | None:
    if kind != METHOD_SELECTION:
        return None
    matches = [
        candidate
        for candidate in candidates
        if any(
            text == f"选择{candidate}{separator}请给完整处置方案"
            for separator in PLAN_REQUEST_SEPARATORS
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _contains_any(text: str, features: tuple[str, ...]) -> bool:
    return any(feature in text for feature in features)
