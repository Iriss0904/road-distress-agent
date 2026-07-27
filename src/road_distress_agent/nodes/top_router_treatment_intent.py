"""Initial-turn treatment intent guards for the top-level router."""

from __future__ import annotations

import re

from road_distress_agent.nodes.top_router_context import (
    TopRouteResult,
    _has_diagnosis_state,
)
from road_distress_agent.state import AgentState

DISTRESS_OBJECT_TERMS = ("开裂", "裂缝", "裂隙", "龟裂", "坑槽", "车辙", "沉陷", "松散", "病害")
TREATMENT_DECISION_TERMS = (
    "怎么修",
    "如何修",
    "怎么处理",
    "如何处理",
    "怎么处治",
    "如何处治",
    "怎么维修",
    "如何维修",
    "用什么方法",
    "用哪种方法",
    "采用什么方法",
    "要不要",
    "需不需要",
    "是否需要",
)
KNOWLEDGE_BOUNDARY_TERMS = (
    "规范",
    "标准",
    "指引",
    "指南",
    "条文",
    "依据",
    "阈值",
    "区别",
    "不同",
    "适用条件",
    "适用场景",
    "完整流程",
    "全过程",
    "从判断到验收",
    "从识别到施工验收",
)


def initial_treatment_route(state: AgentState) -> TopRouteResult | None:
    """Route ambiguous first-turn treatment decisions to diagnosis, not KB."""
    text = state.get("latest_user_text") or ""
    if _has_diagnosis_state(state) or _is_explicit_kb_question(text):
        return None
    if not _mentions_distress_object(text) or not _asks_treatment_decision(text):
        return None
    return TopRouteResult(
        action="diagnosis_proceed",
        target="disease",
        diagnosis_intent="advance",
        reasoning="用户在无诊断上下文时请求处治决策，需先进入诊断链补现场条件。",
    )


def _is_explicit_kb_question(text: str) -> bool:
    return _contains_any(text, KNOWLEDGE_BOUNDARY_TERMS)


def _mentions_distress_object(text: str) -> bool:
    return _contains_any(text, DISTRESS_OBJECT_TERMS)


def _asks_treatment_decision(text: str) -> bool:
    return _contains_any(text, TREATMENT_DECISION_TERMS) or _asks_between_methods(text)


def _asks_between_methods(text: str) -> bool:
    return bool(re.search(r"(?:要|该|应|应该)?用.+还是.+", text))


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term and term in text for term in terms)
