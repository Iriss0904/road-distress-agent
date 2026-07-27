"""Knowledge-aside intent guards for the top-level router."""

from __future__ import annotations

import os

from road_distress_agent.nodes.top_router_context import TopRouterAnswerMode, TopRouteResult
from road_distress_agent.state import AgentState

DISTRESS_TERMS = ("开裂", "裂缝", "裂隙", "龟裂", "坑槽", "车辙", "沉陷", "松散")
RULE_SOURCE_TERMS = ("指引", "指南", "规范", "标准", "规定", "条文", "阈值", "依据")
RULE_QUESTION_TERMS = (
    "多少",
    "是什么",
    "有什么",
    "一样吗",
    "区别",
    "分别",
    "是否",
    "吗",
    "？",
    "?",
)
CONCEPT_RELATION_TERMS = (
    "同一个东西",
    "同一种",
    "同一类",
    "一回事",
    "是否相同",
    "是不是相同",
    "一样吗",
    "等同",
    "别名",
    "区别",
    "不同",
    "分别指",
    "分别是什么",
)
CONCEPT_CONNECTORS = ("和", "与", "及", "、", "/", " and ", " or ")
COMPARE_RELATION_TERMS = ("区别", "不同", "分别指", "分别是什么", "有什么区别")
PROCESS_TERMS = (
    "完整流程",
    "全过程",
    "从判断到验收",
    "从判断到施工到验收",
    "从识别到施工验收",
    "从适用判断",
)
DOMAIN_TERMS = (
    *DISTRESS_TERMS,
    "道路",
    "路面",
    "公路",
    "养护",
    "沥青",
    "水泥混凝土",
    "灌缝",
    "贴缝",
    "压缝",
    "封缝",
    "贴缝带",
    "压缝带",
    "铣刨",
    "挖补",
    "清缝",
    "密封胶",
)
SITE_FACT_TERMS = (
    "现场",
    "该路段",
    "这段路",
    "我测",
    "测得",
    "测出来",
    "宽度是",
    "深度是",
    "高差为",
)
B01_ROUTE_CALIBRATION_ENV = "ROAD_DISTRESS_B01_ROUTE_CALIBRATION_ENABLED"
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
SINGLE_SUBJECT_ENUMERATION_TERMS = (
    "基本要求",
    "验收标准",
    "验收要求",
    "质量要求",
    "多少个测点",
    "取多少个测点",
    "最终宽度值如何确定",
)
TRUE_COMPARE_TERMS = ("区别", "对比", "区分", "优缺点", "是否相同", "各适用于")
CONDITIONAL_DECISION_TERMS = ("应采用哪种", "选择后续", "是否适用", "能否采用")
CONDITION_TERMS = ("如果", "若", "当", "对于", "且", "根据")


def explicit_kb_aside_route(
    state: AgentState,
    proposed: TopRouteResult | None = None,
) -> TopRouteResult | None:
    text = state.get("latest_user_text") or ""
    if not _is_explicit_kb_question(text):
        explicit = None
    elif _is_multi_hop_question(text):
        explicit = multi_hop_kb_route(
            "用户明确询问关系、比较或完整流程，属于多证据知识旁路。",
            _multi_hop_answer_mode(text),
        )
    else:
        explicit = single_hop_kb_route(text, "用户明确询问单点规范知识或适用条件，属于知识旁路。")
    candidate = explicit or proposed
    if _should_calibrate_single_hop(text, candidate):
        return single_hop_kb_route(text, "B-01校准：同一知识对象的多项要求仍属于单点证据查询。")
    if _should_preserve_true_multi_hop(text, candidate):
        return candidate
    return explicit


def single_hop_kb_route(text: str, reasoning: str) -> TopRouteResult:
    return TopRouteResult(
        action="kb_aside",
        rag_tier="single_hop",
        rewritten_query=_normalized_query(text),
        answer_mode="direct",
        reasoning=reasoning,
    )


def multi_hop_kb_route(
    reasoning: str,
    answer_mode: TopRouterAnswerMode = "chain_reasoning",
) -> TopRouteResult:
    return TopRouteResult(
        action="kb_aside",
        rag_tier="multi_hop",
        answer_mode=answer_mode,
        reasoning=reasoning,
    )


def _is_explicit_kb_question(text: str) -> bool:
    if _contains_any(text, SITE_FACT_TERMS):
        return False
    has_normative_source = _contains_any(text, RULE_SOURCE_TERMS)
    asks_about_rule = _contains_any(text, RULE_QUESTION_TERMS)
    has_domain_anchor = _contains_any(text, DOMAIN_TERMS)
    if has_domain_anchor and _is_concept_relation_question(text):
        return True
    if has_normative_source and asks_about_rule and has_domain_anchor:
        return True
    return has_domain_anchor and _contains_any(text, PROCESS_TERMS)


def _is_multi_hop_question(text: str) -> bool:
    if _is_concept_relation_question(text) and _contains_any(text, COMPARE_RELATION_TERMS):
        return True
    return _contains_any(text, PROCESS_TERMS)


def _should_calibrate_single_hop(text: str, route: TopRouteResult | None) -> bool:
    if not b01_route_calibration_enabled() or route is None:
        return False
    if route.action != "kb_aside" or route.rag_tier != "multi_hop":
        return False
    return _is_single_subject_enumeration(text) and not _requires_true_multi_hop(text)


def _is_single_subject_enumeration(text: str) -> bool:
    return _contains_any(text, DOMAIN_TERMS) and _contains_any(
        text, SINGLE_SUBJECT_ENUMERATION_TERMS
    )


def _requires_true_multi_hop(text: str) -> bool:
    if _contains_any(text, TRUE_COMPARE_TERMS) or _contains_any(text, PROCESS_TERMS):
        return True
    return _contains_any(text, CONDITION_TERMS) and _contains_any(text, CONDITIONAL_DECISION_TERMS)


def _should_preserve_true_multi_hop(text: str, route: TopRouteResult | None) -> bool:
    return (
        b01_route_calibration_enabled()
        and route is not None
        and route.action == "kb_aside"
        and route.rag_tier == "multi_hop"
        and not _contains_any(text, SITE_FACT_TERMS)
        and _requires_true_multi_hop(text)
    )


def b01_route_calibration_enabled() -> bool:
    raw = os.environ.get(B01_ROUTE_CALIBRATION_ENV)
    return raw is not None and raw.strip().lower() in ENV_TRUE_VALUES


def _multi_hop_answer_mode(text: str) -> TopRouterAnswerMode:
    if _is_concept_relation_question(text) and _contains_any(text, COMPARE_RELATION_TERMS):
        return "compare_table"
    if _contains_any(text, PROCESS_TERMS):
        return "procedure"
    return "chain_reasoning"


def _is_concept_relation_question(text: str) -> bool:
    return _contains_any(text, CONCEPT_RELATION_TERMS) and _contains_any(text, CONCEPT_CONNECTORS)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term and term in text for term in terms)


def _normalized_query(text: str) -> str:
    return " ".join(text.split())
