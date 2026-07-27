"""Deterministic context shared by method query rewriting and selection."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

_FEATURE_LABELS = MappingProxyType(
    {
        "advisory": "补充现场事实",
        "advisory_context": "补充现场要求",
        "base_condition": "基层状况",
        "base_exposed": "基层外露",
        "crack_depth_layer": "裂缝所在层位",
        "crack_orientation": "裂缝方向",
        "crack_orientation_detail": "裂缝走向细节",
        "crack_pattern": "裂缝形态",
        "crack_wall_condition": "裂缝壁状态",
        "crack_width_mm": "最大裂缝宽度Dmax",
        "defect_description": "病害描述",
        "depth_mm": "裂缝所在层位或深度",
        "road_alignment": "路段线形",
        "rut_depth_mm": "车辙深度",
        "severity": "严重程度",
        "water_condition": "水损害状况",
        "width_mm": "宽度",
    }
)
_FACT_COMPARISON_NOISE_RE = re.compile(
    r"(?<![a-z])d(?:max|s)(?![a-z])|裂缝宽度",
    re.IGNORECASE,
)


@dataclass(frozen=True, kw_only=True)
class MethodRetrievalContext:
    material: str | None
    defect_category: str | None
    conditions: tuple[str, ...]
    original_observation: str | None

    def facts_text(self) -> str:
        parts: list[str] = []
        if self.material:
            parts.append(self.material)
        if self.defect_category:
            parts.append(f"已确认病害为{self.defect_category}")
        if self.original_observation:
            parts.append(f"原始现场描述：{self.original_observation}")
        supplements = self._supplemental_conditions()
        if supplements:
            parts.append(f"已确认条件：{'，'.join(supplements)}")
        return "；".join(parts)

    def _supplemental_conditions(self) -> tuple[str, ...]:
        observation = _normalized_text(self.original_observation or "")
        return tuple(
            condition
            for condition in self.conditions
            if _condition_value(condition) not in observation
        )


def method_context_from_state(state: Mapping[str, Any]) -> MethodRetrievalContext:
    distress = state.get("distress")
    material = state.get("material") or _legacy_value(distress, "material")
    defect = state.get("defect_category") or _legacy_value(distress, "defect_category")
    features = state.get("known_features") or _legacy_value(distress, "known_features") or {}
    observation = state.get("raw_user_text") or _legacy_value(distress, "raw_user_text")
    conditions = tuple(
        _condition_text(str(key), value)
        for key, value in sorted(dict(features).items(), key=lambda item: str(item[0]))
        if value is not None and _value_text(str(key), value)
    )
    return MethodRetrievalContext(
        material=material,
        defect_category=defect,
        conditions=conditions,
        original_observation=observation,
    )


def enrich_method_query(query: str, context: MethodRetrievalContext) -> str:
    facts = context.facts_text()
    if not facts:
        return query
    return f"{facts}；检索处治方法选择依据、适用条件、规范命名方法和病害专属维修流程。"


def method_stage_goal(context: MethodRetrievalContext) -> str:
    facts = context.facts_text() or "当前已确认病害"
    return (
        f"处治方案判别：首先选择明确匹配以下全部现场事实并命名方法的规范选用条款：{facts}。"
        "只按单一宽度匹配的通用工法或施工步骤，不得替代条件更完整的选用条款；"
        "原始描述与已确认结构化条件冲突时，以已确认条件为准；"
        "路段线形描述道路所处线形，裂缝走向细节描述裂缝自身几何，两者是独立维度；"
        "裂缝自身有转角不等于道路处于转弯路段；"
        "heading_path仅表示chunk起点；rawtext若含后续编号条款，必须将每个编号条款逐条独立判断，"
        "不得因起始条款不适用而忽略后续条款；适用条件以“或”连接时，任一项满足即为正向适用，"
        "不得要求全部条件同时满足；"
        "若多条规范选用规则分别满足已知条件，最多三个位置应先覆盖互不相同的方法；"
        "同一方法的操作流程或步骤表不得占用第二个位置，挤掉另一个适用方法的规范选用条款。"
    )


def _legacy_value(distress: Any, field: str) -> Any:
    if isinstance(distress, Mapping):
        return distress.get(field)
    return getattr(distress, field, None)


def _condition_text(key: str, value: Any) -> str:
    label = _FEATURE_LABELS.get(key, key)
    return f"{label}为{_value_text(key, value)}"


def _value_text(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        unit = "mm" if key.endswith("_mm") else ""
        return f"{value}{unit}"
    if isinstance(value, Mapping):
        return json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple, set, frozenset)):
        return "、".join(sorted(str(item).strip() for item in value if str(item).strip()))
    return str(value).strip()


def _condition_value(condition: str) -> str:
    _, _, value = condition.partition("为")
    return _normalized_text(value)


def _normalized_text(value: str) -> str:
    punctuation = " ，。；：、,.!?！？（）()[]【】"
    compact = value.translate(str.maketrans("", "", punctuation)).lower()
    return _FACT_COMPARISON_NOISE_RE.sub("", compact)
