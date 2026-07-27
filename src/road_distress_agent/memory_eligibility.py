"""Deterministic eligibility for long-term memory extraction on KB turns."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

ELIGIBLE_REASON: Final = "kb_explicit_persistence_with_allowed_category"
NO_TEXT_REASON: Final = "kb_missing_user_text"
NO_PERSISTENCE_REASON: Final = "kb_no_explicit_persistence_intent"
NO_CATEGORY_REASON: Final = "kb_no_allowed_category_signal"

_PERSISTENCE_PATTERN = re.compile(
    r"记住|记下|保存(?:下来|为长期)?|以后|今后|后续|长期|一直|总是|通常|常年|"
    r"(?:我|我们|本单位|公司|团队|项目组)常"
)
_SUBJECT_PATTERN = re.compile(r"我|我们|本单位|公司|团队|项目组")
_RESPONSE_PATTERN = re.compile(r"回答|回复|答复|输出|表达|格式|措辞|引用")
_PREFERENCE_PATTERN = re.compile(r"表格|列表|分点|先给结论|简洁|简短|短一点|详细|中文|英文|条款")
_LOCATION_PATTERN = re.compile(
    r"(?:在|来自|位于|负责|管养)(?:的)?(?:某地|某城市|某区域|"
    r"[\u4e00-\u9fff]{2,8}(?:省|市|区|县))"
    r"(?=做|从事|负责|进行|项目|道路|养护|工作|，|,|。|；|;|$)"
)
_ROLE_PATTERN = re.compile(
    r"(?:是|担任|作为|角色是|身份是).{0,8}"
    r"(?:监理|巡检员|养护人员|施工人员|项目经理|业主|设计人员|道路养护单位)"
)
_RESOURCE_CONSTRAINT_PATTERN = re.compile(
    r"没有|缺少|缺乏|无可用|只有|常备|固定配备|预算有限|设备有限|材料有限|人手有限"
)
_RESOURCE_PATTERN = re.compile(
    r"铣刨机|灌缝机|压实设备|冷补料|密封胶|设备|材料|班组|预算|人手|车辆|工具"
)


@dataclass(frozen=True, slots=True)
class KBMemoryEligibility:
    """Immutable decision used to gate the existing Memory LLM."""

    eligible: bool
    reason: str
    category_signals: tuple[str, ...] = ()


def assess_kb_memory_eligibility(text: str | None) -> KBMemoryEligibility:
    normalized = "".join((text or "").split())
    if not normalized:
        return KBMemoryEligibility(False, NO_TEXT_REASON)
    if not _PERSISTENCE_PATTERN.search(normalized):
        return KBMemoryEligibility(False, NO_PERSISTENCE_REASON)

    categories = _allowed_categories(normalized)
    if not categories:
        return KBMemoryEligibility(False, NO_CATEGORY_REASON)
    return KBMemoryEligibility(True, ELIGIBLE_REASON, categories)


def _allowed_categories(text: str) -> tuple[str, ...]:
    categories: list[str] = []
    if _has_response_preference(text):
        categories.append("user_preferences")
    if _has_regional_or_role_context(text):
        categories.append("regional_context")
    if _has_resource_constraint(text):
        categories.append("resource_constraints")
    return tuple(categories)


def _has_response_preference(text: str) -> bool:
    return bool(_RESPONSE_PATTERN.search(text) and _PREFERENCE_PATTERN.search(text))


def _has_regional_or_role_context(text: str) -> bool:
    return bool(
        _SUBJECT_PATTERN.search(text)
        and (_LOCATION_PATTERN.search(text) or _ROLE_PATTERN.search(text))
    )


def _has_resource_constraint(text: str) -> bool:
    return bool(
        _SUBJECT_PATTERN.search(text)
        and _RESOURCE_CONSTRAINT_PATTERN.search(text)
        and _RESOURCE_PATTERN.search(text)
    )
