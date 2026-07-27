"""Deterministic zh-CN projection for safety-reviewed final answers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from road_distress_agent.final_response_citations import with_clause_citation_tokens
from road_distress_agent.final_response_presenter import PresentedFinalAnswer
from road_distress_agent.localization import DEFAULT_LOCALE, normalize_locale
from road_distress_agent.memory_context import compact_memory
from road_distress_agent.state import FinalAnswer, FinalAnswerDisplay, LoadedMemory, ReferenceItem

SUPPORTED_PREFERENCE_KEYS = frozenset({"layout", "citation_detail"})
TABLE_TERMS = ("表格", "分栏")
ACCEPTANCE_FIRST_TERMS = ("验收", "复核")
CITATION_TERMS = ("引用", "依据", "证据", "条款", "资料")


class LayoutMode(str, Enum):
    STANDARD = "standard"
    TABLE = "table"
    ACCEPTANCE_FIRST = "acceptance_first"


@dataclass(frozen=True)
class RenderSection:
    title: str
    items: tuple[str, ...]


def can_render_deterministically(
    memory: LoadedMemory | None,
    locale: str | None,
) -> bool:
    if normalize_locale(locale) != DEFAULT_LOCALE:
        return False
    preferences = _preferences(memory)
    if not set(preferences).issubset(SUPPORTED_PREFERENCE_KEYS):
        return False
    return _layout_mode(preferences.get("layout")) is not None and _citation_supported(
        preferences.get("citation_detail")
    )


def render_final_answer(
    answer: FinalAnswer,
    memory: LoadedMemory | None,
    references: list[ReferenceItem],
) -> PresentedFinalAnswer:
    preferences = _preferences(memory)
    mode = _layout_mode(preferences.get("layout"))
    if mode is None:
        raise ValueError("Unsupported deterministic final-renderer preferences.")
    sections = _ordered_sections(answer, mode)
    rendered, used_ids = _render_sections(sections, references, mode)
    message = _review_banner(answer) + rendered
    return PresentedFinalAnswer(
        message="\n\n".join(message),
        used_reference_ids=used_ids,
        display=_display(answer),
    )


def _preferences(memory: LoadedMemory | None) -> dict[str, object]:
    values = compact_memory(memory).get("user_preferences") or {}
    return values if isinstance(values, dict) else {}


def _layout_mode(value: object) -> LayoutMode | None:
    if value is None or value == "":
        return LayoutMode.STANDARD
    if not isinstance(value, str):
        return None
    if any(term in value for term in TABLE_TERMS):
        return LayoutMode.TABLE
    if any(term in value for term in ACCEPTANCE_FIRST_TERMS) and "优先" in value:
        return LayoutMode.ACCEPTANCE_FIRST
    return None


def _citation_supported(value: object) -> bool:
    if value is None or value == "":
        return True
    return isinstance(value, str) and any(term in value for term in CITATION_TERMS)


def _ordered_sections(answer: FinalAnswer, mode: LayoutMode) -> list[RenderSection]:
    sections = [
        RenderSection("结论", (answer.summary,)),
        RenderSection("病害判断", _optional(answer.diagnosis)),
        RenderSection("处治方法", _optional(answer.method_selection)),
        RenderSection("施工步骤", tuple(answer.steps)),
        RenderSection("材料与设备", tuple(answer.materials)),
        RenderSection("验收标准", tuple(answer.acceptance_criteria)),
        RenderSection("安全提示", tuple(answer.safety_notes)),
        *_weather_sections(answer),
        RenderSection("复核要点", tuple(answer.evidence_gaps)),
    ]
    present = [section for section in sections if section.items]
    if mode != LayoutMode.ACCEPTANCE_FIRST:
        return present
    priority = {"验收标准": 0, "复核要点": 1}
    return sorted(present, key=lambda section: priority.get(section.title, 2))


def _weather_sections(answer: FinalAnswer) -> list[RenderSection]:
    advice = answer.weather_advice
    if advice is None:
        return [RenderSection("天气约束", tuple(answer.weather_constraints))]
    summary = tuple(item for item in (advice.summary, advice.recommended_window) if item)
    constraints = _unique((*answer.weather_constraints, *advice.constraints))
    return [
        RenderSection(f"{advice.title}（非规范施工贴士）", summary),
        RenderSection("天气约束", constraints),
        RenderSection("临时组织建议", tuple(advice.practical_tips)),
    ]


def _render_sections(
    sections: list[RenderSection],
    references: list[ReferenceItem],
    mode: LayoutMode,
) -> tuple[list[str], list[str]]:
    used_ids: list[str] = []
    prepared: list[tuple[RenderSection, list[str]]] = []
    for section in sections:
        items = [_with_citations(item, references, used_ids) for item in section.items]
        prepared.append((section, items))
    rendered = [_render_section(section.title, items, mode) for section, items in prepared]
    remaining = [reference.ref_id for reference in references if reference.ref_id not in used_ids]
    if remaining:
        rendered.append(f"参考依据：{_citation_tokens(remaining)}")
        used_ids.extend(remaining)
    return rendered, used_ids


def _render_section(title: str, items: list[str], mode: LayoutMode) -> str:
    if mode == LayoutMode.TABLE and len(items) == 1:
        return f"| 项目 | 内容 |\n|---|---|\n| {title} | {items[0]} |"
    if mode == LayoutMode.TABLE:
        rows = "\n".join(f"| {title} {index} | {item} |" for index, item in enumerate(items, 1))
        return f"| 项目 | 内容 |\n|---|---|\n{rows}"
    if len(items) == 1:
        return f"{title}：{items[0]}"
    return f"{title}：\n" + "\n".join(f"- {item}" for item in items)


def _with_citations(text: str, references: list[ReferenceItem], used: list[str]) -> str:
    rendered = with_clause_citation_tokens([text], references)[0]
    fresh = [item.ref_id for item in references if f"[[{item.ref_id}]]" in rendered]
    duplicates = [ref_id for ref_id in fresh if ref_id in used]
    for ref_id in duplicates:
        rendered = rendered.replace(f" [[{ref_id}]]", "")
    fresh = [ref_id for ref_id in fresh if ref_id not in used]
    used.extend(fresh)
    return rendered


def _citation_tokens(ref_ids: list[str]) -> str:
    return " ".join(f"[[{ref_id}]]" for ref_id in ref_ids)


def _review_banner(answer: FinalAnswer) -> list[str]:
    if not answer.need_human_review:
        return []
    reasons = "；".join(answer.evidence_gaps) or "落地前需由专业人员复核。"
    return [f"⚠️ 需要人工复核：{reasons}"]


def _display(answer: FinalAnswer) -> FinalAnswerDisplay:
    advice = answer.weather_advice
    constraints = tuple(answer.weather_constraints)
    if advice is not None:
        constraints = _unique((*constraints, *advice.constraints))
    return FinalAnswerDisplay(
        summary=answer.summary,
        diagnosis=answer.diagnosis,
        method_selection=answer.method_selection,
        steps=list(answer.steps),
        materials=list(answer.materials),
        acceptance_criteria=list(answer.acceptance_criteria),
        safety_notes=list(answer.safety_notes),
        weather_title=advice.title if advice else None,
        weather_summary=advice.summary if advice else None,
        weather_constraints=list(constraints),
        evidence_gaps=list(answer.evidence_gaps),
        need_human_review=answer.need_human_review,
    )


def _optional(value: str | None) -> tuple[str, ...]:
    return (value,) if value else ()


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))
