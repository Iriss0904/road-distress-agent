"""Prompt-safe helpers for long-term memory."""

from __future__ import annotations

import json
from typing import Any

from road_distress_agent.state import LoadedMemory

LOCATION_KEYS = (
    "city",
    "location",
    "region",
    "project_area",
    "work_area",
    "organization",
)


def compact_memory(memory: LoadedMemory | None) -> dict[str, Any]:
    if memory is None:
        return {}
    return {
        "user_preferences": _compact_mapping(memory.user_preferences),
        "regional_context": _compact_mapping(memory.regional_context),
        "resource_constraints": _compact_mapping(memory.resource_constraints),
        "case_summaries": [_record_value(item) for item in memory.case_summaries],
    }


def expression_memory_block(memory: LoadedMemory | None) -> str:
    payload = compact_memory(memory)
    if not payload:
        return "（无长期记忆）"
    return "\n".join(
        [
            "【长期记忆，仅可影响表达方式】",
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            "使用边界：不得改变规范判断、检索证据、处治方法、施工步骤或验收指标。",
        ]
    )


def construction_tip_memory_block(memory: LoadedMemory | None) -> str:
    payload = compact_memory(memory)
    if not payload:
        return "（无地区或资源记忆）"
    scoped = {
        "regional_context": payload.get("regional_context") or {},
        "resource_constraints": payload.get("resource_constraints") or {},
    }
    return "\n".join(
        [
            "【长期记忆，仅可用于非规范施工贴士】",
            json.dumps(scoped, ensure_ascii=False, sort_keys=True),
            "使用边界：只能补充建议/提示；不得改写主体规范结论。",
        ]
    )


def memory_location_query(memory: LoadedMemory | None) -> str | None:
    if memory is None:
        return None
    regional = _compact_mapping(memory.regional_context)
    for key in LOCATION_KEYS:
        value = regional.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _compact_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {key: _record_value(value) for key, value in values.items()}


def _record_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    if isinstance(value, dict) and "summary" in value:
        return value["summary"]
    return value
