"""Format promoted defect facts as formal Chinese report prose."""

from __future__ import annotations

from typing import Any

_SKIP_KEYS = {"location"}


def format_feature_sentence(features: dict[str, Any], *, locale: str = "zh-CN") -> str:
    parts = [
        _format_pair(key, value, locale) for key, value in features.items() if key not in _SKIP_KEYS
    ]
    clean = [part for part in parts if part]
    if clean:
        return ("; ".join(clean) + ".") if locale == "en-US" else "；".join(clean) + "。"
    if locale == "en-US":
        return "No quantifiable site features were recorded."
    return "未记录可量化现场特征。"


def _format_pair(key: str, value: Any, locale: str) -> str:
    if key in _FORMATTERS:
        return _FORMATTERS[key](value, locale)
    raise ValueError(f"report feature formatter missing mapping for {key!r}.")


def _text(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value).strip()


def _number_text(value: Any, unit: str) -> str:
    text = _text(value)
    return text if text.endswith(unit) else f"{text}{unit}"


def _crack_orientation(value: Any, locale: str) -> str:
    if locale == "en-US":
        return f"crack orientation: {_text(value)}"
    return f"裂缝走向为{_text(value)}"


def _crack_length(value: Any, locale: str) -> str:
    if locale == "en-US":
        return f"crack length about {_number_text(value, 'm')}"
    return f"裂缝长度约{_number_text(value, 'm')}"


def _crack_width(value: Any, locale: str) -> str:
    if locale == "en-US":
        return f"maximum crack width about {_number_text(value, 'mm')}"
    return f"最大缝宽约{_number_text(value, 'mm')}"


def _crack_pattern(value: Any, locale: str) -> str:
    if locale == "en-US":
        return f"crack pattern: {_text(value)}"
    return f"裂缝形态为{_text(value)}"


def _rut_depth(value: Any, locale: str) -> str:
    if locale == "en-US":
        return f"rut depth about {_number_text(value, 'mm')}"
    return f"车辙深度约{_number_text(value, 'mm')}"


def _depth(value: Any, locale: str) -> str:
    if locale == "en-US":
        return f"distress depth about {_number_text(value, 'mm')}"
    return f"病害深度约{_number_text(value, 'mm')}"


def _depth_cm(value: Any, locale: str) -> str:
    if locale == "en-US":
        return f"distress depth about {_number_text(value, 'cm')}"
    return f"病害深度约{_number_text(value, 'cm')}"


def _area(value: Any, locale: str) -> str:
    if locale == "en-US":
        return f"affected area about {_number_text(value, 'm²')}"
    return f"影响面积约{_number_text(value, '㎡')}"


def _base_exposed(value: Any, locale: str) -> str:
    if locale == "en-US":
        return "base layer exposed" if bool(value) else "no exposed base layer observed"
    return "已露基层" if bool(value) else "未见基层外露"


def _base_condition(value: Any, locale: str) -> str:
    if locale == "en-US":
        return f"base condition: {_text(value)}"
    return f"基层状况为{_text(value)}"


def _wall_condition(value: Any, locale: str) -> str:
    if locale == "en-US":
        return f"crack-wall condition: {_text(value)}"
    return f"裂缝两侧边缘及缝壁状况为{_text(value)}"


def _water_condition(value: Any, locale: str) -> str:
    if locale == "en-US":
        return f"water condition: {_text(value)}"
    return f"现场水相关情况为{_text(value)}"


def _traffic_load(value: Any, locale: str) -> str:
    if locale == "en-US":
        return f"traffic load or work-zone context: {_text(value)}"
    return f"交通荷载或组织条件为{_text(value)}"


def _severity(value: Any, locale: str) -> str:
    if locale == "en-US":
        return f"site severity: {_text(value)}"
    return f"现场判断严重程度为{_text(value)}"


def _advisory_context(value: Any, locale: str) -> str:
    if locale == "en-US":
        return f"construction-planning context: {_text(value)}"
    return f"施工组织背景为{_text(value)}"


def _key_fact(value: Any, locale: str) -> str:
    if locale == "en-US":
        return f"key identifying fact: {_text(value)}"
    return f"关键识别事实为{_text(value)}"


_FORMATTERS = {
    "crack_orientation": _crack_orientation,
    "crack_length_m": _crack_length,
    "length_m": _crack_length,
    "crack_width_mm": _crack_width,
    "width_mm": _crack_width,
    "crack_pattern": _crack_pattern,
    "rut_depth_mm": _rut_depth,
    "depth_mm": _depth,
    "depth_cm": _depth_cm,
    "area_m2": _area,
    "base_exposed": _base_exposed,
    "base_condition": _base_condition,
    "crack_wall_condition": _wall_condition,
    "water_condition": _water_condition,
    "traffic_load": _traffic_load,
    "severity": _severity,
    "advisory_context": _advisory_context,
    "key_distinguishing_fact": _key_fact,
}
