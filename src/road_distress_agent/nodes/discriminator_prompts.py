"""Localized prompts shared by discriminator HITL routing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def selection_prompt(
    label_zh: str,
    label_en: str,
    count: int,
    *,
    state: Mapping[str, Any],
) -> str:
    if state.get("locale") == "en-US":
        if count == 1:
            return f"The system identified the following {label_en}. Please confirm it."
        return f"The system identified these possible {label_en}s. Please choose the best match."
    if count == 1:
        return f"系统识别到以下{label_zh}，请确认是否符合现场情况："
    return f"系统识别到以下几种可能的{label_zh}，请选择最符合现场情况的一种："


def more_site_info_prompt(state: Mapping[str, Any]) -> str:
    if state.get("locale") == "en-US":
        return "Please add more observable site information."
    return "请补充更多现场特征信息。"
