"""Parse optional construction-arrangement input after the main answer."""

from __future__ import annotations

import re
from uuid import uuid4

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.memory_context import memory_location_query
from road_distress_agent.state import (
    AddressContext,
    AgentState,
    AuditEvent,
    FinalAnswer,
    InterruptState,
    WeatherAdvice,
)

SKIP_REPLIES = {
    "跳过",
    "暂不提供",
    "暂不",
    "不用",
    "不用了",
    "不查",
    "不需要",
    "没有",
    "不知道",
    "skip",
    "no",
    "no thanks",
}
TIP_REQUEST_TERMS = {
    "想看",
    "需要",
    "要",
    "可以",
    "好",
    "施工贴士",
    "施工建议",
    "施工安排",
    "安全建议",
    "天气建议",
    "补充",
    "yes",
    "weather",
    "tip",
    "tips",
    "construction tip",
}
ROAD_CLASS_PATTERNS = (
    ("二级及以下", re.compile(r"二级及以下|二级以下|三级|四级|等外", re.I)),
    ("高速", re.compile(r"高速|expressway|freeway|motorway", re.I)),
    ("二级及以下", re.compile(r"二级|二\s*级|class\s*ii|second[-\s]*class", re.I)),
    ("一级", re.compile(r"一级|一\s*级|class\s*i|first[-\s]*class", re.I)),
)
ROAD_CLASS_STRIP_RE = re.compile(
    r"高速公路|高速|一级公路|一级|一\s*级|二级及以下|二级以下|二级公路|二级|"
    r"二\s*级|三级|四级|等外|expressway|freeway|motorway|class\s*i|"
    r"class\s*ii|first[-\s]*class|second[-\s]*class",
    re.I,
)


def _zip_code(text: str) -> str | None:
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
    return match.group(1) if match else None


def _road_class(text: str) -> str | None:
    for normalized, pattern in ROAD_CLASS_PATTERNS:
        if pattern.search(text):
            return normalized
    return None


def _clean_city_text(text: str) -> str:
    cleaned = ROAD_CLASS_STRIP_RE.sub("", text.strip())
    marker_match = re.search(
        r"(?:城市|city|地点|位置|所在地|施工地|施工地点)\s*(?:是|为|:|：)?\s*([^，,。；;\n]+)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if marker_match:
        cleaned = marker_match.group(1)
    cleaned = re.sub(
        (
            r"(邮编|zipcode|zip|不知道|没有|查询|天气|帮我|请|一下|"
            r"想看|施工贴士|施工建议|施工安排|安全建议|天气建议|非规范|补充|"
            r"根据|地区|习惯|常用材料|需要|可以|给出|给我|看)"
        ),
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" ，,。；;：:")


def _is_skip(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized in SKIP_REPLIES:
        return True
    return len(normalized) <= 8 and any(reply in normalized for reply in SKIP_REPLIES)


def _is_tip_request(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized or _is_skip(text):
        return False
    return any(term in normalized for term in TIP_REQUEST_TERMS)


def _with_weather_advice(
    final_answer: FinalAnswer | None,
    advice: WeatherAdvice,
) -> FinalAnswer | None:
    if final_answer is None:
        return None
    return final_answer.model_copy(
        update={"weather_advice": advice, "weather_constraints": advice.constraints}
    )


def _skip_response(state: AgentState) -> AgentState:
    english = state.get("locale") == "en-US"
    advice = WeatherAdvice(
        status="skipped",
        title="Construction Arrangement Advice" if english else "施工安排建议",
        summary=(
            "The user skipped construction arrangement advice; confirm weather and "
            "work-zone safety requirements before formal construction."
            if english
            else "用户选择跳过施工安排建议；正式施工前仍应复核天气与作业区安全要求。"
        ),
        constraints=[
            (
                "Weather and safety-norm arrangement were not verified in this optional step."
                if english
                else "本次可选步骤未校验天气与安全作业规范安排。"
            )
        ],
        practical_tips=[
            (
                "Before work starts, the site lead should re-check weather, pavement "
                "dryness, traffic control, and worker protection."
                if english
                else "开工前应由现场负责人再次确认天气、路面干燥程度、交通组织和人员防护。"
            )
        ],
        source="user_skip",
    )
    return {
        "weather_route": "skip_weather",
        "weather_advice": advice,
        "final_answer": _with_weather_advice(state.get("final_answer"), advice),
        "awaiting_user_input": False,
        "new_user_input_pending": False,
        "interrupt": None,
        "phase": WorkflowPhase.SAFETY,
        "next_action": "run_critic",
        "audit_log": [
            AuditEvent(
                node_name="weather_location_handler",
                message="User skipped optional construction arrangement advice.",
            )
        ],
    }


def _load_weather_response(
    address: AddressContext,
    road_class: str,
    message: str,
    metadata: dict[str, str],
) -> AgentState:
    return {
        "address_context": address,
        "road_class": road_class,
        "weather_route": "load_weather",
        "awaiting_user_input": False,
        "new_user_input_pending": False,
        "interrupt": None,
        "phase": WorkflowPhase.SAFETY,
        "next_action": "rewrite_safety_norm_query",
        "audit_log": [
            AuditEvent(node_name="weather_location_handler", message=message, metadata=metadata)
        ],
    }


def _safety_only_response(road_class: str) -> AgentState:
    return {
        "address_context": None,
        "weather_context": None,
        "road_class": road_class,
        "weather_route": "safety_norm_only",
        "awaiting_user_input": False,
        "new_user_input_pending": False,
        "interrupt": None,
        "phase": WorkflowPhase.SAFETY,
        "next_action": "rewrite_safety_norm_query",
        "audit_log": [
            AuditEvent(
                node_name="weather_location_handler",
                message="Parsed road class without weather location.",
                metadata={"road_class": road_class},
            )
        ],
    }


def _ask_road_class_response(
    state: AgentState,
    address: AddressContext | None,
) -> AgentState:
    english = state.get("locale") == "en-US"
    prompt = (
        "Reply with the road class: expressway, first-class, or second-class and below. "
        "You may also add a city/ZIP code for weather-window advice."
        if english
        else "请补充公路等级：高速、一级、二级及以下。也可同时附城市或 6 位邮编以获取天气安排。"
    )
    interrupt = InterruptState(
        interrupt_id=f"interrupt-{uuid4().hex[:8]}",
        node_name="await_construction_arrangement",
        kind="weather_location_request",
        prompt=prompt,
    )
    return {
        "address_context": address,
        "weather_route": "ask_road_class",
        "awaiting_user_input": True,
        "new_user_input_pending": False,
        "interrupt": interrupt,
        "interrupts": [interrupt],
        "phase": WorkflowPhase.SAFETY,
        "next_action": "await_road_class",
        "audit_log": [
            AuditEvent(
                node_name="weather_location_handler",
                message="Road class was missing; prepared construction-arrangement prompt.",
            )
        ],
    }


def _memory_location(state: AgentState, text: str) -> str | None:
    if not _is_tip_request(text):
        return None
    return memory_location_query(state.get("loaded_memory"))


def _parsed_address(state: AgentState, text: str) -> AddressContext | None:
    if zip_code := _zip_code(text):
        return AddressContext(
            zip_code=zip_code,
            location_query=zip_code,
            status="pending_zip_lookup",
        )
    if city := _clean_city_text(text):
        return AddressContext(location_query=city, city=city, status="provided_by_user")
    if location := _memory_location(state, text):
        return AddressContext(location_query=location, city=location, status="from_memory")
    return state.get("address_context")


def weather_location_handler(state: AgentState) -> AgentState:
    text = state.get("latest_user_text") or ""
    if _is_skip(text):
        return _skip_response(state)

    road_class = _road_class(text) or state.get("road_class")
    address = _parsed_address(state, text)
    if not road_class:
        return _ask_road_class_response(state, address)
    if address:
        return _load_weather_response(
            address,
            road_class,
            "Parsed road class and weather location.",
            {"road_class": road_class, "location_query": address.location_query or ""},
        )
    return _safety_only_response(road_class)
