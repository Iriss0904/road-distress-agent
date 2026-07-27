"""Offer optional construction arrangement advice after safety review."""

from __future__ import annotations

from uuid import uuid4

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.state import AgentState, AuditEvent, InterruptState


def construction_tip_offer(state: AgentState) -> AgentState:
    interrupt = InterruptState(
        interrupt_id=f"interrupt-{uuid4().hex[:8]}",
        node_name="await_weather_location",
        kind="weather_location_request",
        prompt=_offer_prompt(state),
    )
    return {
        "construction_tip_offered": True,
        "awaiting_user_input": True,
        "new_user_input_pending": False,
        "interrupt": interrupt,
        "interrupts": [interrupt],
        "phase": WorkflowPhase.SAFETY,
        "next_action": "await_weather_location",
        "audit_log": [
            AuditEvent(
                node_name="construction_tip_offer",
                message="Offered optional construction arrangement advice.",
            )
        ],
    }


def _offer_prompt(state: AgentState) -> str:
    if state.get("locale") == "en-US":
        return (
            "The standards-based construction steps and acceptance criteria are ready. "
            "Would you like optional construction arrangement advice with cited work-zone "
            "safety norms? Reply with the road class (expressway, first-class, or "
            "second-class and below), optionally plus a city/ZIP code for weather-window "
            "advice, or reply 'skip'."
        )
    return (
        "规范施工步骤与验收标准已生成。是否需要补充「施工安排建议」？需要请回复"
        "公路等级（高速/一级/二级及以下），也可附城市或 6 位邮编以获取天气安排；"
        "不需要可回复“跳过”。"
    )
