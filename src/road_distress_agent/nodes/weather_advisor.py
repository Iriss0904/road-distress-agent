"""Append an optional non-normative construction-tip block to the final answer."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.llm_deepseek import deepseek_chat
from road_distress_agent.llm_runtime import invoke_llm_call, llm_timeout_seconds
from road_distress_agent.localization import user_language_instruction
from road_distress_agent.memory_context import construction_tip_memory_block
from road_distress_agent.state import AgentState, AuditEvent, FinalAnswer, WeatherAdvice
from road_distress_agent.tracing import trace_event


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "weather_advisor.txt"
    return prompt_path.read_text(encoding="utf-8")


def _structured_method() -> str:
    return os.environ.get("DEEPSEEK_STRUCTURED_METHOD") or "json_mode"


def _extract_with_llm(state: AgentState) -> tuple[WeatherAdvice, list[Any]]:
    weather = state.get("weather_context")
    final_answer = state.get("final_answer")
    payload: dict[str, Any] = {
        "weather_context": weather.model_dump(exclude_none=True) if weather else None,
        "final_answer_summary": final_answer.summary if final_answer else None,
        "method_selection": final_answer.method_selection if final_answer else None,
        "target_locale": state.get("locale") or "zh-CN",
    }
    messages = [
        SystemMessage(content=_prompt_text()),
        HumanMessage(
            content="\n".join(
                [
                    construction_tip_memory_block(state.get("loaded_memory")),
                    user_language_instruction(state.get("locale")),
                    f"请生成非规范施工贴士：{payload}",
                ]
            )
        ),
    ]
    timeout = llm_timeout_seconds("weather_advisor")
    llm = deepseek_chat(timeout=timeout)
    method = _structured_method()
    kwargs: dict[str, Any] = {"method": method}
    if method != "json_mode":
        kwargs["strict"] = True
    structured_llm = llm.with_structured_output(WeatherAdvice, **kwargs)
    extracted = invoke_llm_call(
        error_context_name="weather_advisor",
        usage_correlation_name="weather_advisor",
        timeout_seconds=timeout,
        call=lambda: structured_llm.invoke(messages),
    )
    if isinstance(extracted, WeatherAdvice):
        return extracted, messages
    return WeatherAdvice.model_validate(extracted), messages


def _llm_trace_event(
    *,
    state: AgentState,
    messages: list[Any],
    output: WeatherAdvice,
) -> AuditEvent:
    final_answer = state.get("final_answer")
    return trace_event(
        node_name="weather_advisor",
        kind="llm_call",
        title="Weather advisor LLM",
        inputs={
            "weather_context": state.get("weather_context"),
            "locale": state.get("locale") or "zh-CN",
            "final_answer_summary": final_answer.summary if final_answer else None,
            "method_selection": final_answer.method_selection if final_answer else None,
            "memory_present": state.get("loaded_memory") is not None,
        },
        prompt=messages,
        output=output,
    )


def _with_weather_advice(
    final_answer: FinalAnswer | None,
    advice: WeatherAdvice,
) -> FinalAnswer | None:
    if final_answer is None:
        return None
    return final_answer.model_copy(
        update={
            "weather_constraints": advice.constraints,
            "weather_advice": advice,
        }
    )


def weather_advisor(state: AgentState) -> AgentState:
    advice, messages = _extract_with_llm(state)
    return {
        "weather_advice": advice,
        "final_answer": _with_weather_advice(state.get("final_answer"), advice),
        "weather_route": "weather_advised",
        "phase": WorkflowPhase.SAFETY,
        "next_action": "run_critic",
        "errors": [],
        "audit_log": [
            _llm_trace_event(state=state, messages=messages, output=advice),
            AuditEvent(
                node_name="weather_advisor",
                message="Generated optional non-normative construction tips.",
                metadata={"source": advice.source, "status": advice.status},
            ),
        ],
    }
