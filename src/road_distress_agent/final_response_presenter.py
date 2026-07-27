"""LLM presentation layer for reviewed FinalAnswer objects."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from road_distress_agent.llm_deepseek import deepseek_chat
from road_distress_agent.llm_runtime import invoke_llm_call, llm_timeout_seconds
from road_distress_agent.localization import DEFAULT_LOCALE, normalize_locale
from road_distress_agent.memory_context import compact_memory
from road_distress_agent.state import FinalAnswer, FinalAnswerDisplay, LoadedMemory, ReferenceItem


class PresentedFinalAnswer(BaseModel):
    message: str = Field(min_length=1)
    used_reference_ids: list[str] = Field(default_factory=list)
    display: FinalAnswerDisplay | None = None


@lru_cache(maxsize=1)
def _prompt_text() -> str:
    path = Path(__file__).resolve().parent / "prompts" / "final_response_presenter.txt"
    return path.read_text(encoding="utf-8")


def _structured_method() -> str:
    return os.environ.get("DEEPSEEK_STRUCTURED_METHOD") or "json_mode"


def present_final_answer(
    answer: FinalAnswer,
    memory: LoadedMemory | None,
    references: list[ReferenceItem],
    locale: str | None = DEFAULT_LOCALE,
) -> tuple[PresentedFinalAnswer, list[Any]]:
    active_locale = normalize_locale(locale)
    messages = _messages(answer, memory, references, active_locale)
    return _invoke_llm(messages), messages


def _messages(
    answer: FinalAnswer,
    memory: LoadedMemory | None,
    references: list[ReferenceItem],
    locale: str,
) -> list[Any]:
    payload = presentation_payload(answer, memory, references, locale)
    return [
        SystemMessage(content=_prompt_text()),
        HumanMessage(
            content="\n".join(
                [
                    "presentation_input = "
                    + json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    "Return the PresentedFinalAnswer JSON now.",
                ]
            )
        ),
    ]


def presentation_payload(
    answer: FinalAnswer,
    memory: LoadedMemory | None,
    references: list[ReferenceItem],
    locale: str | None = DEFAULT_LOCALE,
) -> dict[str, Any]:
    return {
        "target_locale": normalize_locale(locale),
        "final_answer": answer.model_dump(mode="json"),
        "user_preferences": compact_memory(memory).get("user_preferences") or {},
        "reference_index": [item.model_dump(mode="json") for item in references],
    }


def _invoke_llm(messages: list[Any]) -> PresentedFinalAnswer:
    timeout = llm_timeout_seconds("final_response_presenter")
    llm = deepseek_chat(timeout=timeout)
    method = _structured_method()
    kwargs: dict[str, Any] = {"method": method}
    if method != "json_mode":
        kwargs["strict"] = True
    structured_llm = llm.with_structured_output(PresentedFinalAnswer, **kwargs)
    result = invoke_llm_call(
        error_context_name="final_response_presenter",
        usage_correlation_name="final_response_presenter",
        timeout_seconds=timeout,
        call=lambda: structured_llm.invoke(messages),
    )
    if isinstance(result, PresentedFinalAnswer):
        return result
    return PresentedFinalAnswer.model_validate(result)
