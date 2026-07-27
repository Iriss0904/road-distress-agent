"""Shared LLM plumbing for delivery agent nodes.

Every delivery agent uses a real LLM (no disable/fallback path). Each node keeps
a thin ``_invoke_llm`` seam that calls :func:`invoke_structured`, so tests can
monkeypatch a single node without stubbing the provider globally.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from road_distress_agent.llm_deepseek import deepseek_chat
from road_distress_agent.llm_runtime import invoke_llm_call, llm_timeout_seconds

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=8)
def load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def invoke_structured(
    messages: list[Any],
    schema: type[T],
    *,
    usage_correlation_name: str,
) -> T:
    error_context_name = f"delivery_{schema.__name__}"
    timeout = llm_timeout_seconds(error_context_name)
    llm = deepseek_chat(timeout=timeout)
    method = os.environ.get("DEEPSEEK_STRUCTURED_METHOD") or "json_mode"
    kwargs: dict[str, Any] = {"method": method}
    if method != "json_mode":
        kwargs["strict"] = True
    structured_llm = llm.with_structured_output(schema, **kwargs)
    result = invoke_llm_call(
        error_context_name=error_context_name,
        usage_correlation_name=usage_correlation_name,
        timeout_seconds=timeout,
        call=lambda: structured_llm.invoke(messages),
    )
    return result if isinstance(result, schema) else schema.model_validate(result)
