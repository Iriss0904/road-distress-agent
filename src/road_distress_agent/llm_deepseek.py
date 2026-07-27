"""Shared DeepSeek text-model configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from langchain_deepseek import ChatDeepSeek

from road_distress_agent.llm_model_policy import (
    DEFAULT_PRO_MODEL,
    configured_pro_model,
    model_assignment,
)

DEFAULT_DEEPSEEK_MODEL = DEFAULT_PRO_MODEL
DEFAULT_DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_BETA_API_BASE = "https://api.deepseek.com/beta"
DEFAULT_DEEPSEEK_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"


def deepseek_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise KeyError("DEEPSEEK_API_KEY is required for DeepSeek LLM calls.")
    return key


def deepseek_text_model() -> str:
    return configured_pro_model()


def deepseek_api_base() -> str:
    return _env("DEEPSEEK_API_BASE", DEFAULT_DEEPSEEK_API_BASE)


def deepseek_beta_api_base() -> str:
    return _env("DEEPSEEK_BETA_API_BASE", DEFAULT_DEEPSEEK_BETA_API_BASE)


def contextual_model() -> str:
    return _env("CONTEXTUAL_LLM_MODEL", deepseek_text_model())


def contextual_base_url() -> str:
    return _env("CONTEXTUAL_LLM_BASE_URL", DEFAULT_DEEPSEEK_ANTHROPIC_BASE_URL)


def disabled_thinking() -> dict[str, str]:
    return {"type": "disabled"}


def disabled_thinking_extra_body() -> dict[str, dict[str, str]]:
    return {"thinking": disabled_thinking()}


def deepseek_chat(
    *,
    timeout: int,
    max_retries: int = 1,
    api_base: str | None = None,
    model: str | None = None,
    correlation_name: str | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> ChatDeepSeek:
    selected_model = model
    if selected_model is None and correlation_name is not None:
        selected_model = model_assignment(correlation_name).model
    kwargs: dict[str, Any] = {
        "api_key": deepseek_api_key(),
        "api_base": api_base or deepseek_api_base(),
        "model": selected_model or deepseek_text_model(),
        "temperature": 0,
        "max_retries": max_retries,
        "timeout": timeout,
        "extra_body": disabled_thinking_extra_body(),
    }
    if model_kwargs:
        kwargs["model_kwargs"] = dict(model_kwargs)
    return ChatDeepSeek(**kwargs)


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default
