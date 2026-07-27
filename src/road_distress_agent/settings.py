"""Runtime mode and provider selection helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

RuntimeMode = Literal["live", "dev"]

RUN_MODE_ENV = "ROAD_DISTRESS_RUN_MODE"
KB_QUERY_PLANNING_ENV = "KB_QUERY_PLANNING_ENABLED"
DETERMINISTIC_FINAL_RENDERER_ENV = "ROAD_DISTRESS_DETERMINISTIC_FINAL_RENDERER_ENABLED"
DETERMINISTIC_CONFIRM_ROUTE_ENV = "ROAD_DISTRESS_DETERMINISTIC_CONFIRM_ROUTE_ENABLED"
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class RuntimeProfile:
    mode: RuntimeMode
    llm: str
    rag: str
    vision: str
    weather: str


def _truthy(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def normalize_run_mode(value: str | None) -> RuntimeMode:
    """Return the active run mode. Missing/blank means real user mode."""
    normalized = (value or "live").strip().lower()
    if normalized in {"live", "prod", "production", "user"}:
        return "live"
    if normalized in {"dev", "debug", "mock", "offline"}:
        return "dev"
    raise ValueError(
        f"Invalid {RUN_MODE_ENV}={value!r}; expected 'live' for real use or 'dev' for debugging."
    )


def get_run_mode() -> RuntimeMode:
    return normalize_run_mode(os.environ.get(RUN_MODE_ENV))


def apply_runtime_mode(mode: str | None = None) -> RuntimeMode:
    """Apply a user-facing mode to the legacy provider flags for this process."""
    selected = normalize_run_mode(mode or os.environ.get(RUN_MODE_ENV))
    os.environ[RUN_MODE_ENV] = selected
    if selected == "live":
        os.environ["ROAD_DISTRESS_DISABLE_LLM"] = "0"
        os.environ["ROAD_DISTRESS_USE_REAL_RAG"] = "1"
        os.environ["ROAD_DISTRESS_USE_REAL_VISION"] = "1"
        os.environ["ROAD_DISTRESS_USE_REAL_WEATHER"] = "1"
        os.environ.setdefault(KB_QUERY_PLANNING_ENV, "1")
    else:
        os.environ["ROAD_DISTRESS_DISABLE_LLM"] = "1"
        os.environ["ROAD_DISTRESS_USE_REAL_RAG"] = "0"
        os.environ["ROAD_DISTRESS_USE_REAL_VISION"] = "0"
        os.environ["ROAD_DISTRESS_USE_REAL_WEATHER"] = "0"
    return selected


def llm_disabled() -> bool:
    explicit = _truthy(os.environ.get("ROAD_DISTRESS_DISABLE_LLM"))
    if explicit is not None:
        return explicit
    return get_run_mode() == "dev"


def deterministic_final_renderer_enabled() -> bool:
    """Return whether the B-06 deterministic final renderer is enabled."""
    raw = os.environ.get(DETERMINISTIC_FINAL_RENDERER_ENV)
    if raw is None:
        return False
    explicit = _truthy(raw)
    if explicit is None:
        raise ValueError(
            f"Invalid {DETERMINISTIC_FINAL_RENDERER_ENV}={raw!r}; expected a boolean value."
        )
    return explicit


def deterministic_confirm_route_enabled() -> bool:
    """Return whether the B-04 deterministic confirmation route is enabled."""
    raw = os.environ.get(DETERMINISTIC_CONFIRM_ROUTE_ENV)
    if raw is None:
        return False
    explicit = _truthy(raw)
    if explicit is None:
        raise ValueError(
            f"Invalid {DETERMINISTIC_CONFIRM_ROUTE_ENV}={raw!r}; expected a boolean value."
        )
    return explicit


def use_real_rag() -> bool:
    explicit = _truthy(os.environ.get("ROAD_DISTRESS_USE_REAL_RAG"))
    if explicit is not None:
        return explicit
    return get_run_mode() == "live"


def use_real_vision() -> bool:
    explicit = _truthy(os.environ.get("ROAD_DISTRESS_USE_REAL_VISION"))
    if explicit is not None:
        return explicit
    return get_run_mode() == "live"


def use_real_weather() -> bool:
    explicit = _truthy(os.environ.get("ROAD_DISTRESS_USE_REAL_WEATHER"))
    if explicit is not None:
        return explicit
    return get_run_mode() == "live"


def runtime_profile() -> RuntimeProfile:
    return RuntimeProfile(
        mode=get_run_mode(),
        llm="DeepSeek" if not llm_disabled() else "offline fallback",
        rag="Qdrant" if use_real_rag() else "mock RAG",
        vision="Qwen VL" if use_real_vision() else "mock vision",
        weather="APISpace + wttr.in" if use_real_weather() else "mock weather",
    )


def missing_runtime_requirements(*, has_image: bool = False) -> list[str]:
    """Return missing env names that would make live mode silently degrade."""
    if get_run_mode() != "live":
        return []
    required = [
        "DEEPSEEK_API_KEY",
        "QDRANT_URL",
        "QDRANT_COLLECTION",
        "APISPACE_TOKEN",
    ]
    if has_image:
        required.append("DASHSCOPE_API_KEY")
    return [name for name in required if not os.environ.get(name)]
