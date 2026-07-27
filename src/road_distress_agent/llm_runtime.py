"""Runtime configuration and error context for LLM calls."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, TypeVar
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.tracers.context import register_configure_hook

from road_distress_agent.error_classifiers import classify_llm_error
from road_distress_agent.errors import BoundaryError, ErrorCategory, make_error_info
from road_distress_agent.llm_usage import llm_usage_record as _llm_usage_record

DEFAULT_LLM_TIMEOUT_SECONDS = 60
GLOBAL_LLM_TIMEOUT_ENV = "ROAD_DISTRESS_LLM_TIMEOUT_SECONDS"
NODE_LLM_TIMEOUT_ENV_PREFIX = "ROAD_DISTRESS_LLM_TIMEOUT_"
LLM_USAGE_METADATA_KEY = "usage"

T = TypeVar("T")


class LLMCallError(BoundaryError):
    """Raised when a node-level LLM call fails with execution context."""


class LLMTraceMetadataError(BoundaryError):
    """Raised when LLM usage trace metadata cannot be matched exactly."""


class _LLMUsageCallback(BaseCallbackHandler):
    def __init__(self, trace_correlation_id: str) -> None:
        super().__init__()
        self.raise_error = True
        self.trace_correlation_id = trace_correlation_id
        self.records: list[dict[str, Any]] = []

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        record = _llm_usage_record(response, self.trace_correlation_id)
        if record is not None:
            self.records.append(record)


_LLM_USAGE_CALLBACK: ContextVar[_LLMUsageCallback | None] = ContextVar(
    "road_distress_llm_usage_callback",
    default=None,
)
_PENDING_LLM_TRACE_METADATA: ContextVar[tuple[dict[str, Any], ...]] = ContextVar(
    "road_distress_pending_llm_trace_metadata",
    default=(),
)
register_configure_hook(_LLM_USAGE_CALLBACK, inheritable=True)


def llm_timeout_seconds(node_name: str) -> int:
    """Resolve the timeout for a node-level LLM call."""
    specific = _timeout_from_env(_node_timeout_env(node_name))
    if specific is not None:
        return specific
    global_timeout = _timeout_from_env(GLOBAL_LLM_TIMEOUT_ENV)
    if global_timeout is not None:
        return global_timeout
    return DEFAULT_LLM_TIMEOUT_SECONDS


def invoke_llm_call(
    *,
    error_context_name: str,
    usage_correlation_name: str,
    timeout_seconds: int,
    call: Callable[[], T],
) -> T:
    """Run an LLM call and re-raise failures with node and timeout context."""
    callback = _LLMUsageCallback(trace_correlation_id=uuid4().hex)
    token = _LLM_USAGE_CALLBACK.set(callback)
    try:
        result = call()
    except LLMCallError:
        raise
    except Exception as exc:
        info = classify_llm_error(
            exc,
            node_name=error_context_name,
            timeout_seconds=timeout_seconds,
        )
        raise LLMCallError(info, exc) from exc
    finally:
        _LLM_USAGE_CALLBACK.reset(token)
    _push_pending_llm_trace_metadata(usage_correlation_name, callback.records)
    return result


def pop_pending_llm_trace_metadata(node_name: str) -> dict[str, Any]:
    entries = _PENDING_LLM_TRACE_METADATA.get()
    for index, entry in enumerate(entries):
        if entry.get("node_name") != node_name:
            continue
        remaining = entries[:index] + entries[index + 1 :]
        _PENDING_LLM_TRACE_METADATA.set(remaining)
        return {key: value for key, value in entry.items() if key != "node_name"}
    _PENDING_LLM_TRACE_METADATA.set(())
    raise _llm_trace_metadata_error(
        step=node_name,
        reason=f"llm_call trace for {node_name!r} has no matching provider usage.",
        raw=_pending_raw(entries),
    )


def assert_no_pending_llm_trace_metadata(boundary_name: str) -> None:
    entries = _PENDING_LLM_TRACE_METADATA.get()
    if not entries:
        return
    _PENDING_LLM_TRACE_METADATA.set(())
    raise _llm_trace_metadata_error(
        step=boundary_name,
        reason=f"{len(entries)} LLM provider usage record(s) were not claimed by traces.",
        raw=_pending_raw(entries),
    )


def _push_pending_llm_trace_metadata(
    node_name: str,
    records: list[dict[str, Any]],
) -> None:
    additions = tuple({**record, "node_name": node_name} for record in records)
    if additions:
        _PENDING_LLM_TRACE_METADATA.set((*_PENDING_LLM_TRACE_METADATA.get(), *additions))


def _llm_trace_metadata_error(
    *,
    step: str,
    reason: str,
    raw: str,
) -> LLMTraceMetadataError:
    info = make_error_info(
        domain="TRACE",
        step=step,
        category=ErrorCategory.INTERNAL,
        responsibility="LLM token usage trace attribution failed",
        reason=reason,
        hint="Ensure invoke_llm_call usage_correlation_name matches the llm_call trace node.",
        raw=raw,
        retriable=False,
    )
    return LLMTraceMetadataError(info)


def _pending_raw(entries: tuple[dict[str, Any], ...]) -> str:
    pending = [
        {
            "node_name": entry.get("node_name"),
            "prompt_tokens": entry.get("prompt_tokens"),
            "completion_tokens": entry.get("completion_tokens"),
            "total_tokens": entry.get("total_tokens"),
            "prompt_cache_hit_tokens": entry.get("prompt_cache_hit_tokens"),
            "prompt_cache_miss_tokens": entry.get("prompt_cache_miss_tokens"),
            "cache_token_details_status": entry.get("cache_token_details_status"),
            "cache_token_details_unavailable_reason": entry.get(
                "cache_token_details_unavailable_reason"
            ),
            "trace_correlation_id": entry.get("trace_correlation_id"),
        }
        for entry in entries
    ]
    return f"pending_llm_trace_metadata={pending!r}"


def _timeout_from_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer second value, got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer second value, got {raw!r}.")
    return value


def _node_timeout_env(node_name: str) -> str:
    token = "".join(char if char.isalnum() else "_" for char in node_name).upper()
    return f"{NODE_LLM_TIMEOUT_ENV_PREFIX}{token}_SECONDS"
