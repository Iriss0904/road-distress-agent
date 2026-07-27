"""Runtime timing instrumentation for LangGraph nodes."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from time import perf_counter
from typing import Any, cast

from road_distress_agent.state import AgentState, AuditEvent
from road_distress_agent.tracing import extract_trace_events, trace_event

MS_PER_SECOND = 1000.0
TIMING_DECIMALS = 2


NodeCallable = Callable[[AgentState], Any]


def timed_node(node_name: str, node: NodeCallable) -> NodeCallable:
    """Wrap a graph node and append a node-level timing trace to its delta."""

    @wraps(node)
    def wrapper(state: AgentState) -> Any:
        started = perf_counter()
        delta = node(state)
        duration_ms = (perf_counter() - started) * MS_PER_SECOND
        return append_node_timing(delta, node_name, duration_ms)

    return wrapper


def append_node_timing(delta: Any, node_name: str, duration_ms: float) -> AgentState:
    if not isinstance(delta, dict):
        raise TypeError(f"{node_name} returned {type(delta).__name__}; expected dict delta.")

    audit_log = delta.get("audit_log") or []
    if not isinstance(audit_log, list):
        raise TypeError(f"{node_name}.audit_log must be a list when present.")

    rounded_ms = round(duration_ms, TIMING_DECIMALS)
    timing = node_timing_trace_event(node_name=node_name, duration_ms=rounded_ms)
    return cast(AgentState, {**delta, "audit_log": [*audit_log, timing]})


def elapsed_ms_since(started: float) -> float:
    return round((perf_counter() - started) * MS_PER_SECOND, TIMING_DECIMALS)


def node_timing_trace_event(
    *,
    node_name: str,
    duration_ms: float,
    substage_name: str | None = None,
) -> AuditEvent:
    details: dict[str, Any] = {"duration_ms": duration_ms}
    if substage_name is not None:
        details["substage"] = substage_name
    return trace_event(
        node_name=node_name,
        kind="node_timing",
        title="Retrieval substage runtime" if substage_name else "Node runtime",
        output=details,
        metadata=details,
    )


def turn_timing_trace_event(*, thread_id: str, duration_ms: float) -> AuditEvent:
    return trace_event(
        node_name="turn",
        kind="turn_timing",
        title="Turn runtime",
        output={"thread_id": thread_id, "duration_ms": duration_ms},
        metadata={"thread_id": thread_id, "duration_ms": duration_ms},
    )


def turn_timing_payload(*, thread_id: str, started: float) -> dict[str, Any]:
    event = turn_timing_trace_event(
        thread_id=thread_id,
        duration_ms=elapsed_ms_since(started),
    )
    return extract_trace_events([event])[0]
