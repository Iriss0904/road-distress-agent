"""B-09 speculative detail retrieval during method-selection HITL pauses."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from road_distress_agent.speculative_prefetch_registry import (
    DEFAULT_REGISTRY,
    PrefetchEntry,
    PrefetchRequest,
)
from road_distress_agent.state import AgentState, AuditEvent, InterruptState
from road_distress_agent.tracing import extract_trace_events, to_plain, trace_event

B09_PREFETCH_ENV = "ROAD_DISTRESS_B09_SPECULATIVE_PREFETCH"
DEFAULT_PREFETCH_VALUE = "0"
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class SpeculativePrefetchError(RuntimeError):
    """Raised when the selected top-1 prefetch failed in the background."""


def speculative_prefetch_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    raw = values.get(B09_PREFETCH_ENV, DEFAULT_PREFETCH_VALUE).strip().lower()
    if raw in ENV_TRUE_VALUES:
        return True
    if raw in ENV_FALSE_VALUES:
        return False
    raise ValueError(f"{B09_PREFETCH_ENV} must be true/false, got {raw!r}.")


def start_detail_prefetch(
    state: AgentState,
    interrupt: InterruptState,
    top_method: str,
) -> tuple[dict[str, Any], list[AuditEvent]]:
    if not speculative_prefetch_enabled():
        return {}, []
    request = _prefetch_request(state, interrupt.interrupt_id, top_method)
    worker_state = deepcopy({**state, "chosen_method": top_method})
    DEFAULT_REGISTRY.start(request, lambda: _run_detail_node(worker_state))
    metadata = {
        "thread_id": request.thread_id,
        "interrupt_id": interrupt.interrupt_id,
        "fingerprint": request.fingerprint,
        "top_method": top_method,
    }
    event = _prefetch_event("prefetch_started", metadata)
    return {"speculative_prefetch": metadata}, [event]


def attach_detail_prefetch(
    state: AgentState,
    interrupt: InterruptState,
    delta: AgentState,
) -> AgentState:
    top_method = interrupt.candidate_ids[0]
    patch, events = start_detail_prefetch(state, interrupt, top_method)
    return {**delta, **patch, "audit_log": [*delta.get("audit_log", []), *events]}


def resolve_detail_prefetch(
    state: AgentState,
    serial_call: Callable[[AgentState], AgentState],
) -> AgentState:
    if not speculative_prefetch_enabled():
        return serial_call(state)
    metadata = state.get("speculative_prefetch")
    if not isinstance(metadata, dict):
        return _serial_miss(state, serial_call, "no_prefetch_metadata")
    selected = str(state.get("chosen_method") or "")
    thread_id = str(state.get("thread_id") or "")
    entry = DEFAULT_REGISTRY.pop(thread_id)
    mismatch = _mismatch_reason(state, metadata, entry, selected)
    if mismatch:
        if entry:
            entry.future.cancel()
        return _serial_miss(state, serial_call, mismatch, entry=entry, discarded=True)
    return _prefetch_hit(entry)


def discard_prefetch_for_route(
    state: AgentState,
    *,
    action: str,
    diagnosis_intent: str | None,
) -> tuple[dict[str, Any], list[AuditEvent]]:
    metadata = state.get("speculative_prefetch")
    keeps_prefetch = action == "diagnosis_proceed" and diagnosis_intent == "candidate_choice"
    if not speculative_prefetch_enabled() or not isinstance(metadata, dict) or keeps_prefetch:
        return {}, []
    entry = DEFAULT_REGISTRY.discard(str(state.get("thread_id") or ""))
    payload = {**metadata, "reason": f"route:{action}:{diagnosis_intent or '-'}"}
    payload.update(_cost_summary(entry))
    return {"speculative_prefetch": None}, [_prefetch_event("prefetch_discard", payload)]


def _prefetch_request(state: AgentState, interrupt_id: str, method: str) -> PrefetchRequest:
    thread_id = str(state.get("thread_id") or "")
    if not thread_id:
        raise ValueError("B-09 prefetch requires state.thread_id.")
    return PrefetchRequest(thread_id, _state_fingerprint(state, interrupt_id, method), method)


def _state_fingerprint(state: AgentState, interrupt_id: str, method: str) -> str:
    payload = {
        "thread_id": state.get("thread_id"),
        "interrupt_id": interrupt_id,
        "method": method,
        "material": state.get("material"),
        "defect_category": state.get("defect_category"),
        "defect_subtype": state.get("defect_subtype"),
        "known_features": state.get("known_features") or {},
        "raw_user_text": state.get("raw_user_text"),
        "distress": state.get("distress"),
        "method_discriminator_output": state.get("method_discriminator_output"),
    }
    encoded = json.dumps(to_plain(payload), ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mismatch_reason(
    state: AgentState,
    metadata: dict[str, Any],
    entry: PrefetchEntry | None,
    selected: str,
) -> str | None:
    if entry is None:
        return "prefetch_entry_missing"
    if selected != entry.top_method:
        return "selected_non_top1"
    interrupt_id = str(metadata.get("interrupt_id") or "")
    fingerprint = _state_fingerprint(state, interrupt_id, selected)
    if fingerprint != entry.fingerprint or fingerprint != metadata.get("fingerprint"):
        return "state_fingerprint_changed"
    return None


def _prefetch_hit(entry: PrefetchEntry | None) -> AgentState:
    if entry is None:
        raise AssertionError("A prefetch hit requires an entry.")
    try:
        result = entry.future.result()
    except BaseException as exc:
        raise SpeculativePrefetchError(
            f"B-09 detail prefetch failed for top-1 method {entry.top_method!r}."
        ) from exc
    event = _prefetch_event(
        "prefetch_hit",
        {
            "top_method": entry.top_method,
            "fingerprint": entry.fingerprint,
            **_result_cost_summary(result),
        },
    )
    return {
        **result,
        "speculative_prefetch": None,
        "audit_log": [event, *result.get("audit_log", [])],
    }


def _serial_miss(
    state: AgentState,
    serial_call: Callable[[AgentState], AgentState],
    reason: str,
    *,
    entry: PrefetchEntry | None = None,
    discarded: bool = False,
) -> AgentState:
    result = serial_call(state)
    events = []
    if discarded:
        events.append(
            _prefetch_event("prefetch_discard", {"reason": reason, **_cost_summary(entry)})
        )
    events.append(_prefetch_event("prefetch_miss", {"reason": reason}))
    return {
        **result,
        "speculative_prefetch": None,
        "audit_log": [*events, *result.get("audit_log", [])],
    }


def _cost_summary(entry: PrefetchEntry | None) -> dict[str, Any]:
    if entry is None or not entry.future.done() or entry.future.cancelled():
        return {"cost_status": "pending_or_cancelled"}
    if entry.future.exception() is not None:
        return {"cost_status": "failed"}
    return {"cost_status": "complete", **_result_cost_summary(entry.future.result())}


def _result_cost_summary(result: AgentState) -> dict[str, Any]:
    traces = extract_trace_events(result.get("audit_log") or [])
    llm = [trace for trace in traces if trace.get("kind") == "llm_call"]
    usage = [(trace.get("metadata") or {}).get("usage") for trace in llm]
    return {"speculative_llm_calls": len(llm), "speculative_llm_usage": usage}


def _prefetch_event(kind: str, metadata: dict[str, Any]) -> AuditEvent:
    return trace_event(node_name="detail_retriever_v2", kind=kind, title=kind, metadata=metadata)


def _run_detail_node(state: AgentState) -> AgentState:
    from road_distress_agent.nodes.detail_retriever_v2 import _run_detail_retriever_v2

    return _run_detail_retriever_v2(state)
