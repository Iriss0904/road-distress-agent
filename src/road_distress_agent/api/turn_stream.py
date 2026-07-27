"""SSE turn execution for the web chat API."""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from road_distress_agent.api.paths import web_db_path
from road_distress_agent.api.progressive_delivery import (
    ProgressiveDeliveryFeatures,
    final_text_chunks,
    progressive_delivery_features,
)
from road_distress_agent.api.serialization import serialize_snapshot, stage_label
from road_distress_agent.api.sse import sse_event as _sse
from road_distress_agent.api.stream_context import (
    StreamContext,
    completed_node,
    soft_error_payload,
)
from road_distress_agent.api.thread_history import (
    record_snapshot,
    record_system_message,
    record_user_turn,
)
from road_distress_agent.api.turn_request import prepare_turn_request
from road_distress_agent.api.turn_results import save_turn_result
from road_distress_agent.checkpointing import sqlite_checkpointer
from road_distress_agent.error_classifiers import classify_db_error
from road_distress_agent.errors import BoundaryError
from road_distress_agent.graph import build_graph
from road_distress_agent.llm_runtime import assert_no_pending_llm_trace_metadata
from road_distress_agent.localization import normalize_locale
from road_distress_agent.runtime_timing import turn_timing_payload
from road_distress_agent.state import AgentState, AttachmentRef
from road_distress_agent.tracing import extract_trace_events
from road_distress_agent.turns import prepare_user_turn

DB_PATH = str(web_db_path())
LOGGER = logging.getLogger(__name__)

# SQLite + LangGraph checkpoints are not safe for concurrent writes on one
# thread_id, so serialize turns per conversation.
_thread_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)


@dataclass(frozen=True)
class GraphStreamOptions:
    locale: str
    context: StreamContext
    features: ProgressiveDeliveryFeatures


def turn_event_stream(
    *,
    thread_id: str,
    text: str,
    attachments: list[AttachmentRef],
    user_id: str,
    locale: str,
    request_id: str | None = None,
    fingerprint: str | None = None,
    record_user_input: bool = True,
    replay_snapshot: dict[str, Any] | None = None,
    db_path: str | None = None,
) -> Iterator[str]:
    active_db_path = db_path or DB_PATH
    if (request_id is None) != (fingerprint is None):
        raise ValueError("request_id and fingerprint must be provided together.")
    if request_id is None:
        prepared = prepare_turn_request(
            active_db_path,
            request_id=None,
            thread_id=thread_id,
            user_id=user_id,
            locale=locale,
            text=text,
            attachments=attachments,
        )
        request_id = prepared.request_id
        fingerprint = prepared.fingerprint
        record_user_input = prepared.claim.is_new
        replay_snapshot = prepared.claim.snapshot
    started = perf_counter()
    context = StreamContext(thread_id=thread_id, locale=locale)
    features = progressive_delivery_features()
    lock = _thread_locks[thread_id]
    with lock:
        try:
            if replay_snapshot is not None:
                yield _sse(
                    "turn_start",
                    {"thread_id": thread_id, "request_id": request_id, "locale": locale},
                )
                yield _sse("snapshot", replay_snapshot)
                return
            if record_user_input:
                record_user_turn(active_db_path, thread_id=thread_id, user_id=user_id, text=text)
            yield from _run_turn_events(
                thread_id=thread_id,
                text=text,
                attachments=attachments,
                user_id=user_id,
                started=started,
                locale=locale,
                context=context,
                features=features,
                request_id=request_id,
                fingerprint=fingerprint,
                db_path=active_db_path,
            )
        except Exception as exc:  # surfaced to the UI instead of a dead stream
            classified = _classified_stream_exception(exc)
            LOGGER.exception("Turn stream failed")
            _record_error_message(thread_id, user_id, context.system_error_text(classified))
            yield _sse("trace", turn_timing_payload(thread_id=thread_id, started=started))
            yield _sse("error", context.error_payload(classified))


def _run_turn_events(
    *,
    thread_id: str,
    text: str,
    attachments: list[AttachmentRef],
    user_id: str,
    started: float,
    locale: str,
    context: StreamContext,
    features: ProgressiveDeliveryFeatures,
    request_id: str,
    fingerprint: str,
    db_path: str,
) -> Iterator[str]:
    with sqlite_checkpointer(db_path) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        existing = graph.get_state(config)
        stream_input = prepare_user_turn(
            graph=graph,
            config=config,
            existing=existing,
            user_id=user_id,
            thread_id=thread_id,
            text=text,
            locale=locale,
            attachments=attachments,
            request_id=request_id,
        )
        yield _sse(
            "turn_start",
            {"thread_id": thread_id, "request_id": request_id, "locale": locale},
        )
        options = GraphStreamOptions(locale=locale, context=context, features=features)
        yield from _stream_graph_updates(
            graph=graph,
            stream_input=stream_input,
            config=config,
            options=options,
        )
        assert_no_pending_llm_trace_metadata("webui_turn")
        snapshot = graph.get_state(config)
        serialized = serialize_snapshot(snapshot, thread_id)
        serialized = save_turn_result(
            db_path,
            request_id=request_id,
            fingerprint=fingerprint,
            snapshot=serialized,
        )
        record_snapshot(db_path, thread_id=thread_id, user_id=user_id, snapshot=serialized)
        yield _sse("trace", turn_timing_payload(thread_id=thread_id, started=started))
        yield _sse("snapshot", serialized)


def _stream_graph_updates(
    *,
    graph: Any,
    stream_input: AgentState | None,
    config: dict[str, Any],
    options: GraphStreamOptions,
) -> Iterator[str]:
    stream_mode: str | list[str] = "updates"
    if options.features.progressive_stages:
        stream_mode = ["updates", "tasks"]
    for streamed in graph.stream(stream_input, config=config, stream_mode=stream_mode):
        if options.features.progressive_stages:
            mode, chunk = streamed
            if mode == "tasks":
                yield from _task_events(chunk, options)
                continue
        else:
            chunk = streamed
        for node_name, node_delta in chunk.items():
            if node_name.startswith("__"):
                continue
            yield from _node_update_events(node_name, node_delta, options)


def _node_update_events(
    node_name: str,
    node_delta: Any,
    options: GraphStreamOptions,
) -> Iterator[str]:
    label_locale = _locale_from_delta(node_delta, options.locale)
    label = stage_label(node_name, label_locale)
    traces = _traces_from_delta(node_delta)
    options.context.record_completed(
        completed_node(
            node_name=node_name,
            label=label,
            node_delta=node_delta,
            traces=traces,
        )
    )
    if not options.features.progressive_stages:
        yield _sse("stage", {"node": node_name, "label": label})
    for trace in traces:
        yield _sse("trace", trace)
    for error in _surface_errors(node_delta):
        yield _sse(
            "soft_error",
            soft_error_payload(thread_id=options.context.thread_id, error=error),
        )
    if options.features.progressive_stages:
        yield _sse("stage_complete", {"node": node_name, "label": label})
    elif options.features.final_text_stream and node_name == "safety_critic":
        yield _sse("stage_complete", {"node": node_name, "label": label})
    yield from _final_text_events(node_name, node_delta, options.features)


def _task_events(task: Any, options: GraphStreamOptions) -> Iterator[str]:
    if not isinstance(task, dict) or "result" in task:
        return
    node_name = task.get("name")
    if not isinstance(node_name, str) or node_name.startswith("__"):
        return
    yield _sse("stage", {"node": node_name, "label": stage_label(node_name, options.locale)})


def _final_text_events(
    node_name: str,
    node_delta: Any,
    features: ProgressiveDeliveryFeatures,
) -> Iterator[str]:
    if not features.final_text_stream or node_name != "safety_critic":
        return
    if not isinstance(node_delta, dict):
        raise TypeError("safety_critic update must be an object.")
    final_text = node_delta.get("final_answer_message")
    if final_text is None:
        return
    for payload in final_text_chunks(final_text):
        yield _sse("answer_chunk", payload)


def _traces_from_delta(node_delta: Any) -> list[dict[str, Any]]:
    if not isinstance(node_delta, dict):
        return []
    return extract_trace_events(node_delta.get("audit_log") or [])


def _locale_from_delta(node_delta: Any, default_locale: str) -> str:
    if isinstance(node_delta, dict) and node_delta.get("locale"):
        return normalize_locale(node_delta.get("locale"))
    return default_locale


def _surface_errors(node_delta: Any) -> list[Any]:
    if not isinstance(node_delta, dict):
        return []
    errors = node_delta.get("errors") or []
    return [error for error in errors if _surface_to_user(error)]


def _surface_to_user(error: Any) -> bool:
    if isinstance(error, dict):
        return bool(error.get("surface_to_user"))
    return bool(getattr(error, "surface_to_user", False))


def _classified_stream_exception(exc: Exception) -> Exception:
    if isinstance(exc, BoundaryError):
        return exc
    if isinstance(exc, sqlite3.Error):
        return BoundaryError(classify_db_error(exc, step="WEB_DB"), exc)
    return exc


def _record_error_message(thread_id: str, user_id: str, text: str) -> None:
    try:
        record_system_message(DB_PATH, thread_id=thread_id, user_id=user_id, text=text)
    except Exception:
        LOGGER.exception("Failed to persist system error message")
