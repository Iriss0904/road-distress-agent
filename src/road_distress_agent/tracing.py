"""Structured runtime tracing helpers built on top of ``audit_log``."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel

from road_distress_agent.llm_runtime import (
    LLM_USAGE_METADATA_KEY,
    pop_pending_llm_trace_metadata,
)
from road_distress_agent.state import AuditEvent

TRACE_METADATA_KEY = "trace"
TRACE_SCHEMA_VERSION = 1


def to_plain(value: Any) -> Any:
    """Coerce common runtime objects into JSON-safe primitives."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


def message_payloads(messages: list[Any]) -> list[dict[str, Any]]:
    return [_message_payload(message) for message in messages]


def trace_event(
    *,
    node_name: str,
    kind: str,
    title: str,
    inputs: Any | None = None,
    prompt: list[Any] | None = None,
    output: Any | None = None,
    state_delta: Any | None = None,
    retrieval: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> AuditEvent:
    active_metadata = _metadata_with_pending_llm_usage(node_name, kind, metadata)
    payload: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": node_name,
        "kind": kind,
        "title": title,
    }
    _set_if_present(payload, "input", inputs)
    _set_if_present(payload, "prompt", message_payloads(prompt or []) if prompt else None)
    _set_if_present(payload, "output", output)
    _set_if_present(payload, "state_delta", state_delta)
    _set_if_present(payload, "retrieval", retrieval)
    _set_if_present(payload, "metadata", active_metadata)
    return AuditEvent(
        node_name=node_name,
        message=f"trace:{title}",
        metadata={TRACE_METADATA_KEY: to_plain(payload)},
    )


def retrieval_trace_event(
    *,
    node_name: str,
    title: str,
    query: str,
    filters: dict[str, Any],
    chunks: list[Any],
    tool: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> AuditEvent:
    retrieval = {
        "query": query,
        "filters": filters,
        "tool": _tool_payload(tool),
        "chunk_count": len(chunks),
        "chunks": chunk_payloads(chunks),
    }
    return trace_event(
        node_name=node_name,
        kind="retrieval",
        title=title,
        retrieval=retrieval,
        metadata=metadata,
    )


def chunk_payloads(chunks: list[Any]) -> list[dict[str, Any]]:
    return [_chunk_payload(chunk) for chunk in chunks]


def extract_trace_events(audit_log: list[Any]) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for event in audit_log:
        metadata = _event_metadata(event)
        trace = metadata.get(TRACE_METADATA_KEY)
        if not trace:
            continue
        payload = to_plain(trace)
        if isinstance(payload, dict):
            traces.append({"sequence": len(traces) + 1, **payload})
    return traces


def _set_if_present(payload: dict[str, Any], key: str, value: Any | None) -> None:
    if value is not None:
        payload[key] = to_plain(value)


def _metadata_with_pending_llm_usage(
    node_name: str,
    kind: str,
    metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if kind != "llm_call":
        return metadata
    usage = pop_pending_llm_trace_metadata(node_name)
    merged = dict(metadata or {})
    if LLM_USAGE_METADATA_KEY in merged:
        raise ValueError("llm_call trace metadata must not pre-populate provider usage.")
    merged[LLM_USAGE_METADATA_KEY] = usage
    return merged


def _message_payload(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return {
            "role": str(message.get("type") or message.get("role") or "message"),
            "content": message.get("content"),
        }
    return {
        "role": str(getattr(message, "type", None) or message.__class__.__name__),
        "content": getattr(message, "content", None),
    }


def _chunk_payload(chunk: Any) -> dict[str, Any]:
    data = to_plain(chunk)
    if not isinstance(data, dict):
        return {"value": data}
    annotations = data.get("annotations") or {}
    return {
        "rank": data.get("rank"),
        "score": data.get("score") or data.get("similarity"),
        "chunk_id": data.get("chunk_id") or data.get("citation_id"),
        "clause_id": data.get("clause_id") or data.get("source_clause"),
        "semantic_role": annotations.get("semantic_role"),
        "source_doc": data.get("source_doc") or data.get("document_id"),
        "source_pages": data.get("source_pages"),
        "heading_path": data.get("heading_path") or [],
        "context_prefix": data.get("context_prefix"),
        "sequential_group_id": data.get("sequential_group_id"),
        "step_index": data.get("step_index"),
        "text": data.get("text") or data.get("snippet") or "",
    }


def _tool_payload(tool: Any | None) -> dict[str, Any] | None:
    if tool is None:
        return None
    return {
        "name": tool.__class__.__name__,
        "collection": getattr(tool, "_collection", None),
        "top_k": getattr(tool, "_top_k", None),
    }


def _event_metadata(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        metadata = event.get("metadata")
    else:
        metadata = getattr(event, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}
