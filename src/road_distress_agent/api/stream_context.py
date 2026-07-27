"""SSE stream context used to make runtime errors debuggable."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from road_distress_agent.api.serialization import stage_label
from road_distress_agent.errors import BoundaryError, ensure_boundary_error, error_info_payload

NEXT_ACTION_TO_NODE = {
    "detail_phase": "detail_retriever_v2",
    "discriminate_disease": "disease_discriminator",
    "discriminate_method": "method_discriminator",
    "method_phase": "method_selection_handler",
    "retrieve_disease_definition": "disease_retriever",
    "retrieve_method_evidence": "method_retriever",
    "rewrite_disease_query": "disease_query_rewriter",
    "rewrite_method_query": "method_query_rewriter",
}

GRAPH_NEXT_NODE = {
    "disease_query_rewriter": "disease_retriever",
    "disease_retriever": "disease_discriminator",
    "method_query_rewriter": "method_retriever",
    "method_retriever": "method_discriminator",
}


@dataclass(frozen=True)
class CompletedNode:
    node_name: str
    label: str
    next_action: str | None
    traces: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StreamContext:
    thread_id: str
    locale: str
    last_completed_node: str | None = None
    last_completed_label: str | None = None
    last_next_action: str | None = None
    last_trace: dict[str, Any] | None = None

    def record_completed(self, completed: CompletedNode) -> None:
        self.last_completed_node = completed.node_name
        self.last_completed_label = completed.label
        self.last_next_action = completed.next_action
        self.last_trace = _last_trace_summary(completed.traces)

    def error_payload(self, exc: Exception) -> dict[str, Any]:
        likely = self._likely_next_node()
        boundary = _boundary_error(exc, likely)
        info_payload = error_info_payload(boundary.info)
        return {
            **info_payload,
            "type": exc.__class__.__name__,
            "title": self._error_title(info_payload["message"], info_payload["code"], likely),
            "thread_id": self.thread_id,
            "last_completed_node": self.last_completed_node,
            "last_completed_label": self.last_completed_label,
            "last_next_action": self.last_next_action,
            "likely_next_node": likely,
            "likely_next_label": stage_label(likely, self.locale) if likely else None,
            "last_trace": self.last_trace,
        }

    def system_error_text(self, exc: Exception) -> str:
        payload = self.error_payload(exc)
        message = payload["message"]
        hint = payload.get("hint")
        if self.locale == "en-US":
            return _english_error_text(message, payload, hint)
        return _chinese_error_text(message, payload, hint)

    def _likely_next_node(self) -> str | None:
        if self.last_next_action in NEXT_ACTION_TO_NODE:
            return NEXT_ACTION_TO_NODE[self.last_next_action]
        if self.last_completed_node in GRAPH_NEXT_NODE:
            return GRAPH_NEXT_NODE[self.last_completed_node]
        return None

    def _error_title(self, message: str, code: str, likely: str | None) -> str:
        if likely and self.last_completed_node:
            return (
                f"Stream error {code} after {self.last_completed_node}; likely {likely}: {message}"
            )
        if self.last_completed_node:
            return f"Stream error {code} after {self.last_completed_node}: {message}"
        return f"Stream error {code}: {message}"


def completed_node(
    *,
    node_name: str,
    label: str,
    node_delta: Any,
    traces: list[dict[str, Any]],
) -> CompletedNode:
    return CompletedNode(
        node_name=node_name,
        label=label,
        next_action=_next_action(node_delta),
        traces=traces,
    )


def soft_error_payload(*, thread_id: str, error: Any) -> dict[str, Any]:
    payload = _event_payload(error)
    return {
        **payload,
        "thread_id": thread_id,
        "type": "ErrorEvent",
        "title": f"Recoverable error {payload.get('code')}: {payload.get('message')}",
    }


def _next_action(node_delta: Any) -> str | None:
    if not isinstance(node_delta, dict):
        return None
    value = node_delta.get("next_action")
    return value if isinstance(value, str) and value else None


def _last_trace_summary(traces: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not traces:
        return None
    trace = traces[-1]
    return {
        "kind": trace.get("kind"),
        "node": trace.get("node"),
        "timestamp": trace.get("timestamp"),
        "title": trace.get("title"),
    }


def _boundary_error(exc: Exception, likely: str | None) -> BoundaryError:
    return ensure_boundary_error(
        exc,
        default_domain="INTERNAL",
        default_step=likely or "STREAM",
    )


def _event_payload(error: Any) -> dict[str, Any]:
    if hasattr(error, "model_dump"):
        data = error.model_dump(mode="json")
    else:
        data = dict(error) if isinstance(error, dict) else {"message": str(error)}
    return {
        "message": str(data.get("message") or "可恢复错误未提供 message。"),
        "category": data.get("category") or "internal",
        "code": data.get("code") or "INTERNAL.UNCLASSIFIED",
        "step": data.get("step") or data.get("node_name"),
        "retriable": bool(data.get("retriable")),
        "hint": str(data.get("hint") or ""),
        "raw": str(data.get("raw") or data.get("message") or ""),
        "node": data.get("node_name"),
    }


def _chinese_error_text(message: str, payload: dict[str, Any], hint: str | None) -> str:
    details = _error_details(payload, "最后完成", "推测卡在", "：")
    suffix = f"；建议：{hint}" if hint else ""
    if details:
        return f"系统错误：{message}{suffix}（{details}）"
    return f"系统错误：{message}{suffix}"


def _english_error_text(message: str, payload: dict[str, Any], hint: str | None) -> str:
    details = _error_details(payload, "last completed", "likely stuck at", ": ")
    suffix = f"; hint: {hint}" if hint else ""
    if details:
        return f"System error: {message}{suffix} ({details})"
    return f"System error: {message}{suffix}"


def _error_details(
    payload: dict[str, Any],
    completed_label: str,
    likely_label: str,
    separator: str,
) -> str:
    details: list[str] = []
    if payload.get("last_completed_label"):
        details.append(f"{completed_label}{separator}{payload['last_completed_label']}")
    if payload.get("likely_next_label"):
        details.append(f"{likely_label}{separator}{payload['likely_next_label']}")
    return "; ".join(details)
