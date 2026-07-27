"""Translate LangGraph snapshots into UI-safe JSON payloads."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel

from road_distress_agent.localization import DEFAULT_LOCALE, display_term
from road_distress_agent.localization import stage_label as localized_stage_label
from road_distress_agent.reference_index import build_reference_index
from road_distress_agent.state import Citation
from road_distress_agent.tracing import extract_trace_events

# Map interrupt.kind → UI awaiting_kind. The intent_router is the single
# interrupt_before node now, so we derive the UI kind from state.interrupt.kind.
INTERRUPT_KIND_TO_UI: dict[str, str] = {
    "missing_required_fields": "clarify_missing_info",
    "candidate_selection": "candidate_choice",
    "weather_location_request": "weather_location",
    "disease_clarification": "disease_clarify",
    "disease_selection": "disease_selection",
    "method_clarification": "method_clarify",
    "method_selection": "method_selection",
}


def stage_label(node_name: str, locale: str | None = None) -> str:
    return localized_stage_label(node_name, locale)


def _plain(value: Any) -> Any:
    """Recursively coerce Pydantic models / enums into JSON-safe primitives."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _ui_banner_from_audit(values: dict[str, Any]) -> str | None:
    """Surface the most recent UI banner from the audit log (e.g., redirect events)."""
    audit_log = values.get("audit_log") or []
    for event in reversed(list(audit_log)):
        metadata = getattr(event, "metadata", None) or {}
        banner = metadata.get("ui_banner") if isinstance(metadata, dict) else None
        if banner:
            return str(banner)
    return None


def _direct_message_from_audit(values: dict[str, Any]) -> str | None:
    audit_log = values.get("audit_log") or []
    for event in reversed(list(audit_log)):
        metadata = getattr(event, "metadata", None) or {}
        message = metadata.get("direct_message") if isinstance(metadata, dict) else None
        if message:
            return str(message)
    return None


def _candidate_references(values: dict[str, Any], awaiting_kind: str | None) -> list[Any]:
    if awaiting_kind == "disease_selection":
        return build_reference_index(_citations(values.get("disease_evidence_chunks") or []))
    if awaiting_kind == "method_selection":
        return build_reference_index(_citations(values.get("method_evidence_chunks") or []))
    return []


def _awaiting_message(
    values: dict[str, Any],
    awaiting_kind: str | None,
    locale: str,
    references: list[Any],
) -> str | None:
    if awaiting_kind not in {"disease_selection", "method_selection"}:
        return None
    output = _selection_output(values, awaiting_kind)
    candidates = output.get("candidates") or []
    if not candidates:
        raise ValueError(f"{awaiting_kind} snapshot is missing discriminator candidates.")
    prompt = _interrupt_prompt(values, awaiting_kind)
    lines = [_with_reference_tokens(prompt, references), ""]
    lines.extend(_candidate_lines(candidates, awaiting_kind, locale))
    lines.extend(["", _selection_hint(candidates[0], awaiting_kind, locale)])
    return "\n".join(lines)


def _selection_output(values: dict[str, Any], awaiting_kind: str) -> dict[str, Any]:
    key = (
        "disease_discriminator_output"
        if awaiting_kind == "disease_selection"
        else "method_discriminator_output"
    )
    output = _plain(values.get(key))
    if not isinstance(output, dict):
        raise ValueError(f"{awaiting_kind} snapshot is missing {key}.")
    return output


def _interrupt_prompt(values: dict[str, Any], awaiting_kind: str) -> str:
    interrupt = values.get("interrupt")
    prompt = getattr(interrupt, "prompt", None) if interrupt else None
    if not prompt and isinstance(interrupt, dict):
        prompt = interrupt.get("prompt")
    if not prompt:
        raise ValueError(f"{awaiting_kind} snapshot is missing interrupt.prompt.")
    return str(prompt)


def _candidate_lines(candidates: list[Any], awaiting_kind: str, locale: str) -> list[str]:
    detail_key = "description" if awaiting_kind == "disease_selection" else "reason"
    label = "Confidence" if locale == "en-US" else "置信度"
    lines: list[str] = []
    for index, candidate in enumerate(candidates, 1):
        item = _plain(candidate)
        if not isinstance(item, dict):
            raise ValueError(f"{awaiting_kind} candidate {index} must be an object.")
        name = _candidate_name(item, awaiting_kind, index, locale)
        detail = _candidate_detail(item, detail_key, awaiting_kind, index)
        lines.append(f"{index}. {name} | {label} {_confidence(item, index)}\n   {detail}")
    return lines


def _candidate_name(
    candidate: dict[str, Any],
    awaiting_kind: str,
    index: int,
    locale: str,
) -> str:
    name = candidate.get("name")
    if not name:
        raise ValueError(f"{awaiting_kind} candidate {index} is missing name.")
    return display_term(str(name), locale)


def _candidate_detail(
    candidate: dict[str, Any],
    detail_key: str,
    awaiting_kind: str,
    index: int,
) -> str:
    detail = candidate.get(detail_key)
    if not detail:
        raise ValueError(f"{awaiting_kind} candidate {index} is missing {detail_key}.")
    return str(detail)


def _confidence(candidate: dict[str, Any], index: int) -> str:
    try:
        return f"{float(candidate['confidence']):.2f}"
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"candidate {index} has invalid confidence.") from exc


def _selection_hint(candidate: Any, awaiting_kind: str, locale: str) -> str:
    item = _plain(candidate)
    if not isinstance(item, dict):
        raise ValueError(f"{awaiting_kind} first candidate must be an object.")
    name = _candidate_name(item, awaiting_kind, 1, locale)
    if locale == "en-US":
        if awaiting_kind == "disease_selection":
            return f'You can reply with the distress type, for example "{name}".'
        return f'You can reply with the treatment option, for example "{name}".'
    if awaiting_kind == "disease_selection":
        return f"你可以直接回复病害名称，例如“{name}”。"
    return f"你可以直接回复处治方案名称，例如“{name}”。"


def _with_reference_tokens(text: str, references: list[Any]) -> str:
    tokens = []
    for ref in references:
        ref_id = getattr(ref, "ref_id", None)
        if ref_id is None and isinstance(ref, dict):
            ref_id = ref.get("ref_id")
        if ref_id:
            tokens.append(f"[[{ref_id}]]")
    return f"{text} {' '.join(tokens)}" if tokens else text


def _citations(chunks: list[Any]) -> list[Citation]:
    citations: list[Citation] = []
    for chunk in chunks:
        raw = chunk.model_dump(mode="json") if isinstance(chunk, BaseModel) else chunk
        citations.append(Citation.model_validate(raw))
    return citations


def serialize_snapshot(snapshot: Any, thread_id: str) -> dict[str, Any]:
    """Reduce a LangGraph StateSnapshot to the fields the chat UI needs."""
    values: dict[str, Any] = snapshot.values or {}
    locale = values.get("locale") or DEFAULT_LOCALE
    next_nodes = list(snapshot.next or ())
    awaiting_node = next_nodes[0] if next_nodes else None
    interrupt = values.get("interrupt")
    interrupt_kind = getattr(interrupt, "kind", None) if interrupt else None
    awaiting_kind = INTERRUPT_KIND_TO_UI.get(interrupt_kind) if interrupt_kind else None
    candidate_references = _candidate_references(values, awaiting_kind)
    awaiting_message = _awaiting_message(
        values,
        awaiting_kind,
        locale,
        candidate_references,
    )

    return {
        **_snapshot_status(thread_id, locale, next_nodes, awaiting_node, awaiting_kind, values),
        **_snapshot_context(values),
        **_snapshot_messages(values, candidate_references, awaiting_message),
        **_snapshot_diagnosis(values),
        "trace": extract_trace_events(values.get("audit_log") or []),
    }


def _snapshot_status(
    thread_id: str,
    locale: str,
    next_nodes: list[str],
    awaiting_node: str | None,
    awaiting_kind: str | None,
    values: dict[str, Any],
) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "request_id": values.get("request_id"),
        "locale": locale,
        "next": next_nodes,
        "is_complete": not next_nodes,
        "awaiting_node": awaiting_node,
        "awaiting_kind": awaiting_kind,
        "awaiting_label": stage_label(awaiting_node, locale) if awaiting_node else None,
        "awaiting_user_input": bool(values.get("awaiting_user_input")),
    }


def _snapshot_context(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": _plain(values.get("phase")),
        "interrupt": _plain(values.get("interrupt")),
        "scene_description": values.get("scene_description"),
        "distress": _plain(values.get("distress")),
        "solution_candidates": _plain(values.get("solution_candidates") or []),
        "candidate_selection": _plain(values.get("candidate_selection")),
        "needed_fields": _plain(values.get("needed_fields") or []),
        "discriminator_output": _plain(values.get("discriminator_output")),
        "final_answer": _plain(values.get("final_answer")),
        "final_answer_display": _plain(values.get("final_answer_display")),
        "safety_review": _plain(values.get("safety_review")),
        "user_intent": values.get("user_intent"),
        "guard_decision": _plain(values.get("guard_decision")),
        "standalone_query_plan": _plain(values.get("standalone_query_plan")),
        "top_route": _plain(values.get("top_route")),
        "known_features": _plain(values.get("known_features") or {}),
        "reconcile_result": _plain(values.get("reconcile_result")),
    }


def _snapshot_messages(
    values: dict[str, Any],
    candidate_references: list[Any],
    awaiting_message: str | None,
) -> dict[str, Any]:
    return {
        "direct_message": values.get("direct_message") or _direct_message_from_audit(values),
        "refusal_type": values.get("refusal_type"),
        "final_answer_message": values.get("final_answer_message"),
        "reference_index": _plain(values.get("reference_index") or []),
        "candidate_reference_index": _plain(candidate_references),
        "awaiting_message": awaiting_message,
        "kb_answer": _plain(values.get("kb_answer")),
        "ui_banner": _ui_banner_from_audit(values),
        "errors": _plain(values.get("errors") or []),
    }


def _snapshot_diagnosis(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "disease_discriminator_output": _plain(values.get("disease_discriminator_output")),
        "method_discriminator_output": _plain(values.get("method_discriminator_output")),
        "defect_category": values.get("defect_category"),
        "chosen_method": values.get("chosen_method"),
    }
