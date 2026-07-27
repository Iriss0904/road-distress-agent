"""Helpers for starting or resuming one user turn."""

from __future__ import annotations

from typing import Any

from langgraph.graph import START

from road_distress_agent.factories import begin_user_turn_patch, make_initial_state
from road_distress_agent.localization import DEFAULT_LOCALE
from road_distress_agent.resume import WEATHER_LOCATION_REQUEST
from road_distress_agent.state import AgentState, AttachmentRef


def prepare_user_turn(
    *,
    graph: Any,
    config: dict[str, Any],
    existing: Any,
    user_id: str,
    thread_id: str,
    text: str,
    locale: str = DEFAULT_LOCALE,
    attachments: list[AttachmentRef],
    request_id: str | None = None,
) -> AgentState | None:
    """Return stream input for a new thread, or update checkpoint for a resume."""
    if not existing.values:
        return make_initial_state(
            user_id=user_id,
            thread_id=thread_id,
            latest_user_text=text,
            locale=locale,
            latest_attachments=attachments,
            request_id=request_id,
        )

    if request_id is not None and existing.values.get("request_id") == request_id:
        return None

    patch = begin_user_turn_patch(
        text=text, locale=locale, attachments=attachments, request_id=request_id
    )
    if not _preserve_final_display(existing.values):
        patch["final_answer_message"] = None
        patch["final_answer_display"] = None
        patch["reference_index"] = []
    if existing.next:
        graph.update_state(config, patch, as_node=START)
        return None

    graph.update_state(config, patch, as_node=START)
    return None


def _preserve_final_display(values: dict[str, Any]) -> bool:
    interrupt = values.get("interrupt") if values else None
    kind = getattr(interrupt, "kind", None) if interrupt else None
    return kind == WEATHER_LOCATION_REQUEST
