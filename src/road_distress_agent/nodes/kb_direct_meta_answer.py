"""Return router-provided KB meta answers without retrieval or LLM calls."""

from __future__ import annotations

from typing import Any

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.state import AgentState, AuditEvent
from road_distress_agent.tracing import trace_event


def kb_direct_meta_answer(state: AgentState) -> AgentState:
    """Return a no-retrieval KB-side message from the top router."""
    route = _top_route(state)
    direct_message = _direct_message(route)
    return {
        "direct_message": direct_message,
        "awaiting_user_input": state.get("interrupt") is not None,
        "phase": WorkflowPhase.DONE,
        "next_action": "return_kb_direct_meta_answer",
        "audit_log": [
            trace_event(
                node_name="kb_direct_meta_answer",
                kind="stage",
                title="KB direct meta answer",
                inputs={"top_route": route},
                output={"direct_message": direct_message},
            ),
            AuditEvent(
                node_name="kb_direct_meta_answer",
                message="Returned top-router direct KB meta answer.",
                metadata={
                    "rag_tier": route.get("rag_tier"),
                    "direct_message": direct_message,
                },
            ),
        ],
    }


def _top_route(state: AgentState) -> dict[str, Any]:
    route = state.get("top_route")
    if not isinstance(route, dict):
        raise ValueError("kb_direct_meta_answer requires top_route.")
    return route


def _direct_message(route: dict[str, Any]) -> str:
    message = route.get("direct_message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("kb_direct_meta_answer requires top_route.direct_message.")
    return message
