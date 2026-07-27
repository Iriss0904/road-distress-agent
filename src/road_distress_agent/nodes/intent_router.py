"""Resume gate for HITL turns.

The lightweight guard owns coarse retrieval classification. This node remains
only as the graph's interrupt-before waypoint for resumed user input.
"""

from __future__ import annotations

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.state import AgentState, AuditEvent

INTENT_ANSWERING = "answering"
INTENT_SUPPLEMENTING = "supplementing"
INTENT_CORRECTING = "correcting"
INTENT_ASKING_META = "asking_meta"
INTENT_OFF_TOPIC = "off_topic"


def intent_router(state: AgentState) -> AgentState:
    """Pass resumed turns to the lightweight guard without business routing."""
    interrupt = state.get("interrupt")
    return {
        "phase": WorkflowPhase.INPUT,
        "next_action": "guard_turn",
        "audit_log": [
            AuditEvent(
                node_name="intent_router",
                message="Resume gate passed control to the lightweight guard.",
                metadata={"pending_interrupt_kind": interrupt.kind if interrupt else None},
            )
        ],
    }
