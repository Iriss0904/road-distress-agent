"""Re-emit the prior interrupt for asking_meta / off_topic replies.

Keeps the workflow paused on the same HITL question so the user can answer it
properly. v1 behavior is "polite nudge" — a richer meta-answering branch can
land later.
"""

from __future__ import annotations

from uuid import uuid4

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.state import AgentState, AuditEvent


def handle_off_intent(state: AgentState) -> AgentState:
    interrupt = state.get("interrupt")
    if interrupt is None:
        return {
            "awaiting_user_input": False,
            "next_action": "ack_done",
            "audit_log": [
                AuditEvent(
                    node_name="handle_off_intent",
                    message="Off-intent reply with no active interrupt; nothing to re-ask.",
                    metadata={"user_intent": state.get("user_intent")},
                )
            ],
        }
    nudged = interrupt.model_copy(
        update={
            "interrupt_id": f"interrupt-{uuid4().hex[:8]}",
            "prompt": _nudge_prompt(state, interrupt.prompt),
        }
    )
    return {
        "interrupt": nudged,
        "interrupts": [nudged],
        "awaiting_user_input": True,
        "phase": WorkflowPhase.INPUT,
        "next_action": "ask_user",
        "audit_log": [
            AuditEvent(
                node_name="handle_off_intent",
                message="Off-intent reply detected; re-emitting prior interrupt with nudge.",
                metadata={
                    "user_intent": state.get("user_intent"),
                    "interrupt_kind": interrupt.kind,
                },
            )
        ],
    }


def _nudge_prompt(state: AgentState, prompt: str) -> str:
    if state.get("locale") == "en-US":
        return (
            "I received your message, but I still need your answer to the question "
            f"above:\n{prompt}"
        )
    return f"我先收到您的消息,但还需要您回答上面的问题:\n{prompt}"
