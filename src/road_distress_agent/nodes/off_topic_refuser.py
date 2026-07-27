"""Fixed off-topic boundary response that preserves active diagnosis state."""

from __future__ import annotations

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.evidence_observation import (
    evidence_observation_event,
    observe_off_topic,
)
from road_distress_agent.state import AgentState, AuditEvent
from road_distress_agent.tracing import trace_event

OFF_TOPIC_MESSAGE = (
    "我只能协助道路病害识别、养护处治方法、施工步骤、验收标准和施工组织安全等相关问题。"
)
OFF_TOPIC_MESSAGE_EN = (
    "I can help only with road-distress identification, maintenance treatment methods, "
    "construction steps, acceptance criteria, work-zone safety, and related standards."
)


def off_topic_refuser(state: AgentState) -> AgentState:
    interrupt = state.get("interrupt")
    awaiting = interrupt is not None
    observation = observe_off_topic()
    return {
        "direct_message": (
            OFF_TOPIC_MESSAGE_EN if state.get("locale") == "en-US" else OFF_TOPIC_MESSAGE
        ),
        "refusal_type": "off_topic",
        "awaiting_user_input": awaiting,
        "evidence_assessment": observation.assessment,
        "phase": WorkflowPhase.INPUT,
        "next_action": "return_off_topic_boundary",
        "audit_log": [
            trace_event(
                node_name="off_topic_refuser",
                kind="stage",
                title="Off-topic boundary response",
                inputs={"latest_user_text": state.get("latest_user_text")},
                output={"preserved_interrupt": awaiting},
            ),
            AuditEvent(
                node_name="off_topic_refuser",
                message="Returned fixed off-topic boundary response.",
                metadata={"preserved_interrupt": awaiting},
            ),
            evidence_observation_event(
                node_name="off_topic_refuser",
                observation=observation,
            ),
        ],
    }
