"""Trace helpers for HITL interrupt creation."""

from __future__ import annotations

from road_distress_agent.state import AuditEvent, InterruptState
from road_distress_agent.tracing import trace_event


def interrupt_trace_event(interrupt: InterruptState) -> AuditEvent:
    return trace_event(
        node_name=interrupt.node_name,
        kind="interrupt",
        title="Diagnosis HITL interrupt",
        metadata={
            "interrupt_kind": interrupt.kind,
            "interrupt_prompt": interrupt.prompt,
            "required_fields": [*interrupt.required_fields],
            "candidate_ids": [*interrupt.candidate_ids],
        },
    )
