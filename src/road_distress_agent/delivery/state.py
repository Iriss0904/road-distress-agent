"""State contract for the project-level delivery subgraph.

Deliberately separate from the diagnosis ``AgentState`` so the承重墙 stays clean.
Defect records travel as plain dicts (``DefectRecord.model_dump``) to keep the
checkpoint serde allowlist untouched; the structured guardrail/audit models are
reused from the diagnosis state.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from road_distress_agent.localization import Locale
from road_distress_agent.state import (
    AuditEvent,
    ErrorEvent,
    InterruptState,
    SafetyReview,
    append_list,
)


class DeliveryState(TypedDict, total=False):
    # A. control
    project_id: str
    user_id: str
    locale: Locale
    messages: Annotated[list[AnyMessage], add_messages]
    interrupt: InterruptState | None
    awaiting_user_input: bool

    # B. input ledger (DefectRecord dumps) + archive decision
    active_records: list[dict[str, Any]]
    dedup_preview: dict[str, Any] | None
    archive_decision: dict[str, Any] | None  # user-confirmed: kept ids, metadata, durations

    # C. specialist outputs (each writes its own key — parallel-safe)
    report_result: dict[str, Any] | None
    cost_result: dict[str, Any] | None
    work_order_result: dict[str, Any] | None

    # D. gate + delivery
    compliance_review: SafetyReview | None
    delivery_package: dict[str, Any] | None

    # audit
    audit_log: Annotated[list[AuditEvent], append_list]
    errors: Annotated[list[ErrorEvent], append_list]
