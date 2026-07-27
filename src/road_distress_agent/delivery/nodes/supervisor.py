"""Delivery supervisor: dispatches the confirmed ledger to specialist agents.

Routing to the three specialists is via graph edges (parallel fan-out); this node
records the dispatch decision and surfaces how many defects each specialist will
process.
"""

from __future__ import annotations

from road_distress_agent.delivery.selection import kept_records
from road_distress_agent.delivery.state import DeliveryState
from road_distress_agent.localization import DEFAULT_LOCALE, normalize_locale
from road_distress_agent.state import AuditEvent


def delivery_supervisor(state: DeliveryState) -> DeliveryState:
    records = kept_records(state)
    locale = normalize_locale(state.get("locale") or DEFAULT_LOCALE)
    return {
        "audit_log": [
            AuditEvent(
                node_name="delivery_supervisor",
                message=_message(locale),
                metadata={"defect_count": len(records)},
            )
        ]
    }


def _message(locale: str) -> str:
    if locale == "en-US":
        return "Dispatched the confirmed ledger to report, cost, and work-order specialists."
    return "按确认后的台账分派报告/造价/工单三个专家并行执行。"
