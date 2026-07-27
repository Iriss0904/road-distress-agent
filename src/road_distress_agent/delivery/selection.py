"""Shared helper: the confirmed-for-delivery defect records."""

from __future__ import annotations

from typing import Any

from road_distress_agent.delivery.state import DeliveryState


def kept_records(state: DeliveryState) -> list[dict[str, Any]]:
    """Return the active records the user confirmed to include in the delivery.

    Falls back to all active records when no archive decision is present (e.g. a
    direct/test invocation that skips the confirmation gate).
    """
    records = state.get("active_records") or []
    decision = state.get("archive_decision") or {}
    kept_ids = decision.get("kept_record_ids")
    if kept_ids is None:
        return list(records)
    kept = set(kept_ids)
    return [r for r in records if r.get("record_id") in kept]
