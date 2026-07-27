"""Pure projection of a chat thread lifecycle status.

Status is derived from authoritative sources, never stored on chat_threads.
Precedence: delivered > promoted > ready > draft.
"""

from __future__ import annotations

from typing import Any

STATUS_DRAFT = "draft"
STATUS_READY = "ready"
STATUS_PROMOTED = "promoted"
STATUS_DELIVERED = "delivered"


def derive_thread_status(
    *, snapshot: dict[str, Any] | None, has_active_record: bool, is_delivered: bool
) -> str:
    if is_delivered:
        return STATUS_DELIVERED
    if has_active_record:
        return STATUS_PROMOTED
    if _is_ready(snapshot):
        return STATUS_READY
    return STATUS_DRAFT


def _is_ready(snapshot: dict[str, Any] | None) -> bool:
    snap = snapshot or {}
    return bool(snap.get("chosen_method") or snap.get("final_answer_message"))
