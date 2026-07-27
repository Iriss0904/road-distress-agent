"""Archive intake + ledger loading for the delivery subgraph."""

from __future__ import annotations

from road_distress_agent.delivery.state import DeliveryState
from road_distress_agent.localization import DEFAULT_LOCALE, normalize_locale
from road_distress_agent.projects.store import ProjectStore, make_project_store
from road_distress_agent.state import AuditEvent


def archive_intake(state: DeliveryState) -> DeliveryState:
    """Lock the project and mark it as archiving."""
    project_id = _require_project_id(state)
    store = make_project_store()
    project = store.update_project(project_id, status="archiving")
    return {
        "locale": project.locale,
        "audit_log": [
            AuditEvent(
                node_name="archive_intake",
                message=_message("archive", project.locale),
                metadata={"project_id": project_id, "locale": project.locale},
            )
        ],
    }


def ledger_loader(state: DeliveryState) -> DeliveryState:
    """Load all active defect records for the project."""
    project_id = _require_project_id(state)
    locale = normalize_locale(state.get("locale") or DEFAULT_LOCALE)
    store: ProjectStore = make_project_store()
    records = store.list_records(project_id, status="active")
    dumped = [r.model_dump(mode="json") for r in records]
    return {
        "active_records": dumped,
        "audit_log": [
            AuditEvent(
                node_name="ledger_loader",
                message=_message("ledger", locale),
                metadata={"project_id": project_id, "active_count": len(dumped)},
            )
        ],
    }


def _require_project_id(state: DeliveryState) -> str:
    project_id = state.get("project_id")
    if not project_id:
        raise ValueError("delivery subgraph requires project_id in state.")
    return project_id


def _message(key: str, locale: str) -> str:
    messages = _MESSAGES_EN if locale == "en-US" else _MESSAGES_ZH
    return messages[key]


_MESSAGES_ZH = {
    "archive": "锁定巡查任务并进入归档。",
    "ledger": "读取台账 active 记录。",
}
_MESSAGES_EN = {
    "archive": "Locked the inspection task and started archiving.",
    "ledger": "Loaded active ledger records.",
}
