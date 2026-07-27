"""Dedup preview + archive confirmation gate (HITL).

``dedup_resolver`` uses a real LLM to flag suspected duplicates / incomplete
records; the user makes the final call at the confirmation gate.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from road_distress_agent.delivery.llm import invoke_structured, load_prompt
from road_distress_agent.delivery.state import DeliveryState
from road_distress_agent.localization import DEFAULT_LOCALE, normalize_locale
from road_distress_agent.projects.store import make_project_store
from road_distress_agent.state import AuditEvent, InterruptState
from road_distress_agent.tracing import trace_event

ARCHIVE_CONFIRMATION = "archive_confirmation"
NODE_NAME = "dedup_resolver"
_PROJECT_META_FIELDS = (
    "segment",
    "inspector",
    "crew",
    "unit_name",
    "inspection_time",
    "inspection_method",
    "participants",
)
_STORED_PROJECT_FIELDS = ("segment", "inspector", "crew")


class DuplicateGroup(BaseModel):
    keep: str
    duplicates: list[str] = Field(default_factory=list)
    reason: str | None = None


class DedupAssessment(BaseModel):
    duplicate_groups: list[DuplicateGroup] = Field(default_factory=list)
    incomplete: list[str] = Field(default_factory=list)


def dedup_resolver(state: DeliveryState) -> DeliveryState:
    """Flag suspected duplicates and pause for user confirmation of the ledger."""
    records = state.get("active_records") or []
    locale = _state_locale(state)
    assessment = _invoke_llm(_build_messages(records, locale))
    llm_trace = trace_event(
        node_name=NODE_NAME,
        kind="llm_call",
        title=_message("trace_title", locale),
        inputs={"record_count": len(records), "locale": locale},
        output=assessment,
    )
    preview = _assemble_preview(records, assessment)
    interrupt = InterruptState(
        interrupt_id=f"archive-{state.get('project_id', 'unknown')}",
        node_name=NODE_NAME,
        prompt=_confirmation_prompt(locale),
        kind=ARCHIVE_CONFIRMATION,
        candidate_ids=preview["included"],
    )
    return {
        "dedup_preview": preview,
        "interrupt": interrupt,
        "awaiting_user_input": True,
        "audit_log": [
            llm_trace,
            AuditEvent(
                node_name=NODE_NAME,
                message=_message("preview_audit", locale),
                metadata={
                    "included": len(preview["included"]),
                    "suspected_duplicate_groups": len(preview["suspected_duplicates"]),
                },
            ),
        ],
    }


def _assemble_preview(records: list[dict[str, Any]], assessment: DedupAssessment) -> dict[str, Any]:
    incomplete = list(assessment.incomplete)
    included = [r["record_id"] for r in records if r["record_id"] not in incomplete]
    suspected = [g.model_dump(mode="json") for g in assessment.duplicate_groups]
    return {"included": included, "suspected_duplicates": suspected, "incomplete": incomplete}


def _build_messages(records: list[dict[str, Any]], locale: str) -> list[Any]:
    summary = [
        {
            "record_id": r["record_id"],
            "defect_category": (r.get("payload") or {}).get("defect_category"),
            "chosen_method": (r.get("payload") or {}).get("chosen_method"),
            "known_features": (r.get("payload") or {}).get("known_features"),
        }
        for r in records
    ]
    payload = {"target_locale": locale, "records": summary}
    return [
        SystemMessage(content=load_prompt("dedup_resolver.txt")),
        HumanMessage(content=f"dedup_input = {payload}"),
    ]


def _invoke_llm(messages: list[Any]) -> DedupAssessment:
    return invoke_structured(
        messages,
        DedupAssessment,
        usage_correlation_name=NODE_NAME,
    )


def dedup_confirm_gate(state: DeliveryState) -> DeliveryState:
    """Apply the user's archive decision: supersede rejects, persist task metadata."""
    preview = state.get("dedup_preview") or {"included": []}
    locale = _state_locale(state)
    decision = _normalize_decision(state.get("archive_decision"), preview, locale)
    store = make_project_store()

    for record_id in decision["superseded_record_ids"]:
        store.set_record_status(record_id, "superseded")
    _persist_project_metadata(store, state.get("project_id"), decision)

    return {
        "archive_decision": decision,
        "interrupt": None,
        "awaiting_user_input": False,
        "audit_log": [
            AuditEvent(
                node_name="dedup_confirm_gate",
                message=_message("confirm_audit", locale),
                metadata={
                    "kept": len(decision["kept_record_ids"]),
                    "superseded": len(decision["superseded_record_ids"]),
                },
            )
        ],
    }


def _normalize_decision(
    decision: dict[str, Any] | None, preview: dict[str, Any], locale: str
) -> dict[str, Any]:
    included = list(preview.get("included") or [])
    if not decision:
        _require_locations(included, {}, locale)
        return {
            "kept_record_ids": included,
            "superseded_record_ids": [],
            "durations": {},
            "locations": {},
            **{field: None for field in _PROJECT_META_FIELDS},
        }
    kept = decision.get("kept_record_ids")
    kept = list(kept) if kept is not None else included
    superseded = [rid for rid in included if rid not in kept]
    explicit = decision.get("superseded_record_ids") or []
    superseded += [rid for rid in explicit if rid not in superseded]
    locations = _normalize_text_map(decision.get("locations"))
    _require_locations(kept, locations, locale)
    return {
        "kept_record_ids": kept,
        "superseded_record_ids": superseded,
        "durations": decision.get("durations") or {},
        "locations": locations,
        **{field: decision.get(field) for field in _PROJECT_META_FIELDS},
    }


def _persist_project_metadata(store, project_id: str | None, decision: dict[str, Any]) -> None:
    if not project_id:
        return
    updates = {field: decision[field] for field in _STORED_PROJECT_FIELDS if decision.get(field)}
    if updates:
        store.update_project(project_id, **updates)


def _normalize_text_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item).strip() for key, item in value.items() if str(item).strip()}


def _require_locations(kept_ids: list[str], locations: dict[str, str], locale: str) -> None:
    missing = [record_id for record_id in kept_ids if not locations.get(record_id)]
    if missing:
        raise ValueError(_missing_location_message(locale))


def _state_locale(state: DeliveryState) -> str:
    return normalize_locale(state.get("locale") or DEFAULT_LOCALE)


def _confirmation_prompt(locale: str) -> str:
    if locale == "en-US":
        return (
            "Confirm the archive list: select kept distresses, mark duplicates, and "
            "complete task details and durations."
        )
    return "请确认归档清单：勾选纳入的病害、标记疑似重复、补全任务信息与工期。"


def _missing_location_message(locale: str) -> str:
    if locale == "en-US":
        return "Each kept distress must have a confirmed location before archiving."
    return "归档前必须补全每处保留病害的位置。"


def _message(key: str, locale: str) -> str:
    messages = _MESSAGES_EN if locale == "en-US" else _MESSAGES_ZH
    return messages[key]


_MESSAGES_ZH = {
    "trace_title": "归档去重判定",
    "preview_audit": "生成归档预览，等待用户确认。",
    "confirm_audit": "归档清单已确认。",
}
_MESSAGES_EN = {
    "trace_title": "Archive deduplication assessment",
    "preview_audit": "Generated the archive preview and paused for confirmation.",
    "confirm_audit": "Archive list confirmed.",
}
