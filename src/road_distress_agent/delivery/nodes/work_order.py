"""Work-order specialist: LLM drafts the order, MCP (dry-run) lands the actions.

The LLM composes the order text from the confirmed ledger; method names must
match each defect's confirmed ``chosen_method``. Email/calendar actions go
through the dry-run MCP adapters, which produce inspectable .eml/.ics artifacts
and report ``sent=False`` / ``inserted=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from road_distress_agent.delivery.artifacts import deliverable_dir
from road_distress_agent.delivery.construction_plan_writer import write_construction_plan_document
from road_distress_agent.delivery.llm import invoke_structured, load_prompt
from road_distress_agent.delivery.selection import kept_records
from road_distress_agent.delivery.state import DeliveryState
from road_distress_agent.localization import DEFAULT_LOCALE, normalize_locale
from road_distress_agent.projects.store import make_project_store
from road_distress_agent.state import AuditEvent
from road_distress_agent.tools.mcp_calendar import CalendarEvent, make_calendar_client
from road_distress_agent.tools.mcp_email import EmailDraft, make_email_client
from road_distress_agent.tracing import trace_event

_DEFAULT_RECIPIENT = "责任班组"
_DEFAULT_RECIPIENT_EN = "Responsible crew"
_MIN_DURATION_DAYS = 1
NODE_NAME = "work_order_agent"


class WorkOrderDraft(BaseModel):
    subject: str = ""
    body: str = ""
    materials: list[str] = Field(default_factory=list)
    acceptance: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class WorkOrderContext:
    state: DeliveryState
    project: dict[str, Any]
    records: list[dict[str, Any]]
    draft: WorkOrderDraft
    window: dict[str, str]
    locale: str


def work_order_agent(state: DeliveryState) -> DeliveryState:
    records = kept_records(state)
    locale = _state_locale(state)
    project = _project_meta(state)
    draft = _invoke_llm(_build_messages(records, project, locale))
    llm_trace = trace_event(
        node_name=NODE_NAME,
        kind="llm_call",
        title=_message("trace_title", locale),
        inputs={"record_count": len(records), "locale": locale},
        output=draft,
    )
    window = _schedule_window(state)
    context = WorkOrderContext(
        state=state,
        project=project,
        records=records,
        draft=draft,
        window=window,
        locale=locale,
    )
    plan_path = _write_plan_document(context)

    email_receipt = _dispatch_email(context)
    calendar_receipt = _dispatch_calendar(context)

    return {
        "work_order_result": {
            "subject": draft.subject,
            "file": str(plan_path),
            "materials": draft.materials,
            "acceptance": draft.acceptance,
            "schedule": window,
            "email": _public_receipt(email_receipt),
            "calendar": _public_receipt(calendar_receipt),
        },
        "audit_log": [
            llm_trace,
            AuditEvent(
                node_name=NODE_NAME,
                message=_message("audit", locale),
                metadata={
                    "email_sent": email_receipt["sent"],
                    "calendar_inserted": calendar_receipt["inserted"],
                },
            ),
        ],
    }


def _write_plan_document(context: WorkOrderContext) -> Path:
    project_id = context.state.get("project_id") or "unknown"
    path = deliverable_dir(project_id) / f"maintenance_plan_{project_id}.docx"
    return write_construction_plan_document(
        project=context.project,
        plan={
            "subject": context.draft.subject,
            "body": context.draft.body,
            "materials": context.draft.materials,
            "acceptance": context.draft.acceptance,
            "schedule": context.window,
            "defects": _plan_defects(context.records, context.state),
        },
        path=path,
        locale=context.locale,
    )


def _plan_defects(records: list[dict[str, Any]], state: DeliveryState) -> list[dict[str, Any]]:
    locations = (state.get("archive_decision") or {}).get("locations") or {}
    return [_plan_defect(record, locations) for record in records]


def _plan_defect(record: dict[str, Any], locations: dict[str, str]) -> dict[str, Any]:
    payload = record.get("payload") or {}
    record_id = record["record_id"]
    return {
        "record_id": record_id,
        "defect_category": payload.get("defect_category"),
        "chosen_method": payload.get("chosen_method"),
        "location": locations.get(record_id),
    }


def _dispatch_email(context: WorkOrderContext) -> dict[str, Any]:
    body = _compose_email_body(context.draft, context.locale)
    recipient = context.project.get("crew") or _default_recipient(context.locale)
    receipt = make_email_client().create_draft(
        EmailDraft(to=recipient, subject=context.draft.subject, body=body)
    )
    path = _artifact_path(context.state, "work_order", "eml")
    path.write_text(receipt.pop("raw"), encoding="utf-8")
    receipt["artifact"] = str(path)
    return receipt


def _dispatch_calendar(context: WorkOrderContext) -> dict[str, Any]:
    receipt = make_calendar_client().preview_event(
        CalendarEvent(
            title=context.draft.subject or _calendar_title(context.locale),
            start_date=context.window["start_date"],
            end_date=context.window["end_date"],
            location=context.project.get("segment"),
            description=context.draft.body,
        )
    )
    path = _artifact_path(context.state, "work_order", "ics")
    path.write_text(receipt.pop("ics"), encoding="utf-8")
    receipt["artifact"] = str(path)
    return receipt


def _schedule_window(state: DeliveryState) -> dict[str, str]:
    durations = (state.get("archive_decision") or {}).get("durations") or {}
    total = sum(_as_int(v) for v in durations.values())
    total = max(total, _MIN_DURATION_DAYS)
    start = date.today()
    end = start + timedelta(days=total)
    return {"start_date": start.isoformat(), "end_date": end.isoformat(), "total_days": str(total)}


def _build_messages(
    records: list[dict[str, Any]], project: dict[str, Any], locale: str
) -> list[Any]:
    defects = [
        {
            "defect_category": (r.get("payload") or {}).get("defect_category"),
            "chosen_method": (r.get("payload") or {}).get("chosen_method"),
            "materials": ((r.get("payload") or {}).get("final_answer") or {}).get("materials"),
            "acceptance": ((r.get("payload") or {}).get("final_answer") or {}).get(
                "acceptance_criteria"
            ),
        }
        for r in records
    ]
    payload = {
        "target_locale": locale,
        "segment": project.get("segment"),
        "crew": project.get("crew"),
        "defects": defects,
    }
    return [
        SystemMessage(content=load_prompt("work_order.txt")),
        HumanMessage(content=f"work_order_input = {payload}"),
    ]


def _compose_email_body(draft: WorkOrderDraft, locale: str) -> str:
    parts = [draft.body, ""]
    material_label = "Materials: " if locale == "en-US" else "材料清单："
    acceptance_label = "Acceptance Criteria: " if locale == "en-US" else "验收标准："
    parts.append(material_label + _joined(draft.materials, locale))
    parts.append(acceptance_label + _joined(draft.acceptance, locale))
    return "\n".join(parts)


def _project_meta(state: DeliveryState) -> dict[str, Any]:
    project_id = state.get("project_id") or "unknown"
    project = make_project_store().get_project(project_id)
    return project.model_dump(mode="json") if project else {"name": project_id}


def _artifact_path(state: DeliveryState, stem: str, suffix: str) -> Path:
    project_id = state.get("project_id") or "unknown"
    return deliverable_dir(project_id) / f"{stem}_{project_id}.{suffix}"


def _public_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in receipt.items() if k not in ("raw", "ics")}


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _invoke_llm(messages: list[Any]) -> WorkOrderDraft:
    return invoke_structured(
        messages,
        WorkOrderDraft,
        usage_correlation_name=NODE_NAME,
    )


def _state_locale(state: DeliveryState) -> str:
    return normalize_locale(state.get("locale") or DEFAULT_LOCALE)


def _default_recipient(locale: str) -> str:
    return _DEFAULT_RECIPIENT_EN if locale == "en-US" else _DEFAULT_RECIPIENT


def _calendar_title(locale: str) -> str:
    return "Road Maintenance Work Window" if locale == "en-US" else "道路养护施工窗口"


def _joined(items: list[str], locale: str) -> str:
    if items:
        separator = ", " if locale == "en-US" else "、"
        return separator.join(items)
    return "Not specified." if locale == "en-US" else "未列明。"


def _message(key: str, locale: str) -> str:
    messages = _MESSAGES_EN if locale == "en-US" else _MESSAGES_ZH
    return messages[key]


_MESSAGES_ZH = {
    "trace_title": "施工工单生成",
    "audit": "工单已生成，邮件/日历按 dry-run 落地。",
}
_MESSAGES_EN = {
    "trace_title": "Construction work-order generation",
    "audit": "Generated the work order; email and calendar artifacts were written in dry-run mode.",
}
