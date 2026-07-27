"""Inspection-report specialist: LLM narrative + deterministic docx assembly.

Runs after the cost agent so the report can cite the segment estimate. The LLM
only writes prose (overall summary + per-defect recommendation); all facts,
citations and numbers come from the ledger and the cost result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from road_distress_agent.delivery.artifacts import deliverable_dir
from road_distress_agent.delivery.docx_writer import write_report_document
from road_distress_agent.delivery.llm import invoke_structured, load_prompt
from road_distress_agent.delivery.report_formatting import format_feature_sentence
from road_distress_agent.delivery.selection import kept_records
from road_distress_agent.delivery.state import DeliveryState
from road_distress_agent.localization import DEFAULT_LOCALE, normalize_locale
from road_distress_agent.projects.store import make_project_store
from road_distress_agent.state import AuditEvent
from road_distress_agent.tracing import trace_event

NODE_NAME = "report_agent"


class DefectNote(BaseModel):
    record_id: str
    recommendation: str = ""


class ReportNarrative(BaseModel):
    overall_summary: str = ""
    defect_notes: list[DefectNote] = Field(default_factory=list)


def report_agent(state: DeliveryState) -> DeliveryState:
    records = kept_records(state)
    locale = _state_locale(state)
    cost_summary = state.get("cost_result")
    narrative = _invoke_llm(_build_messages(records, cost_summary, locale))
    llm_trace = trace_event(
        node_name=NODE_NAME,
        kind="llm_call",
        title=_message("trace_title", locale),
        inputs={"record_count": len(records), "locale": locale},
        output=narrative,
    )
    notes = {n.record_id: n.recommendation for n in narrative.defect_notes}
    locations = (state.get("archive_decision") or {}).get("locations") or {}
    sections = [
        _defect_section(r, notes.get(r["record_id"], ""), locations, locale) for r in records
    ]
    path = _write_document(state, narrative.overall_summary, sections, cost_summary, locale)

    return {
        "report_result": {"file": str(path), "defect_count": len(sections)},
        "audit_log": [
            llm_trace,
            AuditEvent(
                node_name=NODE_NAME,
                message=_message("audit", locale),
                metadata={"defect_count": len(sections)},
            ),
        ],
    }


def _defect_section(
    record: dict[str, Any],
    recommendation: str,
    locations: dict[str, str],
    locale: str,
) -> dict[str, Any]:
    payload = record.get("payload") or {}
    features = payload.get("known_features") or {}
    citations = payload.get("citations") or []
    clause_ids = [c.get("clause_id") or c.get("chunk_id") for c in citations if c]
    record_id = record["record_id"]
    return {
        "record_id": record_id,
        "defect_category": payload.get("defect_category"),
        "location": _required_location(record_id, locations),
        "features": format_feature_sentence(features, locale=locale),
        "chosen_method": payload.get("chosen_method"),
        "citations": "、".join(str(c) for c in clause_ids if c) or None,
        "summary": payload.get("summary"),
        "recommendation": recommendation,
    }


def _build_messages(
    records: list[dict[str, Any]],
    cost_summary: dict[str, Any] | None,
    locale: str,
) -> list[Any]:
    defects = [
        {
            "record_id": r["record_id"],
            "defect_category": (r.get("payload") or {}).get("defect_category"),
            "known_features": (r.get("payload") or {}).get("known_features"),
            "chosen_method": (r.get("payload") or {}).get("chosen_method"),
            "citations": [
                c.get("clause_id") for c in ((r.get("payload") or {}).get("citations") or []) if c
            ],
        }
        for r in records
    ]
    payload = {
        "target_locale": locale,
        "defects": defects,
        "cost_summary": {
            "total_cny": (cost_summary or {}).get("total_cny"),
            "mobilization_subtotal_cny": (cost_summary or {}).get("mobilization_subtotal_cny"),
        },
    }
    return [
        SystemMessage(content=load_prompt("report_narrative.txt")),
        HumanMessage(content=f"report_input = {payload}"),
    ]


def _write_document(
    state: DeliveryState,
    overall_summary: str,
    sections: list[dict[str, Any]],
    cost_summary: dict[str, Any] | None,
    locale: str,
) -> Path:
    project_id = state.get("project_id") or "unknown"
    project = make_project_store().get_project(project_id)
    meta = project.model_dump(mode="json") if project else {"name": project_id}
    path = deliverable_dir(project_id) / f"report_{project_id}.docx"
    return write_report_document(
        project=meta,
        report_meta=_report_meta(meta, state.get("archive_decision") or {}),
        overall_summary=overall_summary,
        defect_sections=sections,
        cost_summary=cost_summary,
        path=path,
        locale=locale,
    )


def _invoke_llm(messages: list[Any]) -> ReportNarrative:
    return invoke_structured(
        messages,
        ReportNarrative,
        usage_correlation_name=NODE_NAME,
    )


def _state_locale(state: DeliveryState) -> str:
    return normalize_locale(state.get("locale") or DEFAULT_LOCALE)


def _message(key: str, locale: str) -> str:
    messages = _MESSAGES_EN if locale == "en-US" else _MESSAGES_ZH
    return messages[key]


_MESSAGES_ZH = {
    "trace_title": "巡查报告叙述生成",
    "audit": "生成巡查报告 docx。",
}
_MESSAGES_EN = {
    "trace_title": "Inspection report narrative generation",
    "audit": "Generated the inspection report DOCX.",
}


def _required_location(record_id: str, locations: dict[str, str]) -> str:
    location = str(locations.get(record_id) or "").strip()
    if not location:
        raise ValueError(f"report record {record_id} missing confirmed location.")
    return location


def _report_meta(project: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "name",
        "segment",
        "inspector",
        "crew",
        "unit_name",
        "inspection_time",
        "inspection_method",
        "participants",
    )
    return {field: decision.get(field) or project.get(field) for field in fields}
