"""Cost & quantity specialist.

LLM does only the fuzzy step: map each defect to a ``norm_code`` and pull
dimensions out of ``known_features``. All quantities, prices and the
segment-level mobilization aggregation are deterministic (``costing``), so the
numbers are auditable. Emits ``trace_event(kind="llm_call")`` and a retrieval
trace for the norm lookup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from road_distress_agent.delivery.artifacts import deliverable_dir
from road_distress_agent.delivery.cost_norm_lookup import CostNormLookup, make_cost_norm_lookup
from road_distress_agent.delivery.costing import CostInput, Dimensions, build_cost_sheet
from road_distress_agent.delivery.llm import invoke_structured, load_prompt
from road_distress_agent.delivery.selection import kept_records
from road_distress_agent.delivery.state import DeliveryState
from road_distress_agent.delivery.xlsx_writer import write_cost_workbook
from road_distress_agent.localization import DEFAULT_LOCALE, normalize_locale
from road_distress_agent.projects.store import make_project_store
from road_distress_agent.state import AuditEvent
from road_distress_agent.tracing import retrieval_trace_event, trace_event

NODE_NAME = "cost_quantity_agent"


class CostMapping(BaseModel):
    record_id: str
    norm_code: str = ""
    length_m: float | None = None
    area_m2: float | None = None
    depth_mm: float | None = None
    note: str | None = None


class CostMappingBatch(BaseModel):
    mappings: list[CostMapping] = Field(default_factory=list)


def cost_quantity_agent(state: DeliveryState) -> DeliveryState:
    records = kept_records(state)
    locale = _state_locale(state)
    lookup = make_cost_norm_lookup()
    if not records:
        return _empty_result(locale)

    batch = _invoke_llm(_build_messages(records, lookup, locale))
    llm_trace = trace_event(
        node_name=NODE_NAME,
        kind="llm_call",
        title=_message("trace_title", locale),
        inputs={"record_count": len(records), "locale": locale},
        output=batch,
    )
    inputs, skipped = _to_cost_inputs(records, batch, locale)
    sheet = build_cost_sheet(lookup, inputs)
    path = _write_workbook(state, sheet, locale)

    return {
        "cost_result": {
            "file": str(path),
            "total_cny": sheet.total_cny,
            "defect_subtotal_cny": sheet.defect_subtotal_cny,
            "mobilization_subtotal_cny": sheet.mobilization_subtotal_cny,
            "defect_line_count": len(sheet.defect_lines),
            "norm_codes": sorted({i.norm_code for i in inputs}),
            "mobilization": [m.model_dump(mode="json") for m in sheet.mobilization_lines],
            "skipped": skipped,
        },
        "audit_log": [
            llm_trace,
            retrieval_trace_event(
                node_name=NODE_NAME,
                title=_message("retrieval_title", locale),
                query="norm_items/resource_lines/shared_cost_rules",
                filters={"norm_codes": sorted({i.norm_code for i in inputs})},
                chunks=[],
            ),
            AuditEvent(
                node_name=NODE_NAME,
                message=_message("audit", locale),
                metadata={"total_cny": sheet.total_cny, "skipped": len(skipped)},
            ),
        ],
    }


def _to_cost_inputs(
    records: list[dict[str, Any]], batch: CostMappingBatch, locale: str
) -> tuple[list[CostInput], list[dict[str, Any]]]:
    by_id = {m.record_id: m for m in batch.mappings}
    inputs: list[CostInput] = []
    skipped: list[dict[str, Any]] = []
    for record in records:
        record_id = record["record_id"]
        mapping = by_id.get(record_id)
        if mapping is None or not mapping.norm_code:
            reason = mapping.note if mapping else _reason("missing_mapping", locale)
            skipped.append(
                {"record_id": record_id, "reason": reason or _reason("unmatched", locale)}
            )
            continue
        inputs.append(
            CostInput(
                record_id=record_id,
                source_thread_id=record.get("source_thread_id"),
                defect_category=(record.get("payload") or {}).get("defect_category"),
                norm_code=mapping.norm_code,
                dimensions=Dimensions(
                    length_m=mapping.length_m, area_m2=mapping.area_m2, depth_mm=mapping.depth_mm
                ),
            )
        )
    return inputs, skipped


def _build_messages(
    records: list[dict[str, Any]], lookup: CostNormLookup, locale: str
) -> list[Any]:
    catalog = [
        {
            "norm_code": n.norm_code,
            "process_name": n.process_name,
            "applicable_defect": n.applicable_defect,
            "calculation_unit": n.calculation_unit,
        }
        for n in lookup.list_norms()
    ]
    defects = [
        {
            "record_id": r["record_id"],
            "defect_category": (r.get("payload") or {}).get("defect_category"),
            "chosen_method": (r.get("payload") or {}).get("chosen_method"),
            "known_features": (r.get("payload") or {}).get("known_features"),
        }
        for r in records
    ]
    payload = {"target_locale": locale, "norm_catalog": catalog, "defects": defects}
    return [
        SystemMessage(content=load_prompt("cost_mapping.txt")),
        HumanMessage(content=f"cost_mapping_input = {payload}"),
    ]


def _write_workbook(state: DeliveryState, sheet: Any, locale: str) -> Path:
    project_id = state.get("project_id") or "unknown"
    project = make_project_store().get_project(project_id)
    meta = project.model_dump(mode="json") if project else {"name": project_id}
    path = deliverable_dir(project_id) / f"cost_{project_id}.xlsx"
    return write_cost_workbook(sheet, project=meta, path=path, locale=locale)


def _empty_result(locale: str) -> DeliveryState:
    message = "No billable defects." if locale == "en-US" else "无可计价病害。"
    return {
        "cost_result": {"file": None, "total_cny": 0.0, "defect_line_count": 0, "skipped": []},
        "audit_log": [AuditEvent(node_name=NODE_NAME, message=message)],
    }


def _invoke_llm(messages: list[Any]) -> CostMappingBatch:
    return invoke_structured(
        messages,
        CostMappingBatch,
        usage_correlation_name=NODE_NAME,
    )


def _state_locale(state: DeliveryState) -> str:
    return normalize_locale(state.get("locale") or DEFAULT_LOCALE)


def _reason(key: str, locale: str) -> str:
    if locale == "en-US":
        return {"missing_mapping": "No mapping returned", "unmatched": "No norm matched"}[key]
    return {"missing_mapping": "未返回映射", "unmatched": "未匹配定额"}[key]


def _message(key: str, locale: str) -> str:
    messages = _MESSAGES_EN if locale == "en-US" else _MESSAGES_ZH
    return messages[key]


_MESSAGES_ZH = {
    "trace_title": "病害→定额映射与尺寸抽取",
    "retrieval_title": "定额库结构化查询",
    "audit": "生成工程量与造价表。",
}
_MESSAGES_EN = {
    "trace_title": "Distress-to-norm mapping and dimension extraction",
    "retrieval_title": "Structured norm-library lookup",
    "audit": "Generated the quantity and cost workbook.",
}
