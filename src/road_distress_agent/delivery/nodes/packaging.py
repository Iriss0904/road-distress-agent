"""Delivery packager + project memory writer (terminal nodes)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from road_distress_agent.delivery.selection import kept_records
from road_distress_agent.delivery.state import DeliveryState
from road_distress_agent.localization import DEFAULT_LOCALE, normalize_locale
from road_distress_agent.projects.store import make_project_store
from road_distress_agent.state import AuditEvent, LoadedMemory
from road_distress_agent.tools.memory import make_memory_tool


def delivery_packager(state: DeliveryState) -> DeliveryState:
    """Assemble the downloadable package manifest from specialist outputs."""
    locale = _state_locale(state)
    review = state.get("compliance_review")
    report = state.get("report_result") or {}
    cost = state.get("cost_result") or {}
    work_order = state.get("work_order_result") or {}
    package = {
        "compliance_passed": bool(review and review.passed),
        "need_human_review": bool(review and review.forced_need_human_review),
        "files": _collect_files(report, cost, work_order),
        "report": report,
        "cost": cost,
        "work_order": work_order,
        "compliance_issues": [i.model_dump(mode="json") for i in (review.issues if review else [])],
    }
    return {
        "delivery_package": package,
        "audit_log": [
            AuditEvent(
                node_name="delivery_packager",
                message=_message("package", locale),
                metadata={
                    "compliance_passed": package["compliance_passed"],
                    "file_count": len(package["files"]),
                },
            )
        ],
    }


def project_memory_writer(state: DeliveryState) -> DeliveryState:
    """Close out: persist project-level memory and mark the project delivered."""
    locale = _state_locale(state)
    project_id = state.get("project_id")
    user_id = state.get("user_id")
    if project_id:
        make_project_store().update_project(project_id, status="delivered")
    wrote = _write_project_memory(state, project_id, user_id)
    return {
        "audit_log": [
            AuditEvent(
                node_name="project_memory_writer",
                message=_message("memory", locale),
                metadata={"project_id": project_id, "wrote_memory": wrote},
            )
        ]
    }


def _collect_files(*results: dict[str, Any]) -> list[str]:
    files: list[str] = []
    for result in results:
        for key in ("file",):
            if result.get(key):
                files.append(result[key])
        for channel in ("email", "calendar"):
            artifact = (result.get(channel) or {}).get("artifact")
            if artifact:
                files.append(artifact)
    return files


def _write_project_memory(
    state: DeliveryState, project_id: str | None, user_id: str | None
) -> bool:
    if not (project_id and user_id):
        return False
    project = make_project_store().get_project(project_id)
    if project is None:
        return False
    patch = _memory_patch(project, kept_records(state), state.get("cost_result") or {})
    if not patch:
        return False

    tool = make_memory_tool()
    existing = state.get("loaded_memory") or tool.load(user_id)
    tool.save(user_id, _merge(existing, patch))
    return True


def _memory_patch(
    project: Any, records: list[dict[str, Any]], cost: dict[str, Any]
) -> dict[str, Any]:
    regional = {f"项目_{project.name}_路段": project.segment} if project.segment else {}
    resources = {f"项目_{project.name}_班组": project.crew} if project.crew else {}
    summary = _case_summary(project, records, cost) if records else None
    if not (regional or resources or summary):
        return {}
    return {"regional": regional, "resources": resources, "summary": summary}


def _case_summary(project: Any, records: list[dict[str, Any]], cost: dict[str, Any]) -> str:
    methods = sorted(
        {method for r in records if (method := (r.get("payload") or {}).get("chosen_method"))}
    )
    total = cost.get("total_cny")
    parts = [f"项目「{project.name}」交付 {len(records)} 处病害"]
    if methods:
        parts.append("处治方法：" + "、".join(m for m in methods if m))
    if total is not None:
        parts.append(f"估算造价 {total} 元")
    return "；".join(parts) + "。"


def _merge(existing: LoadedMemory, patch: dict[str, Any]) -> LoadedMemory:
    now = datetime.now(timezone.utc).isoformat()
    regional = {**existing.regional_context, **_records(patch["regional"], now)}
    resources = {**existing.resource_constraints, **_records(patch["resources"], now)}
    summaries = list(existing.case_summaries)
    if patch.get("summary"):
        summaries.append({"value": patch["summary"], "updated_at": now, "source": "delivery"})
    return LoadedMemory(
        user_preferences=existing.user_preferences,
        regional_context=regional,
        resource_constraints=resources,
        case_summaries=summaries,
    )


def _records(values: dict[str, str], now: str) -> dict[str, dict[str, str]]:
    return {
        k: {"value": v, "updated_at": now, "source": "delivery"} for k, v in values.items() if v
    }


def _state_locale(state: DeliveryState) -> str:
    return normalize_locale(state.get("locale") or DEFAULT_LOCALE)


def _message(key: str, locale: str) -> str:
    messages = _MESSAGES_EN if locale == "en-US" else _MESSAGES_ZH
    return messages[key]


_MESSAGES_ZH = {
    "package": "打包交付物。",
    "memory": "写入项目级记忆并标记已交付。",
}
_MESSAGES_EN = {
    "package": "Packaged delivery artifacts.",
    "memory": "Wrote project memory and marked the project delivered.",
}
