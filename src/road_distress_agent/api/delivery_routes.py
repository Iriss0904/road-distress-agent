"""HTTP + SSE routes for project delivery runs and delivery archives."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from road_distress_agent.api.serialization import stage_label
from road_distress_agent.checkpointing import sqlite_checkpointer
from road_distress_agent.delivery.artifacts import deliverable_dir
from road_distress_agent.delivery.capture import capture_finished_package
from road_distress_agent.delivery.graph import build_delivery_graph
from road_distress_agent.delivery.store import make_delivery_store
from road_distress_agent.delivery.store_models import DeliveryVersion
from road_distress_agent.error_classifiers import classify_db_error, classify_delivery_error
from road_distress_agent.errors import BoundaryError, error_info_payload
from road_distress_agent.localization import DEFAULT_LOCALE, normalize_locale
from road_distress_agent.projects.store import make_project_store
from road_distress_agent.tracing import extract_trace_events, to_plain

router = APIRouter(tags=["delivery"])
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

_SNAPSHOT_FIELDS = (
    "locale",
    "dedup_preview",
    "archive_decision",
    "delivery_package",
    "cost_result",
)


def _delivery_db() -> str:
    return os.environ.get("ROAD_DISTRESS_DELIVERY_DB", ".delivery-workflow.db")


class ConfirmRequest(BaseModel):
    archive_decision: dict[str, Any] = Field(default_factory=dict)


@router.post("/api/projects/{project_id}/archive")
def start_archive(project_id: str) -> StreamingResponse:
    _require_project(project_id)
    return StreamingResponse(
        _stream(project_id, resume_input={"project_id": project_id, "user_id": _user(project_id)}),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/api/projects/{project_id}/archive/confirm")
def confirm_archive(project_id: str, req: ConfirmRequest) -> StreamingResponse:
    _require_project(project_id)
    return StreamingResponse(
        _stream(project_id, archive_decision=req.archive_decision),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get("/api/projects/{project_id}/deliverables")
def list_deliverables(project_id: str) -> dict[str, Any]:
    _require_project(project_id)
    base = deliverable_dir(project_id)
    files = [{"name": p.name, "path": str(p)} for p in sorted(base.glob("*")) if p.is_file()]
    return {"project_id": project_id, "files": files}


@router.get("/api/projects/{project_id}/deliverables/download")
def download_deliverable(project_id: str, name: str) -> FileResponse:
    base = deliverable_dir(project_id).resolve()
    target = (base / name).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="未找到该交付文件。")
    return FileResponse(target, filename=target.name)


@router.get("/api/deliveries")
def list_delivery_archive(user_id: str) -> list[dict[str, Any]]:
    store = make_delivery_store()
    deliveries = store.list_deliveries(None, user_id=user_id)
    return [_delivery_summary(store, delivery) for delivery in deliveries]


@router.get("/api/deliveries/{delivery_id}")
def get_delivery_archive(delivery_id: str) -> dict[str, Any]:
    store = make_delivery_store()
    delivery = store.get_delivery(delivery_id)
    if delivery is None:
        raise HTTPException(status_code=404, detail="未找到该交付归档。")
    versions = store.get_versions(delivery_id)
    latest = versions[-1] if versions else None
    return {
        "delivery": delivery.model_dump(mode="json"),
        "versions": [v.model_dump(mode="json") for v in versions],
        "provenance": _provenance(delivery.project_id, latest),
    }


@router.get("/api/deliveries/{delivery_id}/versions/{version_no}/download")
def download_delivery_version(delivery_id: str, version_no: int) -> FileResponse:
    store = make_delivery_store()
    delivery = store.get_delivery(delivery_id)
    if delivery is None:
        raise HTTPException(status_code=404, detail="未找到该交付归档。")
    version = _version_by_no(store.get_versions(delivery_id), version_no)
    target = _sandboxed_version_path(delivery.project_id, version)
    return FileResponse(target, filename=target.name)


@router.post("/api/deliveries/{delivery_id}/regenerate")
def regenerate_delivery(delivery_id: str) -> StreamingResponse:
    delivery = make_delivery_store().get_delivery(delivery_id)
    if delivery is None:
        raise HTTPException(status_code=404, detail="未找到该交付归档。")
    return start_archive(delivery.project_id)


def _stream(
    project_id: str,
    *,
    resume_input: dict[str, Any] | None = None,
    archive_decision: dict[str, Any] | None = None,
) -> Iterator[str]:
    config = {"configurable": {"thread_id": f"delivery-{project_id}"}}
    locale = _project_locale(project_id)
    try:
        with sqlite_checkpointer(_delivery_db()) as checkpointer:
            graph = build_delivery_graph(checkpointer=checkpointer)
            if archive_decision is not None:
                graph.update_state(config, {"archive_decision": archive_decision})
                stream_input = None
            else:
                stream_input = resume_input

            yield _sse("archive_start", {"project_id": project_id})
            for chunk in graph.stream(stream_input, config=config, stream_mode="updates"):
                yield from _emit_chunk(chunk, locale)
            values = graph.get_state(config).values
            yield _sse("snapshot", _snapshot(values))
            capture_finished_package(project_id, values)
    except Exception as exc:  # surfaced to the UI instead of a dead stream
        yield _sse("error", _error_payload(exc))


def _emit_chunk(chunk: dict[str, Any], locale: str) -> Iterator[str]:
    for node_name, node_delta in chunk.items():
        if node_name.startswith("__"):
            continue
        label_locale = _locale_from_delta(node_delta, locale)
        yield _sse("stage", {"node": node_name, "label": stage_label(node_name, label_locale)})
        if isinstance(node_delta, dict):
            for trace in extract_trace_events(node_delta.get("audit_log") or []):
                yield _sse("trace", trace)


def _locale_from_delta(node_delta: Any, default_locale: str) -> str:
    if isinstance(node_delta, dict) and node_delta.get("locale"):
        return normalize_locale(node_delta.get("locale"))
    return default_locale


def _snapshot(values: dict[str, Any]) -> dict[str, Any]:
    interrupt = values.get("interrupt")
    return {
        "interrupt": to_plain(interrupt) if interrupt else None,
        **{field: to_plain(values.get(field)) for field in _SNAPSHOT_FIELDS},
    }


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _delivery_summary(store: Any, delivery: Any) -> dict[str, Any]:
    versions = store.get_versions(delivery.delivery_id)
    latest = versions[-1].version_no if versions else None
    return {
        "delivery_id": delivery.delivery_id,
        "project_id": delivery.project_id,
        "title": delivery.title,
        "status": delivery.status,
        "latest_version_no": latest,
        "created_at": delivery.created_at,
    }


def _version_by_no(versions: list[DeliveryVersion], version_no: int) -> DeliveryVersion:
    for version in versions:
        if version.version_no == version_no:
            return version
    raise HTTPException(status_code=404, detail="未找到该交付版本。")


def _sandboxed_version_path(project_id: str, version: DeliveryVersion) -> Path:
    base = deliverable_dir(project_id).resolve()
    target = Path(version.file_path).resolve()
    if (target != base and base not in target.parents) or not target.is_file():
        raise HTTPException(status_code=404, detail="未找到该交付文件。")
    return target


def _provenance(project_id: str, version: DeliveryVersion | None) -> list[dict[str, Any]]:
    if version is None:
        return []
    by_id = {record.record_id: record for record in make_project_store().list_records(project_id)}
    return [_provenance_row(record_id, by_id) for record_id in version.record_ids]


def _provenance_row(record_id: str, by_id: dict[str, Any]) -> dict[str, Any]:
    record = by_id.get(record_id)
    if record is None:
        raise HTTPException(status_code=500, detail=f"交付归档引用了不存在的病害记录：{record_id}")
    return {
        "record_id": record.record_id,
        "defect_category": record.payload.defect_category,
        "source_thread_id": record.source_thread_id,
    }


def _error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, BoundaryError):
        payload = error_info_payload(exc.info)
    elif isinstance(exc, sqlite3.Error):
        payload = error_info_payload(classify_db_error(exc, step="DELIVERY_DB"))
    else:
        payload = error_info_payload(classify_delivery_error(exc, step="ARCHIVE"))
    return {"type": exc.__class__.__name__, **payload}


def _require_project(project_id: str) -> None:
    if make_project_store().get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="未找到该巡查任务。")


def _project_locale(project_id: str) -> str:
    project = make_project_store().get_project(project_id)
    return project.locale if project else DEFAULT_LOCALE


def _user(project_id: str) -> str:
    project = make_project_store().get_project(project_id)
    return project.user_id if project else "unknown"
