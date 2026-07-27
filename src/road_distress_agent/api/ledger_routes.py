"""Aggregation endpoints for the task-ledger view."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from road_distress_agent.api.paths import web_db_path
from road_distress_agent.api.thread_status_service import threads_with_status
from road_distress_agent.projects.evidence import evidence_completeness
from road_distress_agent.projects.models import DefectRecord
from road_distress_agent.projects.store import _project_db_path, make_project_store

router = APIRouter(prefix="/api/ledger", tags=["ledger"])


@router.get("/unfiled")
def unfiled(user_id: str) -> list[dict[str, Any]]:
    threads = threads_with_status(str(web_db_path()), str(_project_db_path()), user_id=user_id)
    return [thread for thread in threads if thread["status"] == "ready"]


@router.get("/{project_id}")
def ledger(project_id: str) -> dict[str, Any]:
    store = make_project_store()
    project = store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="未找到该巡查任务。")
    rows = [_ledger_row(record) for record in store.list_records(project_id)]
    return {"project": project.model_dump(mode="json"), "rows": rows}


def _ledger_row(record: DefectRecord) -> dict[str, Any]:
    payload = record.payload
    return {
        "record_id": record.record_id,
        "defect_category": payload.defect_category,
        "chosen_method": payload.chosen_method,
        "location": payload.known_features.get("location"),
        "evidence": evidence_completeness(payload),
        "review": record.status,
        "source_thread_id": record.source_thread_id,
    }
