"""HTTP routes for the inspection-project ledger and defect promotion.

Kept separate from ``server.py`` so the turn/SSE surface stays focused. The
promotion route reads a finished thread's checkpoint terminal state and projects
it into a DefectRecord; it never mutates the diagnosis graph.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from road_distress_agent.api.paths import web_db_path
from road_distress_agent.checkpointing import sqlite_checkpointer
from road_distress_agent.graph import build_graph
from road_distress_agent.localization import DEFAULT_LOCALE, normalize_locale
from road_distress_agent.projects.promotion import extract_defect_payload, promotion_blocker
from road_distress_agent.projects.store import (
    DUPLICATE_ACTIVE_PROJECT_NAME,
    DuplicateProjectNameError,
    make_project_store,
)

router = APIRouter(prefix="/api/projects", tags=["projects"])


class CreateProjectRequest(BaseModel):
    user_id: str
    name: str
    locale: str = DEFAULT_LOCALE
    segment: str | None = None
    inspector: str | None = None
    crew: str | None = None


class PromoteRequest(BaseModel):
    thread_id: str


@router.post("")
def create_project(req: CreateProjectRequest) -> dict[str, Any]:
    store = make_project_store()
    try:
        locale = normalize_locale(req.locale)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        project = store.create_project(
            user_id=req.user_id,
            name=req.name,
            locale=locale,
            segment=req.segment,
            inspector=req.inspector,
            crew=req.crew,
        )
    except DuplicateProjectNameError as exc:
        raise HTTPException(status_code=409, detail=DUPLICATE_ACTIVE_PROJECT_NAME) from exc
    return project.model_dump(mode="json")


@router.get("")
def list_projects(user_id: str) -> list[dict[str, Any]]:
    store = make_project_store()
    return [p.model_dump(mode="json") for p in store.list_projects(user_id)]


@router.get("/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    store = make_project_store()
    project = store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="未找到该巡查任务。")
    records = [r.model_dump(mode="json") for r in store.list_records(project_id)]
    return {"project": project.model_dump(mode="json"), "records": records}


@router.post("/{project_id}/promote")
def promote_defect(project_id: str, req: PromoteRequest) -> dict[str, Any]:
    store = make_project_store()
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="未找到该巡查任务。")

    values = _read_thread_state(req.thread_id)
    if not values:
        raise HTTPException(status_code=404, detail="未找到该会话。")

    blocker = promotion_blocker(values)
    if blocker is not None:
        raise HTTPException(status_code=409, detail=blocker)

    record = store.promote_defect(
        project_id=project_id,
        source_thread_id=req.thread_id,
        payload=extract_defect_payload(values),
    )
    return record.model_dump(mode="json")


def _read_thread_state(thread_id: str) -> dict[str, Any]:
    with sqlite_checkpointer(str(web_db_path())) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    return snapshot.values or {}
