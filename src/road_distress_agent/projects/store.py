"""Persistent inspection-project ledger.

Mirrors ``tools/memory.py``'s ``SQLiteMemoryTool`` style: one SQLite file, plain
schema, dependency-injectable via the ``ProjectStore`` Protocol. Two tables:
``projects`` and ``defect_records``.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Protocol

from road_distress_agent.localization import DEFAULT_LOCALE, normalize_locale
from road_distress_agent.projects.models import (
    DefectPayload,
    DefectRecord,
    InspectionProject,
    ProjectStatus,
    RecordStatus,
)

DEFAULT_PROJECT_DB = "runtime/inspection_projects.sqlite3"
DUPLICATE_ACTIVE_PROJECT_NAME = "已有命名相同的任务存在"
ACTIVE_PROJECT_STATUSES = ("active", "archiving")


class DuplicateProjectNameError(ValueError):
    """Raised when an active/archiving project name already exists for a user."""


class ProjectStore(Protocol):
    def create_project(self, *, user_id: str, name: str, **fields: Any) -> InspectionProject: ...

    def get_project(self, project_id: str) -> InspectionProject | None: ...

    def list_projects(self, user_id: str) -> list[InspectionProject]: ...

    def update_project(self, project_id: str, **fields: Any) -> InspectionProject: ...

    def promote_defect(
        self, *, project_id: str, source_thread_id: str, payload: DefectPayload
    ) -> DefectRecord: ...

    def list_records(
        self, project_id: str, *, status: RecordStatus | None = None
    ) -> list[DefectRecord]: ...

    def set_record_status(self, record_id: str, status: RecordStatus) -> DefectRecord: ...


class SQLiteProjectStore:
    """SQLite-backed implementation of :class:`ProjectStore`."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or _project_db_path())

    # ----- projects -----

    def create_project(self, *, user_id: str, name: str, **fields: Any) -> InspectionProject:
        _validate_user_id(user_id)
        clean_name = _clean_project_name(name)
        fields = _normalized_project_fields(fields)
        self._ensure_schema()
        if self._active_name_exists(user_id=user_id, name=clean_name):
            raise DuplicateProjectNameError(DUPLICATE_ACTIVE_PROJECT_NAME)
        project = InspectionProject(user_id=user_id, name=clean_name, **fields)
        with self._connect() as conn:
            conn.execute(_INSERT_PROJECT, _project_row(project))
        return project

    def get_project(self, project_id: str) -> InspectionProject | None:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(_SELECT_PROJECT, (project_id,)).fetchone()
        return _project_from_row(row) if row else None

    def list_projects(self, user_id: str) -> list[InspectionProject]:
        _validate_user_id(user_id)
        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(_SELECT_PROJECTS_BY_USER, (user_id,)).fetchall()
        return [_project_from_row(row) for row in rows]

    def update_project(self, project_id: str, **fields: Any) -> InspectionProject:
        current = self.get_project(project_id)
        if current is None:
            raise KeyError(f"unknown project_id: {project_id}")
        fields = _normalized_project_fields(fields)
        updated = current.touched(**fields)
        with self._connect() as conn:
            conn.execute(_UPDATE_PROJECT, _project_update_row(updated))
        return updated

    # ----- defect records -----

    def promote_defect(
        self, *, project_id: str, source_thread_id: str, payload: DefectPayload
    ) -> DefectRecord:
        if self.get_project(project_id) is None:
            raise KeyError(f"unknown project_id: {project_id}")
        record = DefectRecord(
            project_id=project_id, source_thread_id=source_thread_id, payload=payload
        )
        with self._connect() as conn:
            conn.execute(_INSERT_RECORD, _record_row(record))
        return record

    def list_records(
        self, project_id: str, *, status: RecordStatus | None = None
    ) -> list[DefectRecord]:
        self._ensure_schema()
        query, params = _records_query(project_id, status)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_record_from_row(row) for row in rows]

    def list_active_records(self, project_id: str) -> list[DefectRecord]:
        return self.list_records(project_id, status="active")

    def set_record_status(self, record_id: str, status: RecordStatus) -> DefectRecord:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(_SELECT_RECORD, (record_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown record_id: {record_id}")
            conn.execute(_UPDATE_RECORD_STATUS, (status, record_id))
        return _record_from_row(row).with_status(status)

    # ----- infra -----

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            _ensure_project_locale_column(conn)

    def _active_name_exists(self, *, user_id: str, name: str) -> bool:
        params = (user_id, name, *ACTIVE_PROJECT_STATUSES)
        with self._connect() as conn:
            row = conn.execute(_SELECT_ACTIVE_PROJECT_NAME, params).fetchone()
        return row is not None


def make_project_store(db_path: str | Path | None = None) -> ProjectStore:
    return SQLiteProjectStore(db_path=db_path)


# ─────────────────────────────────────────────────────────────
# SQL + row mapping
# ─────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    locale TEXT NOT NULL DEFAULT 'zh-CN',
    segment TEXT,
    inspector TEXT,
    crew TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS defect_records (
    record_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    source_thread_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_project ON defect_records(project_id);
"""

_PROJECT_COLUMNS = (
    "project_id, user_id, name, locale, segment, inspector, crew, status, created_at, updated_at"
)
_INSERT_PROJECT = f"INSERT INTO projects({_PROJECT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
_SELECT_PROJECT = f"SELECT {_PROJECT_COLUMNS} FROM projects WHERE project_id = ?"
_SELECT_PROJECTS_BY_USER = (
    f"SELECT {_PROJECT_COLUMNS} FROM projects WHERE user_id = ? ORDER BY created_at DESC"
)
_SELECT_ACTIVE_PROJECT_NAME = (
    "SELECT project_id FROM projects WHERE user_id = ? AND name = ? AND status IN (?, ?) LIMIT 1"
)
_UPDATE_PROJECT = (
    "UPDATE projects SET name = ?, locale = ?, segment = ?, inspector = ?, crew = ?, "
    "status = ?, updated_at = ? WHERE project_id = ?"
)

_RECORD_COLUMNS = "record_id, project_id, source_thread_id, status, payload_json, created_at"
_INSERT_RECORD = f"INSERT INTO defect_records({_RECORD_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)"
_SELECT_RECORD = f"SELECT {_RECORD_COLUMNS} FROM defect_records WHERE record_id = ?"
_UPDATE_RECORD_STATUS = "UPDATE defect_records SET status = ? WHERE record_id = ?"


def _records_query(project_id: str, status: RecordStatus | None) -> tuple[str, tuple[Any, ...]]:
    base = f"SELECT {_RECORD_COLUMNS} FROM defect_records WHERE project_id = ?"
    if status is None:
        return f"{base} ORDER BY created_at", (project_id,)
    return f"{base} AND status = ? ORDER BY created_at", (project_id, status)


def _project_row(p: InspectionProject) -> tuple[Any, ...]:
    return (
        p.project_id,
        p.user_id,
        p.name,
        p.locale,
        p.segment,
        p.inspector,
        p.crew,
        p.status,
        p.created_at,
        p.updated_at,
    )


def _project_update_row(p: InspectionProject) -> tuple[Any, ...]:
    return (
        p.name,
        p.locale,
        p.segment,
        p.inspector,
        p.crew,
        p.status,
        p.updated_at,
        p.project_id,
    )


def _project_from_row(row: sqlite3.Row) -> InspectionProject:
    return InspectionProject(
        project_id=row["project_id"],
        user_id=row["user_id"],
        name=row["name"],
        locale=normalize_locale(row["locale"] or DEFAULT_LOCALE),
        segment=row["segment"],
        inspector=row["inspector"],
        crew=row["crew"],
        status=_cast_project_status(row["status"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _record_row(r: DefectRecord) -> tuple[Any, ...]:
    payload_json = json.dumps(r.payload.model_dump(mode="json"), ensure_ascii=False)
    return (r.record_id, r.project_id, r.source_thread_id, r.status, payload_json, r.created_at)


def _record_from_row(row: sqlite3.Row) -> DefectRecord:
    return DefectRecord(
        record_id=row["record_id"],
        project_id=row["project_id"],
        source_thread_id=row["source_thread_id"],
        status=_cast_record_status(row["status"]),
        payload=DefectPayload.model_validate(json.loads(row["payload_json"])),
        created_at=row["created_at"],
    )


def _cast_project_status(value: str) -> ProjectStatus:
    if value not in ("active", "archiving", "delivered"):
        raise ValueError(f"corrupt project status: {value!r}")
    return value  # type: ignore[return-value]


def _cast_record_status(value: str) -> RecordStatus:
    if value not in ("active", "superseded", "incomplete"):
        raise ValueError(f"corrupt record status: {value!r}")
    return value  # type: ignore[return-value]


def _project_db_path() -> Path:
    raw = Path(os.environ.get("ROAD_DISTRESS_PROJECT_DB", DEFAULT_PROJECT_DB))
    if raw.is_absolute():
        return raw
    from road_distress_agent.api.paths import data_dir

    return data_dir() / raw


def _ensure_project_locale_column(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(projects)").fetchall()}
    if "locale" not in columns:
        conn.execute("ALTER TABLE projects ADD COLUMN locale TEXT NOT NULL DEFAULT 'zh-CN'")


def _clean_project_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("project name is required.")
    return cleaned


def _normalized_project_fields(fields: dict[str, Any]) -> dict[str, Any]:
    if "locale" not in fields or fields["locale"] is None:
        return fields
    return {**fields, "locale": normalize_locale(str(fields["locale"]))}


def _validate_user_id(user_id: str) -> None:
    if not user_id or not user_id.strip():
        raise ValueError("user_id is required for project ledger access.")
