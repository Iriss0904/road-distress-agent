"""SQLite metadata index for filesystem-backed delivery archives."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from road_distress_agent.delivery.store_models import (
    Delivery,
    DeliveryStatus,
    DeliveryVersion,
    FileFormat,
    GeneratedBy,
)

DEFAULT_DELIVERY_INDEX_DB = "runtime/delivery_index.sqlite3"


class DeliveryStore(Protocol):
    def create_delivery(self, *, project_id: str, user_id: str, title: str) -> Delivery: ...

    def get_delivery(self, delivery_id: str) -> Delivery | None: ...

    def add_version(
        self,
        *,
        delivery_id: str,
        file_path: str,
        file_format: FileFormat,
        generated_by: GeneratedBy,
        record_ids: list[str],
        note: str | None = None,
    ) -> DeliveryVersion: ...

    def list_deliveries(self, project_id: str | None, *, user_id: str) -> list[Delivery]: ...

    def get_versions(self, delivery_id: str) -> list[DeliveryVersion]: ...

    def set_status(self, delivery_id: str, status: DeliveryStatus) -> Delivery: ...


class SQLiteDeliveryStore:
    """SQLite-backed delivery archive metadata store."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or _delivery_index_db_path())

    def create_delivery(self, *, project_id: str, user_id: str, title: str) -> Delivery:
        delivery = Delivery(project_id=project_id, user_id=user_id, title=title)
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute(_INSERT_DELIVERY, _delivery_row(delivery))
        return delivery

    def get_delivery(self, delivery_id: str) -> Delivery | None:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(_SELECT_DELIVERY, (delivery_id,)).fetchone()
        return _delivery_from_row(row) if row else None

    def add_version(
        self,
        *,
        delivery_id: str,
        file_path: str,
        file_format: FileFormat,
        generated_by: GeneratedBy,
        record_ids: list[str],
        note: str | None = None,
    ) -> DeliveryVersion:
        if self.get_delivery(delivery_id) is None:
            raise KeyError(f"unknown delivery_id: {delivery_id}")
        version = self._build_version(
            delivery_id=delivery_id,
            file_path=file_path,
            file_format=file_format,
            generated_by=generated_by,
            record_ids=record_ids,
            note=note,
        )
        with self._connect() as conn:
            conn.execute(_INSERT_VERSION, _version_row(version))
        return version

    def list_deliveries(self, project_id: str | None, *, user_id: str) -> list[Delivery]:
        self._ensure_schema()
        query, params = _deliveries_query(project_id, user_id)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_delivery_from_row(row) for row in rows]

    def get_versions(self, delivery_id: str) -> list[DeliveryVersion]:
        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(_SELECT_VERSIONS, (delivery_id,)).fetchall()
        return [_version_from_row(row) for row in rows]

    def set_status(self, delivery_id: str, status: DeliveryStatus) -> Delivery:
        current = self.get_delivery(delivery_id)
        if current is None:
            raise KeyError(f"unknown delivery_id: {delivery_id}")
        updated = current.model_copy(update={"status": status, "updated_at": _now_from_model()})
        with self._connect() as conn:
            conn.execute(_UPDATE_DELIVERY_STATUS, (updated.status, updated.updated_at, delivery_id))
        return updated

    def _build_version(
        self,
        *,
        delivery_id: str,
        file_path: str,
        file_format: FileFormat,
        generated_by: GeneratedBy,
        record_ids: list[str],
        note: str | None,
    ) -> DeliveryVersion:
        checksum = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
        return DeliveryVersion(
            delivery_id=delivery_id,
            version_no=self._next_version_no(delivery_id),
            file_path=file_path,
            file_format=file_format,
            checksum=checksum,
            generated_by=generated_by,
            record_ids=record_ids,
            note=note,
        )

    def _next_version_no(self, delivery_id: str) -> int:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(_SELECT_MAX_VERSION_NO, (delivery_id,)).fetchone()
        return int(row["max_no"] or 0) + 1

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)


def make_delivery_store(db_path: str | Path | None = None) -> DeliveryStore:
    return SQLiteDeliveryStore(db_path=db_path)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS deliveries (
    delivery_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS delivery_versions (
    version_id TEXT PRIMARY KEY,
    delivery_id TEXT NOT NULL,
    version_no INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    file_format TEXT NOT NULL,
    checksum TEXT NOT NULL,
    generated_by TEXT NOT NULL,
    note TEXT,
    record_ids_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_delivery_versions_delivery
ON delivery_versions(delivery_id);
"""

_DELIVERY_COLUMNS = "delivery_id, project_id, user_id, status, title, created_at, updated_at"
_VERSION_COLUMNS = (
    "version_id, delivery_id, version_no, file_path, file_format, checksum, "
    "generated_by, note, record_ids_json, created_at"
)
_INSERT_DELIVERY = f"INSERT INTO deliveries({_DELIVERY_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)"
_SELECT_DELIVERY = f"SELECT {_DELIVERY_COLUMNS} FROM deliveries WHERE delivery_id = ?"
_SELECT_DELIVERIES_BY_USER = (
    f"SELECT {_DELIVERY_COLUMNS} FROM deliveries WHERE user_id = ? ORDER BY created_at DESC"
)
_SELECT_DELIVERIES_BY_PROJECT = (
    f"SELECT {_DELIVERY_COLUMNS} FROM deliveries "
    "WHERE project_id = ? AND user_id = ? ORDER BY created_at DESC"
)
_UPDATE_DELIVERY_STATUS = "UPDATE deliveries SET status = ?, updated_at = ? WHERE delivery_id = ?"
_INSERT_VERSION = (
    f"INSERT INTO delivery_versions({_VERSION_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_SELECT_VERSIONS = (
    f"SELECT {_VERSION_COLUMNS} FROM delivery_versions WHERE delivery_id = ? ORDER BY version_no"
)
_SELECT_MAX_VERSION_NO = (
    "SELECT MAX(version_no) AS max_no FROM delivery_versions WHERE delivery_id = ?"
)


def _deliveries_query(project_id: str | None, user_id: str) -> tuple[str, tuple[Any, ...]]:
    if project_id is None:
        return _SELECT_DELIVERIES_BY_USER, (user_id,)
    return _SELECT_DELIVERIES_BY_PROJECT, (project_id, user_id)


def _delivery_row(delivery: Delivery) -> tuple[Any, ...]:
    return (
        delivery.delivery_id,
        delivery.project_id,
        delivery.user_id,
        delivery.status,
        delivery.title,
        delivery.created_at,
        delivery.updated_at,
    )


def _version_row(version: DeliveryVersion) -> tuple[Any, ...]:
    return (
        version.version_id,
        version.delivery_id,
        version.version_no,
        version.file_path,
        version.file_format,
        version.checksum,
        version.generated_by,
        version.note,
        json.dumps(version.record_ids, ensure_ascii=False),
        version.created_at,
    )


def _delivery_from_row(row: sqlite3.Row) -> Delivery:
    return Delivery(
        delivery_id=row["delivery_id"],
        project_id=row["project_id"],
        user_id=row["user_id"],
        status=_cast_status(row["status"]),
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _version_from_row(row: sqlite3.Row) -> DeliveryVersion:
    return DeliveryVersion(
        version_id=row["version_id"],
        delivery_id=row["delivery_id"],
        version_no=row["version_no"],
        file_path=row["file_path"],
        file_format=_cast_format(row["file_format"]),
        checksum=row["checksum"],
        generated_by=_cast_generated_by(row["generated_by"]),
        note=row["note"],
        record_ids=json.loads(row["record_ids_json"]),
        created_at=row["created_at"],
    )


def _cast_status(value: str) -> DeliveryStatus:
    if value not in ("draft", "final"):
        raise ValueError(f"corrupt delivery status: {value!r}")
    return value  # type: ignore[return-value]


def _cast_format(value: str) -> FileFormat:
    if value not in ("docx", "pdf", "xlsx"):
        raise ValueError(f"corrupt file format: {value!r}")
    return value  # type: ignore[return-value]


def _cast_generated_by(value: str) -> GeneratedBy:
    if value not in ("auto", "human"):
        raise ValueError(f"corrupt generated_by: {value!r}")
    return value  # type: ignore[return-value]


def _delivery_index_db_path() -> Path:
    raw = Path(os.environ.get("ROAD_DISTRESS_DELIVERY_INDEX_DB", DEFAULT_DELIVERY_INDEX_DB))
    if raw.is_absolute():
        return raw
    from road_distress_agent.api.paths import data_dir

    return data_dir() / raw


def _now_from_model() -> str:
    return datetime.now(timezone.utc).isoformat()
