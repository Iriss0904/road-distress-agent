"""Persistent long-term memory tool."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from road_distress_agent.state import LoadedMemory
from road_distress_agent.tools.memory_idempotency import (
    MemoryWriteCommand,
    MemoryWriteResult,
    apply_memory_once,
)

MEMORY_NAMESPACE = "memory"
DEFAULT_MEMORY_DB = "runtime/long_term_memory.sqlite3"
MEMORY_CATEGORIES = (
    "user_preferences",
    "regional_context",
    "resource_constraints",
    "case_summaries",
)


class MemoryTool(Protocol):
    def load(self, user_id: str) -> LoadedMemory: ...

    def save(self, user_id: str, memory: LoadedMemory) -> None: ...

    def apply_once(self, command: MemoryWriteCommand) -> MemoryWriteResult: ...


class SQLiteMemoryTool:
    """Store one JSON document per memory category and user."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or _memory_db_path())

    def load(self, user_id: str) -> LoadedMemory:
        _validate_user_id(user_id)
        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT key, value_json
                FROM long_term_memory
                WHERE namespace = ? AND user_id = ?
                """,
                (MEMORY_NAMESPACE, user_id),
            ).fetchall()
        values = {key: json.loads(value_json) for key, value_json in rows}
        return LoadedMemory.model_validate(_category_values(values))

    def save(self, user_id: str, memory: LoadedMemory) -> None:
        _validate_user_id(user_id)
        self._ensure_schema()
        payload = _category_values(memory.model_dump(mode="json"))
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO long_term_memory(namespace, user_id, key, value_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, user_id, key)
                DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at
                """,
                [_upsert_row(user_id, key, payload[key]) for key in MEMORY_CATEGORIES],
            )

    def apply_once(self, command: MemoryWriteCommand) -> MemoryWriteResult:
        _validate_user_id(command.user_id)
        if not command.request_id.strip():
            raise ValueError("request_id is required for idempotent memory writes.")
        return apply_memory_once(self.db_path, command)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    namespace TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(namespace, user_id, key)
                )
                """
            )


def make_memory_tool(db_path: str | Path | None = None) -> MemoryTool:
    return SQLiteMemoryTool(db_path=db_path)


def _category_values(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: values.get(key, [] if key == "case_summaries" else {}) for key in MEMORY_CATEGORIES
    }


def _upsert_row(user_id: str, key: str, value: Any) -> tuple[str, str, str, str, str]:
    return (
        MEMORY_NAMESPACE,
        user_id,
        key,
        json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True),
        _utc_now_iso(),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _memory_db_path() -> Path:
    raw = Path(os.environ.get("ROAD_DISTRESS_MEMORY_DB", DEFAULT_MEMORY_DB))
    if raw.is_absolute():
        return raw
    from road_distress_agent.api.paths import data_dir

    return data_dir() / raw


def _validate_user_id(user_id: str) -> None:
    if not user_id or not user_id.strip():
        raise ValueError("user_id is required for long-term memory access.")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
