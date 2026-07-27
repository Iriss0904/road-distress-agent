"""Request-scoped idempotent writes for long-term memory."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from road_distress_agent.state import LoadedMemory

MEMORY_NAMESPACE = "memory"
MEMORY_TYPES = (
    "user_preferences",
    "regional_context",
    "resource_constraints",
    "case_summaries",
)


@dataclass(frozen=True)
class MemoryWriteCommand:
    user_id: str
    request_id: str
    thread_id: str | None
    user_preferences: Mapping[str, str]
    regional_context: Mapping[str, str]
    resource_constraints: Mapping[str, str]
    case_summary: str | None


@dataclass(frozen=True)
class MemoryWriteResult:
    memory: LoadedMemory
    applied_types: tuple[str, ...]
    skipped_types: tuple[str, ...]


def apply_memory_once(db_path: str | Path, command: MemoryWriteCommand) -> MemoryWriteResult:
    """Apply one request's complete memory decision in a single transaction."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        _ensure_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        receipts = _load_receipts(conn, command)
        current = _load_memory(conn, command.user_id)
        if receipts:
            _validate_complete_receipts(receipts)
            return MemoryWriteResult(current, (), MEMORY_TYPES)
        updated, applied = _merge(current, command)
        _upsert_applied(conn, command.user_id, updated, applied)
        _insert_receipts(conn, command, applied)
        conn.commit()
    return MemoryWriteResult(updated, applied, ())


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS long_term_memory (
            namespace TEXT NOT NULL, user_id TEXT NOT NULL, key TEXT NOT NULL,
            value_json TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY(namespace, user_id, key)
        );
        CREATE TABLE IF NOT EXISTS memory_write_receipts (
            request_id TEXT NOT NULL, memory_type TEXT NOT NULL,
            user_id TEXT NOT NULL, effect_applied INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(request_id, memory_type)
        );
        """
    )


def _load_receipts(conn: sqlite3.Connection, command: MemoryWriteCommand) -> list[tuple[str, str]]:
    rows = conn.execute(
        "SELECT memory_type, user_id FROM memory_write_receipts WHERE request_id = ?",
        (command.request_id,),
    ).fetchall()
    if rows and any(user_id != command.user_id for _, user_id in rows):
        raise ValueError("request_id is already bound to another memory user.")
    return [(str(memory_type), str(user_id)) for memory_type, user_id in rows]


def _validate_complete_receipts(receipts: list[tuple[str, str]]) -> None:
    found = {memory_type for memory_type, _ in receipts}
    if found != set(MEMORY_TYPES):
        raise RuntimeError("memory write receipts are incomplete for request_id.")


def _load_memory(conn: sqlite3.Connection, user_id: str) -> LoadedMemory:
    rows = conn.execute(
        "SELECT key, value_json FROM long_term_memory WHERE namespace = ? AND user_id = ?",
        (MEMORY_NAMESPACE, user_id),
    ).fetchall()
    values = {key: json.loads(value_json) for key, value_json in rows}
    payload = {key: values.get(key, [] if key == "case_summaries" else {}) for key in MEMORY_TYPES}
    return LoadedMemory.model_validate(payload)


def _merge(
    current: LoadedMemory, command: MemoryWriteCommand
) -> tuple[LoadedMemory, tuple[str, ...]]:
    now = _now()
    updates = {
        "user_preferences": _merge_records(
            current.user_preferences, command.user_preferences, command, now
        ),
        "regional_context": _merge_records(
            current.regional_context, command.regional_context, command, now
        ),
        "resource_constraints": _merge_records(
            current.resource_constraints, command.resource_constraints, command, now
        ),
        "case_summaries": _merge_summary(current.case_summaries, command, now),
    }
    applied = tuple(key for key in MEMORY_TYPES if updates[key] != getattr(current, key))
    return LoadedMemory.model_validate(updates), applied


def _merge_records(
    current: dict[str, Any], updates: Mapping[str, str], command: MemoryWriteCommand, now: str
) -> dict[str, Any]:
    merged = dict(current)
    for key, value in updates.items():
        if cleaned := value.strip():
            merged[key] = _record(cleaned, command, now)
    return merged


def _merge_summary(
    current: list[dict[str, Any]], command: MemoryWriteCommand, now: str
) -> list[dict[str, Any]]:
    summary = command.case_summary
    if not summary or not summary.strip():
        return list(current)
    return [*current, _record(summary.strip(), command, now)]


def _record(value: str, command: MemoryWriteCommand, now: str) -> dict[str, str | None]:
    return {
        "value": value,
        "updated_at": now,
        "source_thread_id": command.thread_id,
        "source_request_id": command.request_id,
    }


def _upsert_applied(
    conn: sqlite3.Connection, user_id: str, memory: LoadedMemory, applied: tuple[str, ...]
) -> None:
    payload = memory.model_dump(mode="json")
    for memory_type in applied:
        conn.execute(
            """INSERT INTO long_term_memory(namespace,user_id,key,value_json,updated_at)
            VALUES (?, ?, ?, ?, ?) ON CONFLICT(namespace,user_id,key) DO UPDATE SET
            value_json=excluded.value_json, updated_at=excluded.updated_at""",
            (MEMORY_NAMESPACE, user_id, memory_type, _json(payload[memory_type]), _now()),
        )


def _insert_receipts(
    conn: sqlite3.Connection, command: MemoryWriteCommand, applied: tuple[str, ...]
) -> None:
    now = _now()
    conn.executemany(
        """INSERT INTO memory_write_receipts
        (request_id,memory_type,user_id,effect_applied,created_at) VALUES (?, ?, ?, ?, ?)""",
        [
            (command.request_id, key, command.user_id, int(key in applied), now)
            for key in MEMORY_TYPES
        ],
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
