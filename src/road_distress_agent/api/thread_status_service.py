"""Enrich thread summaries with derived status in one project-ledger pass."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from road_distress_agent.api.thread_history import list_threads
from road_distress_agent.projects.store import SQLiteProjectStore
from road_distress_agent.projects.thread_status import derive_thread_status


def threads_with_status(
    db_path: str, project_db: str | Path, *, user_id: str, q: str | None = None
) -> list[dict[str, Any]]:
    promotions = _promotion_index(project_db, user_id)
    summaries = list_threads(db_path, user_id=user_id, q=q)
    snapshots = _snapshots_by_thread(db_path, [row["thread_id"] for row in summaries])
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        thread_id = summary["thread_id"]
        rows.append(
            {
                **summary,
                "status": derive_thread_status(
                    snapshot=snapshots.get(thread_id),
                    has_active_record=thread_id in promotions,
                    is_delivered=promotions.get(thread_id, False),
                ),
            }
        )
    return rows


def _snapshots_by_thread(db_path: str, thread_ids: list[str]) -> dict[str, dict[str, Any] | None]:
    if not thread_ids:
        return {}
    placeholders = ",".join("?" for _ in thread_ids)
    query = f"SELECT thread_id, snapshot_json FROM chat_threads WHERE thread_id IN ({placeholders})"
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, thread_ids).fetchall()
    return {row["thread_id"]: _snapshot(row["snapshot_json"]) for row in rows}


def _snapshot(raw: str | None) -> dict[str, Any] | None:
    return json.loads(raw) if raw else None


def _promotion_index(project_db: str | Path, user_id: str) -> dict[str, bool]:
    store = SQLiteProjectStore(project_db)
    store._ensure_schema()
    with sqlite3.connect(store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(_PROMOTION_ROWS, (user_id, "active")).fetchall()
        delivered_ids = _delivered_record_ids(conn)
    return _index_rows(rows, delivered_ids)


def _delivered_record_ids(conn: sqlite3.Connection) -> set[str]:
    if not _has_delivery_versions(conn):
        return set()
    rows = conn.execute("SELECT record_ids_json FROM delivery_versions").fetchall()
    return {record_id for row in rows for record_id in _record_ids(row[0])}


def _has_delivery_versions(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        ("delivery_versions",),
    ).fetchone()
    return row is not None


def _record_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("delivery_versions.record_ids_json must be a list.")
    return [str(item) for item in data]


def _index_rows(rows: list[sqlite3.Row], delivered_ids: set[str]) -> dict[str, bool]:
    index: dict[str, bool] = {}
    for row in rows:
        delivered = row["project_status"] == "delivered" or row["record_id"] in delivered_ids
        index[row["source_thread_id"]] = index.get(row["source_thread_id"], False) or delivered
    return index


_PROMOTION_ROWS = (
    "SELECT r.record_id, r.source_thread_id, p.status AS project_status "
    "FROM defect_records r JOIN projects p ON p.project_id = r.project_id "
    "WHERE p.user_id = ? AND r.status = ?"
)
