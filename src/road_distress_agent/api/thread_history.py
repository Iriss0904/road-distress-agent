"""Persistent chat-history index for the web workbench."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from road_distress_agent.localization import display_term

TITLE_LIMIT = 28
SEARCH_LIMIT = 12000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_threads (
    thread_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    transcript_json TEXT NOT NULL,
    snapshot_json TEXT,
    search_text TEXT NOT NULL,
    deleted_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_chat_threads_user_updated
ON chat_threads(user_id, updated_at DESC);
"""


def record_user_turn(db_path: str, *, thread_id: str, user_id: str, text: str) -> None:
    _validate(thread_id, user_id)
    now = _now()
    with _connect(db_path) as conn:
        row = conn.execute(_SELECT_THREAD, (thread_id,)).fetchone()
        if row is None:
            _insert_thread(conn, thread_id, user_id, text, now)
            return
        transcript = _append_entry(_transcript(row), _entry("user", text), now)
        _update_thread(conn, row, transcript, _title(row, text), None, now)


def record_snapshot(
    db_path: str, *, thread_id: str, user_id: str, snapshot: dict[str, Any]
) -> None:
    _validate(thread_id, user_id)
    now = _now()
    entries = _system_entries(snapshot)
    with _connect(db_path) as conn:
        row = conn.execute(_SELECT_THREAD, (thread_id,)).fetchone()
        if row is None:
            raise ValueError(f"chat thread {thread_id} has no recorded user turn.")
        transcript = _transcript(row)
        for entry in entries:
            transcript = _append_entry(transcript, entry, now)
        title = _snapshot_title(snapshot) or _title(row, thread_id)
        _update_thread(conn, row, transcript, title, snapshot, now)


def record_system_message(db_path: str, *, thread_id: str, user_id: str, text: str) -> None:
    _validate(thread_id, user_id)
    now = _now()
    with _connect(db_path) as conn:
        row = conn.execute(_SELECT_THREAD, (thread_id,)).fetchone()
        if row is None:
            raise ValueError(f"chat thread {thread_id} has no recorded user turn.")
        transcript = _append_entry(_transcript(row), _entry("system", text), now)
        _update_thread(conn, row, transcript, _title(row, thread_id), None, now)


def list_threads(db_path: str, *, user_id: str, q: str | None = None) -> list[dict[str, Any]]:
    _validate_user(user_id)
    query = _LIST_THREADS_SEARCH if q and q.strip() else _LIST_THREADS
    params = (user_id, f"%{q.strip()}%") if q and q.strip() else (user_id,)
    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [_thread_summary(row) for row in rows]


def get_thread(db_path: str, thread_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(_SELECT_THREAD, (thread_id,)).fetchone()
    return _thread_detail(row) if row else None


def rename_thread(db_path: str, *, thread_id: str, title: str) -> dict[str, Any]:
    clean = " ".join(title.strip().split())
    if not clean:
        raise ValueError("title is required.")
    clean = clean[:TITLE_LIMIT]
    with _connect(db_path) as conn:
        row = conn.execute(_SELECT_THREAD, (thread_id,)).fetchone()
        if row is None:
            raise KeyError(thread_id)
        conn.execute(
            "UPDATE chat_threads SET title = ?, updated_at = ? WHERE thread_id = ?",
            (clean, _now(), thread_id),
        )
    return {**_thread_summary(row), "title": clean}


def soft_delete_thread(db_path: str, *, thread_id: str) -> None:
    if not thread_id.strip():
        raise ValueError("thread_id is required.")
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE chat_threads SET deleted_at = ? WHERE thread_id = ?",
            (_now(), thread_id),
        )


def _connect(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _ensure_deleted_at_column(conn)
    return conn


def _ensure_deleted_at_column(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(chat_threads)")}
    if "deleted_at" not in columns:
        conn.execute("ALTER TABLE chat_threads ADD COLUMN deleted_at TEXT")


def _insert_thread(
    conn: sqlite3.Connection, thread_id: str, user_id: str, text: str, now: str
) -> None:
    title = _truncate(text)
    transcript = [{"role": "user", "text": text, "created_at": now}]
    conn.execute(
        _INSERT_THREAD,
        (
            thread_id,
            user_id,
            title,
            now,
            now,
            _json(transcript),
            None,
            _search(thread_id, title, transcript),
        ),
    )


def _update_thread(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    transcript: list[dict[str, Any]],
    title: str,
    snapshot: dict[str, Any] | None,
    now: str,
) -> None:
    snapshot_json = _json(snapshot) if snapshot is not None else row["snapshot_json"]
    conn.execute(
        _UPDATE_THREAD,
        (
            _truncate(title),
            now,
            _json(transcript),
            snapshot_json,
            _search(row["thread_id"], title, transcript),
            row["thread_id"],
        ),
    )


def _append_entry(
    transcript: list[dict[str, Any]], entry: dict[str, Any], now: str
) -> list[dict[str, Any]]:
    role = str(entry["role"])
    references = entry.get("references")
    cleaned = str(entry["text"]).strip()
    if not cleaned:
        return transcript
    if transcript and transcript[-1]["role"] == role and transcript[-1]["text"] == cleaned:
        if references and transcript[-1].get("references") != references:
            return [*transcript[:-1], {**transcript[-1], "references": references}]
        return transcript
    entry = {"role": role, "text": cleaned, "created_at": now}
    if references:
        entry["references"] = references
    return [*transcript, entry]


def _system_entries(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    english = snapshot.get("locale") == "en-US"
    if snapshot.get("scene_description"):
        prefix = "Site image analysis: " if english else "现场图片分析："
        entries.append(_entry("system", prefix + str(snapshot["scene_description"])))
    answer_refs = snapshot.get("reference_index") or []
    candidate_refs = snapshot.get("candidate_reference_index") or []
    for key in ("direct_message", "final_answer_message"):
        if snapshot.get(key):
            entries.append(_entry("system", str(snapshot[key]), references=answer_refs))
    if snapshot.get("awaiting_message"):
        entries.append(
            _entry("system", str(snapshot["awaiting_message"]), references=candidate_refs)
        )
    return entries


def _entry(role: str, text: str, *, references: list[Any] | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"role": role, "text": text}
    if references:
        entry["references"] = references
    return entry


def _snapshot_title(snapshot: dict[str, Any]) -> str | None:
    category = snapshot.get("defect_category")
    if not category:
        return None
    title_category = display_term(str(category), snapshot.get("locale"))
    features = snapshot.get("known_features") or {}
    location = features.get("location")
    size = _size_text(features)
    detail = " ".join(str(item) for item in (location, size) if item)
    return f"{title_category} {detail}".strip()


def _size_text(features: dict[str, Any]) -> str | None:
    for key, unit in (
        ("crack_width_mm", "mm"),
        ("width_mm", "mm"),
        ("depth_mm", "mm"),
        ("area_m2", "㎡"),
    ):
        if features.get(key) is not None:
            return f"{features[key]}{unit}"
    return None


def _transcript(row: sqlite3.Row) -> list[dict[str, Any]]:
    return json.loads(row["transcript_json"])


def _title(row: sqlite3.Row, fallback: str) -> str:
    return str(row["title"] or fallback)


def _thread_summary(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "thread_id": row["thread_id"],
        "user_id": row["user_id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _thread_detail(row: sqlite3.Row) -> dict[str, Any]:
    snapshot = json.loads(row["snapshot_json"]) if row["snapshot_json"] else None
    return {**_thread_summary(row), "transcript": _transcript(row), "stored_snapshot": snapshot}


def _search(thread_id: str, title: str, transcript: list[dict[str, Any]]) -> str:
    body = "\n".join(item["text"] for item in transcript)
    return f"{thread_id}\n{title}\n{body}"[:SEARCH_LIMIT]


def _truncate(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    return cleaned[:TITLE_LIMIT] or "未命名对话"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate(thread_id: str, user_id: str) -> None:
    if not thread_id.strip():
        raise ValueError("thread_id is required.")
    _validate_user(user_id)


def _validate_user(user_id: str) -> None:
    if not user_id.strip():
        raise ValueError("user_id is required.")


_SELECT_THREAD = "SELECT * FROM chat_threads WHERE thread_id = ?"
_INSERT_THREAD = (
    "INSERT INTO chat_threads(thread_id, user_id, title, created_at, updated_at, "
    "transcript_json, snapshot_json, search_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)
_UPDATE_THREAD = (
    "UPDATE chat_threads SET title = ?, updated_at = ?, transcript_json = ?, "
    "snapshot_json = ?, search_text = ? WHERE thread_id = ?"
)
_LIST_THREADS = (
    "SELECT * FROM chat_threads WHERE user_id = ? AND deleted_at IS NULL ORDER BY updated_at DESC"
)
_LIST_THREADS_SEARCH = (
    "SELECT * FROM chat_threads WHERE user_id = ? AND search_text LIKE ? "
    "AND deleted_at IS NULL ORDER BY updated_at DESC"
)
