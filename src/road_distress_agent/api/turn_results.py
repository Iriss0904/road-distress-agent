"""Request identity binding and idempotent final-result persistence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from road_distress_agent.state import AttachmentRef

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turn_results (
    request_id TEXT PRIMARY KEY,
    input_fingerprint TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    snapshot_json TEXT,
    created_at TEXT NOT NULL,
    finalized_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_turn_results_thread
ON turn_results(thread_id);
"""


class RequestIdConflictError(ValueError):
    """Raised when one request ID is reused for different input."""


@dataclass(frozen=True)
class TurnResultClaim:
    request_id: str
    is_new: bool
    snapshot: dict[str, Any] | None


def input_fingerprint(
    *,
    thread_id: str,
    user_id: str,
    locale: str,
    text: str,
    attachments: list[AttachmentRef],
) -> str:
    payload = {
        "attachments": [
            {"media_type": item.media_type, "sha256": item.sha256} for item in attachments
        ],
        "locale": locale,
        "text": text,
        "thread_id": thread_id,
        "user_id": user_id,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def claim_turn_result(
    db_path: str, *, request_id: str, fingerprint: str, thread_id: str
) -> TurnResultClaim:
    with _connect(db_path) as conn:
        inserted = conn.execute(
            "INSERT OR IGNORE INTO turn_results VALUES (?, ?, ?, NULL, ?, NULL)",
            (request_id, fingerprint, thread_id, _now()),
        ).rowcount
        row = conn.execute(
            "SELECT * FROM turn_results WHERE request_id = ?", (request_id,)
        ).fetchone()
    if row["input_fingerprint"] != fingerprint or row["thread_id"] != thread_id:
        raise RequestIdConflictError("request_id 已绑定到不同的请求输入。")
    snapshot = json.loads(row["snapshot_json"]) if row["snapshot_json"] else None
    return TurnResultClaim(request_id=request_id, is_new=bool(inserted), snapshot=snapshot)


def save_turn_result(
    db_path: str, *, request_id: str, fingerprint: str, snapshot: dict[str, Any]
) -> dict[str, Any]:
    encoded = json.dumps(snapshot, ensure_ascii=False)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM turn_results WHERE request_id = ?", (request_id,)
        ).fetchone()
        if row is None or row["input_fingerprint"] != fingerprint:
            raise RequestIdConflictError("request_id 没有匹配的请求声明。")
        if row["snapshot_json"]:
            return json.loads(row["snapshot_json"])
        conn.execute(
            "UPDATE turn_results SET snapshot_json = ?, finalized_at = ? "
            "WHERE request_id = ? AND snapshot_json IS NULL",
            (encoded, _now(), request_id),
        )
    return snapshot


def _connect(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
