"""Filesystem paths configured for a standalone public deployment."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = "data"
DEFAULT_WEB_DB = "runtime/web-workflow.sqlite3"


def data_dir() -> Path:
    """Return the configured data root, resolving relative paths to the repository."""
    raw = Path(os.environ.get("DATA_DIR", DEFAULT_DATA_DIR))
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def web_db_path() -> Path:
    raw = os.environ.get("ROAD_DISTRESS_WEB_DB") or DEFAULT_WEB_DB
    path = Path(raw)
    return path if path.is_absolute() else data_dir() / path
