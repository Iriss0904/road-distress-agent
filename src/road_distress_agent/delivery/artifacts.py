"""Filesystem location for generated delivery artifacts."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DELIVERY_DIR = "runtime/deliverables"


def deliverable_dir(project_id: str) -> Path:
    """Return (and create) the output directory for one project's deliverables."""
    base = Path(os.environ.get("ROAD_DISTRESS_DELIVERY_DIR", DEFAULT_DELIVERY_DIR))
    if not base.is_absolute():
        from road_distress_agent.api.paths import data_dir

        base = data_dir() / base
    path = base / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path
