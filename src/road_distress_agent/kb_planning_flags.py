"""Feature gates for KB query planning."""

from __future__ import annotations

import os
from collections.abc import Mapping

KB_QUERY_PLANNING_ENABLED_ENV = "KB_QUERY_PLANNING_ENABLED"
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
DEFAULT_KB_QUERY_PLANNING_ENABLED = "1"


def kb_query_planning_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    raw = values.get(KB_QUERY_PLANNING_ENABLED_ENV, DEFAULT_KB_QUERY_PLANNING_ENABLED)
    return raw.strip().lower() in ENV_TRUE_VALUES
