"""Server-sent event serialization."""

from __future__ import annotations

import json
from typing import Any


def sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
