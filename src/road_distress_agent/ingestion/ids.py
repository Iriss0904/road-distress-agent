"""Stable identifier helpers for ingestion artifacts."""

from __future__ import annotations

import hashlib


def stable_id(*parts: object, length: int = 16) -> str:
    payload = "\n".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]
