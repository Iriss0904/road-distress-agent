"""Feature flags and payload helpers for progressive SSE delivery."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

FEATURE_ENV_NAME = "ROAD_DISTRESS_PROGRESSIVE_DELIVERY"
PROGRESSIVE_STAGE_FLAG = "d1"
FINAL_TEXT_STREAM_FLAG = "d2"
FINAL_CHUNK_CHARACTERS = 64
_KNOWN_FLAGS = frozenset({PROGRESSIVE_STAGE_FLAG, FINAL_TEXT_STREAM_FLAG})


@dataclass(frozen=True)
class ProgressiveDeliveryFeatures:
    progressive_stages: bool = False
    final_text_stream: bool = False


def progressive_delivery_features(
    env: Mapping[str, str] | None = None,
) -> ProgressiveDeliveryFeatures:
    source = os.environ if env is None else env
    enabled = _enabled_flags(source.get(FEATURE_ENV_NAME, ""))
    return ProgressiveDeliveryFeatures(
        progressive_stages=PROGRESSIVE_STAGE_FLAG in enabled,
        final_text_stream=FINAL_TEXT_STREAM_FLAG in enabled,
    )


def final_text_chunks(text: str) -> Iterator[dict[str, object]]:
    if not isinstance(text, str):
        raise TypeError("Safety-approved final text must be a string.")
    for index, offset in enumerate(range(0, len(text), FINAL_CHUNK_CHARACTERS)):
        end = offset + FINAL_CHUNK_CHARACTERS
        yield {
            "index": index,
            "text": text[offset:end],
            "final": end >= len(text),
        }


def _enabled_flags(raw: str) -> frozenset[str]:
    values = frozenset(value.strip().lower() for value in raw.split(",") if value.strip())
    unknown = values - _KNOWN_FLAGS
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown {FEATURE_ENV_NAME} flags: {joined}")
    return values
