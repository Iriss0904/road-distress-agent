"""Execution policy for independent planned KB retrieval hops."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

KB_HOP_PARALLEL_ENV = "ROAD_DISTRESS_KB01_PARALLEL_HOP_RETRIEVAL"
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
DEFAULT_PARALLEL_VALUE = "0"

HopResult = TypeVar("HopResult")


def kb_hop_parallel_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return the explicit KB-01 gate, which is disabled by default."""
    values = env if env is not None else os.environ
    raw = values.get(KB_HOP_PARALLEL_ENV, DEFAULT_PARALLEL_VALUE)
    normalized = raw.strip().lower()
    if normalized in ENV_TRUE_VALUES:
        return True
    if normalized in ENV_FALSE_VALUES:
        return False
    raise ValueError(f"{KB_HOP_PARALLEL_ENV} must be true/false, got {raw!r}.")


def run_kb_hop_calls(
    calls: Sequence[Callable[[], HopResult]],
    *,
    parallel: bool,
) -> list[HopResult]:
    """Execute hop calls and always collect their results in plan order."""
    if not parallel or len(calls) < 2:
        return [call() for call in calls]
    with ThreadPoolExecutor(
        max_workers=len(calls),
        thread_name_prefix="kb-hop-retrieval",
    ) as executor:
        futures = [executor.submit(call) for call in calls]
        return [future.result() for future in futures]
