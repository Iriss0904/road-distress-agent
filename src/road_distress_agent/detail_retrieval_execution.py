"""Execution policy for the independent detail-retrieval pipelines."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Generic, TypeVar

DETAIL_RETRIEVAL_PARALLEL_ENV = "ROAD_DISTRESS_B07_PARALLEL_DETAIL_RETRIEVAL"
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
DEFAULT_PARALLEL_VALUE = "0"
DETAIL_PIPELINE_COUNT = 2

PipelineResult = TypeVar("PipelineResult")


@dataclass(frozen=True)
class DetailPipelines(Generic[PipelineResult]):
    procedure: Callable[[], PipelineResult]
    acceptance: Callable[[], PipelineResult]


def detail_retrieval_parallel_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return the explicit B-07 gate, which is disabled by default."""
    values = env if env is not None else os.environ
    raw = values.get(DETAIL_RETRIEVAL_PARALLEL_ENV, DEFAULT_PARALLEL_VALUE)
    normalized = raw.strip().lower()
    if normalized in ENV_TRUE_VALUES:
        return True
    if normalized in ENV_FALSE_VALUES:
        return False
    raise ValueError(f"{DETAIL_RETRIEVAL_PARALLEL_ENV} must be true/false, got {raw!r}.")


def run_detail_pipelines(
    pipelines: DetailPipelines[PipelineResult],
    *,
    parallel: bool,
) -> tuple[PipelineResult, PipelineResult]:
    """Run both pipelines and always collect results in baseline order."""
    if not parallel:
        return pipelines.procedure(), pipelines.acceptance()
    with ThreadPoolExecutor(
        max_workers=DETAIL_PIPELINE_COUNT,
        thread_name_prefix="detail-retrieval",
    ) as executor:
        procedure = executor.submit(pipelines.procedure)
        acceptance = executor.submit(pipelines.acceptance)
        return procedure.result(), acceptance.result()
