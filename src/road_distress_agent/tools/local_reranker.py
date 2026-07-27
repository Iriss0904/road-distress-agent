"""Local BGE reranker used by online retrieval."""

from __future__ import annotations

import os
from collections.abc import Sequence
from threading import Lock
from typing import Any

from road_distress_agent.error_classifiers import classify_model_load_error
from road_distress_agent.errors import BoundaryError, ErrorCategory, make_error_info, raw_exception
from road_distress_agent.tools.locked_lazy import LockedLazy

_DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_RERANKER_MODEL = LockedLazy[Any]()
_DEFAULT_RERANKER = LockedLazy["LocalBgeReranker"]()
_RERANKER_LOCK = Lock()


def _load_reranker_model() -> Any:
    """Load the cross-encoder reranker once per Python process."""
    return _RERANKER_MODEL.get(_create_reranker_model)


def _create_reranker_model() -> Any:
    try:
        from FlagEmbedding import FlagReranker  # noqa: PLC0415

        return FlagReranker(_model_name(), use_fp16=_use_fp16())
    except Exception as exc:
        raise BoundaryError(classify_model_load_error(exc, step="RERANKER_LOAD"), exc) from exc


def _model_name() -> str:
    return os.environ.get("BGE_RERANKER_MODEL") or _DEFAULT_RERANKER_MODEL


def _use_fp16() -> bool:
    raw = os.environ.get("BGE_RERANKER_USE_FP16", "1").strip().lower()
    return raw not in {"0", "false", "no"}


class LocalBgeReranker:
    """Thin wrapper around FlagEmbedding's in-process BGE reranker."""

    def __init__(self, model: Any | None = None) -> None:
        self._model = model or _load_reranker_model()

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        try:
            with _RERANKER_LOCK:
                scores = self._model.compute_score(pairs)
            return _as_float_list(scores, len(passages))
        except Exception as exc:
            raise BoundaryError(_rerank_error_info(exc), exc) from exc


def get_default_bge_reranker() -> LocalBgeReranker:
    return _DEFAULT_RERANKER.get(LocalBgeReranker)


def _as_float_list(scores: Any, expected_len: int) -> list[float]:
    if expected_len == 1 and isinstance(scores, int | float):
        return [float(scores)]
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    values = [float(score) for score in scores]
    if len(values) != expected_len:
        raise ValueError(f"BGE reranker returned {len(values)} scores for {expected_len} passages.")
    return values


def _rerank_error_info(exc: Exception):
    return make_error_info(
        domain="EMBED",
        step="RERANK",
        category=ErrorCategory.INTERNAL,
        responsibility="本地重排模型打分失败",
        reason=raw_exception(exc),
        hint="查看 raw，确认 reranker 模型、输入 passage 和运行环境。",
        raw=raw_exception(exc),
        retriable=False,
    )
