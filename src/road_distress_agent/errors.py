"""Structured runtime boundary errors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    CONFIG_MISSING = "config_missing"
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    BAD_REQUEST = "bad_request"
    UPSTREAM_5XX = "upstream_5xx"
    NOT_FOUND = "not_found"
    PARSE = "parse"
    MODEL_LOAD = "model_load"
    DB = "db"
    NOT_IMPLEMENTED = "not_implemented"
    INTERNAL = "internal"


RETRIABLE_CATEGORIES = frozenset(
    {
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.TIMEOUT,
        ErrorCategory.CONNECTION,
        ErrorCategory.UPSTREAM_5XX,
    }
)


@dataclass(frozen=True)
class ErrorInfo:
    category: ErrorCategory
    code: str
    step: str
    message: str
    hint: str
    raw: str
    retriable: bool


class BoundaryError(RuntimeError):
    """Exception carrying user-visible classification and raw debug detail."""

    def __init__(self, info: ErrorInfo, cause: Exception | None = None) -> None:
        super().__init__(info.message)
        self.info = info
        if cause is not None:
            self.__cause__ = cause


def make_error_info(
    *,
    domain: str,
    step: str,
    category: ErrorCategory,
    responsibility: str,
    reason: str,
    hint: str,
    raw: str | None = None,
    retriable: bool | None = None,
) -> ErrorInfo:
    active_retriable = category in RETRIABLE_CATEGORIES if retriable is None else retriable
    code = f"{domain.upper()}.{_step_token(step)}.{category.value.upper()}"
    return ErrorInfo(
        category=category,
        code=code,
        step=step,
        message=f"[{code}] {responsibility} — {reason} — {hint}",
        hint=hint,
        raw=raw or reason,
        retriable=active_retriable,
    )


def ensure_boundary_error(
    exc: Exception,
    *,
    default_domain: str,
    default_step: str,
) -> BoundaryError:
    if isinstance(exc, BoundaryError):
        return exc
    info = make_error_info(
        domain=default_domain,
        step=default_step,
        category=ErrorCategory.INTERNAL,
        responsibility=f"{default_step} 执行失败",
        reason=raw_exception(exc),
        hint="查看运行日志中的 raw 字段定位代码缺陷。",
        raw=raw_exception(exc),
        retriable=False,
    )
    return BoundaryError(info, exc)


def error_info_payload(info: ErrorInfo) -> dict[str, Any]:
    return {
        "message": info.message,
        "category": info.category.value,
        "code": info.code,
        "step": info.step,
        "retriable": info.retriable,
        "hint": info.hint,
        "raw": info.raw,
    }


def raw_exception(exc: Exception) -> str:
    text = str(exc)
    return f"{exc.__class__.__name__}: {text}" if text else exc.__class__.__name__


def status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return code
    direct = getattr(exc, "status_code", None)
    return int(direct) if isinstance(direct, int) else None


def looks_timeout(exc: Exception) -> bool:
    text = raw_exception(exc).lower()
    return isinstance(exc, TimeoutError) or "timeout" in text or "timed out" in text


def _step_token(step: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in step).strip("_")
