"""Qdrant-specific boundary error classification."""

from __future__ import annotations

from road_distress_agent.errors import (
    ErrorCategory,
    ErrorInfo,
    looks_timeout,
    make_error_info,
    raw_exception,
    status_code,
)


def classify_qdrant_error(
    exc: Exception,
    *,
    step: str,
    url: str | None,
    collection: str,
) -> ErrorInfo:
    category = _qdrant_category(exc)
    return make_error_info(
        domain="QDRANT",
        step=step,
        category=category,
        responsibility="知识库检索失败",
        reason=_qdrant_reason(exc, url, collection),
        hint=_qdrant_hint(category, collection),
        raw=raw_exception(exc),
    )


def _qdrant_category(exc: Exception) -> ErrorCategory:
    code = status_code(exc)
    name = exc.__class__.__name__
    text = raw_exception(exc).lower()
    if looks_timeout(exc):
        return ErrorCategory.TIMEOUT
    if code in {401, 403}:
        return ErrorCategory.AUTH
    if code == 404:
        return ErrorCategory.NOT_FOUND
    if code == 400:
        return ErrorCategory.BAD_REQUEST
    if code is not None and code >= 500:
        return ErrorCategory.UPSTREAM_5XX
    if name == "ResponseHandlingException":
        return _qdrant_response_handling_category(exc)
    if "connection" in text or "connect" in text or "refused" in text:
        return ErrorCategory.CONNECTION
    return ErrorCategory.INTERNAL


def _qdrant_response_handling_category(exc: Exception) -> ErrorCategory:
    source = getattr(exc, "source", None)
    if not isinstance(source, Exception):
        return ErrorCategory.INTERNAL
    if looks_timeout(source):
        return ErrorCategory.TIMEOUT
    text = raw_exception(source).lower()
    if "connect" in text or "connection" in text or "refused" in text:
        return ErrorCategory.CONNECTION
    return ErrorCategory.INTERNAL


def _qdrant_reason(exc: Exception, url: str | None, collection: str) -> str:
    code = status_code(exc)
    if code == 404:
        return f'集合 "{collection}" 不存在'
    if code == 400:
        return "向量维度、过滤字段或查询参数不匹配"
    if code:
        return f"Qdrant HTTP {code}"
    category = _qdrant_category(exc)
    if category == ErrorCategory.CONNECTION:
        return f"无法连接 Qdrant({url})"
    if category == ErrorCategory.TIMEOUT:
        return "Qdrant 查询超时"
    return raw_exception(exc)


def _qdrant_hint(category: ErrorCategory, collection: str) -> str:
    return {
        ErrorCategory.CONNECTION: "确认 Qdrant 已启动，例如 docker compose up qdrant。",
        ErrorCategory.NOT_FOUND: f'先运行 build_qdrant_index 建立集合 "{collection}"。',
        ErrorCategory.TIMEOUT: "检查数据量、服务负载和 Qdrant timeout。",
        ErrorCategory.BAD_REQUEST: "检查索引向量维度、payload 字段和编码器是否同源。",
        ErrorCategory.AUTH: "检查 QDRANT_API_KEY 或云端访问权限。",
        ErrorCategory.UPSTREAM_5XX: "Qdrant 服务内部故障，查看服务日志。",
    }.get(category, "查看 raw 定位未分类 Qdrant 异常。")
