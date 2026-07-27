"""Boundary-specific exception classifiers."""

from __future__ import annotations

import sqlite3

from road_distress_agent.errors import (
    BoundaryError,
    ErrorCategory,
    ErrorInfo,
    looks_timeout,
    make_error_info,
    raw_exception,
    status_code,
)


def classify_llm_error(
    exc: Exception,
    *,
    node_name: str,
    timeout_seconds: int | None = None,
) -> ErrorInfo:
    if isinstance(exc, BoundaryError):
        return exc.info
    if _looks_parse_failure(exc):
        return classify_parse_error(exc, node_name=node_name, mode="schema")
    category = _llm_category(exc)
    return make_error_info(
        domain="LLM",
        step=node_name,
        category=category,
        responsibility=f"{node_name} LLM 调用失败",
        reason=_llm_reason(exc, category, timeout_seconds),
        hint=_llm_hint(category),
        raw=raw_exception(exc),
    )


def classify_parse_error(exc: Exception, *, node_name: str, mode: str) -> ErrorInfo:
    return make_error_info(
        domain="PARSE",
        step=node_name,
        category=ErrorCategory.PARSE,
        responsibility=f"{node_name} 输出解析失败",
        reason=_parse_reason(mode),
        hint="查看 raw，对照该节点的 Pydantic schema 和 prompt。",
        raw=_parse_raw(exc),
        retriable=False,
    )


def classify_http_error(
    exc: Exception,
    *,
    domain: str,
    step: str,
    responsibility: str,
    service: str,
    url: str | None = None,
) -> ErrorInfo:
    category = _http_category(exc)
    return make_error_info(
        domain=domain,
        step=step,
        category=category,
        responsibility=responsibility,
        reason=_http_reason(exc, service, url),
        hint=_http_hint(category, service),
        raw=raw_exception(exc),
    )


def classify_model_load_error(exc: Exception, *, step: str = "MODEL_LOAD") -> ErrorInfo:
    category = (
        ErrorCategory.CONFIG_MISSING if isinstance(exc, ImportError) else ErrorCategory.MODEL_LOAD
    )
    code_step = "DEPENDENCY" if category == ErrorCategory.CONFIG_MISSING else step
    return make_error_info(
        domain="EMBED",
        step=code_step,
        category=category,
        responsibility="嵌入模型加载失败",
        reason=raw_exception(exc),
        hint=_embed_hint(category),
        raw=raw_exception(exc),
        retriable=False,
    )


def classify_db_error(exc: Exception, *, step: str) -> ErrorInfo:
    raw = raw_exception(exc).lower()
    return make_error_info(
        domain="DB",
        step=step,
        category=ErrorCategory.DB,
        responsibility="会话状态读写失败",
        reason=_db_reason(exc, raw),
        hint=_db_hint(raw),
        raw=raw_exception(exc),
    )


def classify_delivery_error(exc: Exception, *, step: str) -> ErrorInfo:
    if isinstance(exc, FileNotFoundError):
        category = ErrorCategory.NOT_FOUND
    elif isinstance(exc, PermissionError | OSError):
        category = ErrorCategory.DB
    else:
        category = ErrorCategory.INTERNAL
    return make_error_info(
        domain="DELIVERY",
        step=step,
        category=category,
        responsibility="交付物生成失败",
        reason=raw_exception(exc),
        hint="检查模板、输出目录权限、磁盘空间和交付输入字段。",
        raw=raw_exception(exc),
        retriable=False,
    )


def not_implemented_error(*, domain: str, step: str, responsibility: str) -> BoundaryError:
    info = make_error_info(
        domain=domain,
        step=step,
        category=ErrorCategory.NOT_IMPLEMENTED,
        responsibility=responsibility,
        reason="该工具尚未接入 live 实现",
        hint="保持 dry-run，或接入真实 MCP client 后再关闭 dry-run。",
        retriable=False,
    )
    return BoundaryError(info)


def _llm_category(exc: Exception) -> ErrorCategory:
    code = status_code(exc)
    name = exc.__class__.__name__
    if isinstance(exc, KeyError) and "DEEPSEEK_API_KEY" in str(exc):
        return ErrorCategory.CONFIG_MISSING
    if looks_timeout(exc) or name == "APITimeoutError":
        return ErrorCategory.TIMEOUT
    if code in {401, 403} or name in {"AuthenticationError", "PermissionDeniedError"}:
        return ErrorCategory.AUTH
    if code == 429 or name == "RateLimitError":
        return ErrorCategory.RATE_LIMIT
    if code == 400 or name in {"BadRequestError", "UnprocessableEntityError"}:
        return ErrorCategory.BAD_REQUEST
    if code == 404 or name == "NotFoundError":
        return ErrorCategory.NOT_FOUND
    if code is not None and code >= 500:
        return ErrorCategory.UPSTREAM_5XX
    if name == "APIConnectionError" or "connection" in raw_exception(exc).lower():
        return ErrorCategory.CONNECTION
    return ErrorCategory.INTERNAL


def _looks_parse_failure(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    text = raw_exception(exc).lower()
    markers = ("validationerror", "outputparser", "jsondecode", "parse")
    return any(marker in name or marker in text for marker in markers)


def _parse_raw(exc: Exception) -> str:
    current: BaseException | None = exc
    while current is not None:
        llm_output = getattr(current, "llm_output", None)
        if llm_output:
            return str(llm_output)
        current = getattr(current, "__cause__", None)
    return raw_exception(exc)


def _llm_reason(exc: Exception, category: ErrorCategory, timeout: int | None) -> str:
    if category == ErrorCategory.CONFIG_MISSING:
        return "未配置 DEEPSEEK_API_KEY"
    if category == ErrorCategory.TIMEOUT and timeout:
        return f"{timeout}s 未返回"
    code = status_code(exc)
    return f"HTTP {code}" if code else raw_exception(exc)


def _llm_hint(category: ErrorCategory) -> str:
    return {
        ErrorCategory.CONFIG_MISSING: "在 .env 设置 DEEPSEEK_API_KEY 后重试。",
        ErrorCategory.AUTH: "检查 DEEPSEEK_API_KEY 是否有效、欠费或权限不足。",
        ErrorCategory.RATE_LIMIT: "稍后重试或降低并发。",
        ErrorCategory.TIMEOUT: "检查网络或提高该节点 LLM timeout 后重试。",
        ErrorCategory.CONNECTION: "检查网络和 DEEPSEEK_API_BASE。",
        ErrorCategory.BAD_REQUEST: "检查模型参数、上下文长度和结构化输出设置。",
        ErrorCategory.NOT_FOUND: "检查 DEEPSEEK_MODEL 是否存在。",
        ErrorCategory.UPSTREAM_5XX: "DeepSeek 上游故障，稍后重试。",
    }.get(category, "查看 raw 定位未分类 LLM 异常。")


def _parse_reason(mode: str) -> str:
    return {
        "empty": "模型未产出有效内容",
        "schema": "JSON 合法但缺字段或类型不符",
        "format": "模型未返回可解析 JSON 对象",
    }.get(mode, "结构化输出无法解析")


def _http_category(exc: Exception) -> ErrorCategory:
    code = status_code(exc)
    name = exc.__class__.__name__
    if looks_timeout(exc) or name.endswith("Timeout"):
        return ErrorCategory.TIMEOUT
    if name in {"ConnectionError", "ConnectError"}:
        return ErrorCategory.CONNECTION
    if code in {401, 403}:
        return ErrorCategory.AUTH
    if code == 429:
        return ErrorCategory.RATE_LIMIT
    if code == 400:
        return ErrorCategory.BAD_REQUEST
    if code == 404:
        return ErrorCategory.NOT_FOUND
    if code is not None and code >= 500:
        return ErrorCategory.UPSTREAM_5XX
    if name in {"JSONDecodeError", "InvalidJSONError"}:
        return ErrorCategory.PARSE
    return ErrorCategory.INTERNAL


def _http_reason(exc: Exception, service: str, url: str | None) -> str:
    code = status_code(exc)
    target = f"{service}({url})" if url else service
    if code:
        return f"{target} HTTP {code}"
    if looks_timeout(exc):
        return f"{target} 请求超时"
    if _http_category(exc) == ErrorCategory.CONNECTION:
        return f"无法连接 {target}"
    if _http_category(exc) == ErrorCategory.PARSE:
        return f"{target} 返回内容不是预期 JSON"
    return raw_exception(exc)


def _http_hint(category: ErrorCategory, service: str) -> str:
    return {
        ErrorCategory.AUTH: f"检查 {service} token/key 是否有效、欠费或权限不足。",
        ErrorCategory.RATE_LIMIT: f"{service} 触发限流，稍后重试或升级配额。",
        ErrorCategory.TIMEOUT: f"检查网络，或提高 {service} timeout 后重试。",
        ErrorCategory.CONNECTION: f"检查网络、代理和 {service} 服务地址。",
        ErrorCategory.BAD_REQUEST: "检查请求参数是否合法。",
        ErrorCategory.NOT_FOUND: "确认输入位置/资源名称是否正确。",
        ErrorCategory.PARSE: "查看 raw，可能是接口改版或返回 HTML 错误页。",
        ErrorCategory.UPSTREAM_5XX: f"{service} 上游故障，稍后重试。",
    }.get(category, "查看 raw 定位未分类 HTTP 异常。")


def _embed_hint(category: ErrorCategory) -> str:
    if category == ErrorCategory.CONFIG_MISSING:
        return "安装 FlagEmbedding/torch，或确认当前 Python 环境依赖完整。"
    return "预先下载 BGE 模型、恢复网络、释放显存/内存或关闭 fp16。"


def _db_reason(exc: Exception, raw: str) -> str:
    if "locked" in raw:
        return "SQLite database is locked"
    if "malformed" in raw or "corrupt" in raw:
        return "数据库镜像损坏或 checkpoint 反序列化失败"
    if isinstance(exc, sqlite3.OperationalError):
        return f"SQLite operational error: {exc}"
    return raw_exception(exc)


def _db_hint(raw: str) -> str:
    if "locked" in raw:
        return "检查是否有另一进程占用 .web-workflow.db。"
    if "malformed" in raw or "corrupt" in raw:
        return "备份后修复或重建项目根目录 .web-workflow.db。"
    if "readonly" in raw or "permission" in raw or "disk" in raw:
        return "检查磁盘空间、目录权限和 ROAD_DISTRESS_WEB_DB 路径。"
    return "查看 SQLite 原始错误和 checkpoint 版本兼容性。"
