"""VLM-specific boundary error classification."""

from __future__ import annotations

from road_distress_agent.error_classifiers import classify_llm_error
from road_distress_agent.errors import ErrorCategory, ErrorInfo, make_error_info, raw_exception


def classify_vlm_error(
    exc: Exception,
    *,
    step: str,
    timeout_seconds: int | None = None,
) -> ErrorInfo:
    if isinstance(exc, FileNotFoundError):
        return _info(exc, step, ErrorCategory.NOT_FOUND, "确认图片已上传且路径仍然存在。")
    if isinstance(exc, ValueError):
        return _info(exc, step, ErrorCategory.BAD_REQUEST, "压缩图片或检查图片格式后重试。")
    if isinstance(exc, RuntimeError) and "DASHSCOPE_API_KEY" in str(exc):
        return _info(
            exc,
            step,
            ErrorCategory.CONFIG_MISSING,
            "在 .env 设置 DASHSCOPE_API_KEY 后重试。",
        )
    llm_info = classify_llm_error(exc, node_name=step, timeout_seconds=timeout_seconds)
    return make_error_info(
        domain="VLM",
        step=step,
        category=llm_info.category,
        responsibility="图像识别失败",
        reason=llm_info.raw,
        hint=llm_info.hint,
        raw=llm_info.raw,
        retriable=llm_info.retriable,
    )


def classify_vlm_parse_error(raw: object, *, step: str = "PARSE") -> ErrorInfo:
    return make_error_info(
        domain="VLM",
        step=step,
        category=ErrorCategory.PARSE,
        responsibility="图像识别失败",
        reason="VLM 返回内容为空",
        hint="查看 raw，确认 VLM 返回结构和模型输出。",
        raw=str(raw),
        retriable=False,
    )


def _info(exc: Exception, step: str, category: ErrorCategory, hint: str) -> ErrorInfo:
    reason = (
        "未配置 DASHSCOPE_API_KEY"
        if category == ErrorCategory.CONFIG_MISSING
        else raw_exception(exc)
    )
    return make_error_info(
        domain="VLM",
        step=step,
        category=category,
        responsibility="图像识别失败",
        reason=reason,
        hint=hint,
        raw=raw_exception(exc),
        retriable=False,
    )
