"""Small helpers for LLM prompt payload construction."""

from __future__ import annotations

from typing import Any


def plain_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: plain_model(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain_model(item) for item in value]
    return value


def recent_user_messages(messages: list[Any], limit: int) -> list[str]:
    texts = [_message_text(item) for item in messages]
    return [text for text in texts if text][-limit:]


def _message_text(message: Any) -> str | None:
    content = getattr(message, "content", None)
    message_type = getattr(message, "type", None)
    if isinstance(message, dict):
        content = message.get("content", content)
        message_type = message.get("type", message_type)
    if message_type not in {"human", "user"} or content is None:
        return None
    return content if isinstance(content, str) else str(content)
