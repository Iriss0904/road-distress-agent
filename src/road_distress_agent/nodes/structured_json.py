"""Strict JSON-object parsing for LLM nodes."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, ValidationError

from road_distress_agent.error_classifiers import classify_parse_error
from road_distress_agent.errors import BoundaryError

T = TypeVar("T", bound=BaseModel)


def invoke_json_model(
    llm: ChatDeepSeek,
    messages: list[Any],
    schema: type[T],
    node_name: str | None = None,
) -> T:
    """Invoke an LLM in JSON mode and validate the returned object."""
    active_node = node_name or schema.__name__
    response = llm.invoke(messages)
    raw = _message_text(response)
    if not raw.strip():
        exc = ValueError("LLM output was empty.")
        raise BoundaryError(classify_parse_error(exc, node_name=active_node, mode="empty"), exc)
    try:
        data = _loads_json_object(raw)
    except Exception as exc:
        info = classify_parse_error(exc, node_name=active_node, mode="format")
        raise BoundaryError(info, exc) from exc
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        excerpt = raw[:1000]
        wrapped = ValueError(f"LLM JSON failed {schema.__name__} validation: {excerpt}")
        raise BoundaryError(
            classify_parse_error(wrapped, node_name=active_node, mode="schema"),
            exc,
        ) from exc


def json_mode_kwargs() -> dict[str, Any]:
    return {"model_kwargs": json_mode_model_kwargs()}


def json_mode_model_kwargs() -> dict[str, Any]:
    return {"response_format": {"type": "json_object"}}


def _message_text(response: Any) -> str:
    content = response.content if isinstance(response, BaseMessage) else response
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_block_text(block) for block in content)
    raise ValueError(f"LLM response content must be text, got {type(content).__name__}.")


def _content_block_text(block: Any) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        value = block.get("text") or block.get("content") or ""
        return str(value)
    return str(block)


def _loads_json_object(raw: str) -> dict[str, Any]:
    text = _strip_fence(raw.strip())
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = json.loads(_first_json_object(text))
    if not isinstance(value, dict):
        raise ValueError("LLM JSON output must be an object.")
    return value


def _strip_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    stripped = text.strip("`").strip()
    if stripped.lower().startswith("json"):
        return stripped[4:].strip()
    return stripped


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("LLM output did not contain a JSON object.")
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = in_string
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("LLM output contained an unterminated JSON object.")
