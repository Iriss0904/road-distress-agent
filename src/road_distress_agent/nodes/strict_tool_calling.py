"""Strict DeepSeek tool-calling helpers for structured node outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

from pydantic import BaseModel, ValidationError

from road_distress_agent.error_classifiers import classify_parse_error
from road_distress_agent.errors import BoundaryError
from road_distress_agent.llm_deepseek import deepseek_beta_api_base, deepseek_chat
from road_distress_agent.llm_runtime import invoke_llm_call, llm_timeout_seconds
from road_distress_agent.tracing import to_plain


@dataclass(frozen=True)
class StrictToolResult:
    tool_name: str
    parsed: BaseModel
    raw_tool_call: dict[str, Any]


def invoke_strict_tool_call(
    *,
    node_name: str,
    messages: list[Any],
    tools: Sequence[type[BaseModel]],
) -> StrictToolResult:
    timeout = llm_timeout_seconds(node_name)
    llm = deepseek_chat(timeout=timeout, api_base=deepseek_beta_api_base())
    runnable = llm.bind_tools(
        list(tools),
        tool_choice=_tool_choice(tools),
        strict=True,
        parallel_tool_calls=False,
    )
    return invoke_llm_call(
        error_context_name=node_name,
        usage_correlation_name=node_name,
        timeout_seconds=timeout,
        call=lambda: parse_strict_tool_response(
            runnable.invoke(messages),
            tools=tools,
            node_name=node_name,
        ),
    )


def parse_strict_tool_response(
    response: Any,
    *,
    tools: Sequence[type[BaseModel]],
    node_name: str,
) -> StrictToolResult:
    invalid = list(getattr(response, "invalid_tool_calls", None) or [])
    if invalid:
        _raise_parse(node_name, "Strict tool response contains invalid tool calls", invalid)
    tool_calls = list(getattr(response, "tool_calls", None) or [])
    if len(tool_calls) != 1:
        _raise_parse(
            node_name,
            "Strict tool response must contain exactly one tool call",
            tool_calls,
        )
    raw_call = dict(tool_calls[0])
    tool_name = _tool_name(raw_call)
    schema = _tool_map(tools).get(tool_name)
    if schema is None:
        _raise_parse(node_name, f"Unknown strict tool call: {tool_name}", raw_call)
    return StrictToolResult(
        tool_name=tool_name,
        parsed=_validate_args(schema, _tool_args(raw_call), node_name, raw_call),
        raw_tool_call=raw_call,
    )


def strict_tool_trace_payload(result: StrictToolResult) -> dict[str, Any]:
    return {
        "tool_name": result.tool_name,
        "arguments": to_plain(result.parsed),
        "raw_tool_call": to_plain(result.raw_tool_call),
    }


def _tool_choice(tools: Sequence[type[BaseModel]]) -> str:
    if len(tools) == 1:
        return tools[0].__name__
    return "required"


def _tool_map(tools: Sequence[type[BaseModel]]) -> dict[str, type[BaseModel]]:
    return {tool.__name__: tool for tool in tools}


def _tool_name(raw_call: dict[str, Any]) -> str:
    value = raw_call.get("name")
    if not isinstance(value, str) or not value:
        return "<missing>"
    return value


def _tool_args(raw_call: dict[str, Any]) -> dict[str, Any]:
    args = raw_call.get("args")
    return args if isinstance(args, dict) else {}


def _validate_args(
    schema: type[BaseModel],
    args: dict[str, Any],
    node_name: str,
    raw_call: dict[str, Any],
) -> BaseModel:
    try:
        return schema.model_validate(args)
    except ValidationError:
        _raise_parse(node_name, f"Strict tool args failed {schema.__name__} validation", raw_call)


def _raise_parse(node_name: str, message: str, raw: Any) -> NoReturn:
    exc = ValueError(f"{message}: {to_plain(raw)}")
    raise BoundaryError(classify_parse_error(exc, node_name=node_name, mode="schema"), exc)
