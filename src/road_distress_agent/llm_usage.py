"""Provider usage extraction for LLM runtime traces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

CACHE_FIELDS_ABSENT_REASON = (
    "Provider token_usage did not include prompt_cache_hit_tokens or prompt_cache_miss_tokens."
)


def llm_usage_record(
    response: LLMResult,
    trace_correlation_id: str,
) -> dict[str, Any] | None:
    generation = _first_generation(response)
    message = generation.message if generation is not None else None
    if not isinstance(message, AIMessage):
        return None
    response_metadata = _mapping(message.response_metadata)
    llm_output = _mapping(response.llm_output)
    counts = usage_counts(response_metadata, llm_output, message.usage_metadata)
    if counts is None:
        return None
    return {
        "provider": _text_or_none(
            response_metadata.get("model_provider") or llm_output.get("model_provider")
        ),
        "model": _text_or_none(response_metadata.get("model_name") or llm_output.get("model_name")),
        "prompt_tokens": counts["prompt_tokens"],
        "completion_tokens": counts["completion_tokens"],
        "total_tokens": counts["total_tokens"],
        "prompt_cache_hit_tokens": counts["prompt_cache_hit_tokens"],
        "prompt_cache_miss_tokens": counts["prompt_cache_miss_tokens"],
        "cache_token_details_status": counts["cache_token_details_status"],
        "cache_token_details_unavailable_reason": counts["cache_token_details_unavailable_reason"],
        "call_id": _text_or_none(response_metadata.get("id") or llm_output.get("id")),
        "trace_correlation_id": trace_correlation_id,
    }


def usage_counts(
    response_metadata: Mapping[str, Any],
    llm_output: Mapping[str, Any],
    usage_metadata: Any,
) -> dict[str, Any] | None:
    token_usage = _mapping(response_metadata.get("token_usage")) or _mapping(
        llm_output.get("token_usage")
    )
    normalized_usage = _mapping(usage_metadata)
    prompt = _token_value(token_usage, "prompt_tokens")
    completion = _token_value(token_usage, "completion_tokens")
    total = _token_value(token_usage, "total_tokens")
    prompt = prompt if prompt is not None else _token_value(normalized_usage, "input_tokens")
    completion = (
        completion if completion is not None else _token_value(normalized_usage, "output_tokens")
    )
    total = total if total is not None else _token_value(normalized_usage, "total_tokens")
    if prompt is None or completion is None or total is None:
        return None
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        **_cache_token_details(token_usage, prompt),
    }


def _cache_token_details(
    token_usage: Mapping[str, Any],
    prompt_tokens: int,
) -> dict[str, Any]:
    hit = _token_value(token_usage, "prompt_cache_hit_tokens")
    miss = _token_value(token_usage, "prompt_cache_miss_tokens")
    if hit is None and miss is None:
        return _unavailable_cache_details(CACHE_FIELDS_ABSENT_REASON)
    if hit is None or miss is None:
        missing = [
            name
            for name, value in (
                ("prompt_cache_hit_tokens", hit),
                ("prompt_cache_miss_tokens", miss),
            )
            if value is None
        ]
        reason = f"Provider token_usage cache fields were incomplete; missing {missing!r}."
        return _unavailable_cache_details(reason)
    if hit + miss != prompt_tokens:
        raise ValueError(
            "Provider cache token attribution does not match prompt_tokens: "
            f"{hit} + {miss} != {prompt_tokens}."
        )
    return {
        "prompt_cache_hit_tokens": hit,
        "prompt_cache_miss_tokens": miss,
        "cache_token_details_status": "reported",
        "cache_token_details_unavailable_reason": None,
    }


def _unavailable_cache_details(reason: str) -> dict[str, Any]:
    return {
        "prompt_cache_hit_tokens": None,
        "prompt_cache_miss_tokens": None,
        "cache_token_details_status": "not_reported",
        "cache_token_details_unavailable_reason": reason,
    }


def _first_generation(response: LLMResult) -> ChatGeneration | None:
    generations = response.generations
    if not generations or not generations[0]:
        return None
    generation = generations[0][0]
    return generation if isinstance(generation, ChatGeneration) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _token_value(data: Mapping[str, Any], key: str) -> int | None:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
