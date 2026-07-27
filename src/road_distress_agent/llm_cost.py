"""Pure conversion of cache-aware provider usage into USD cost."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

PRICE_KEY = "prices_usd_per_million_tokens"


def calculate_usage_cost_usd(
    usage: Mapping[str, Any],
    pricing: Mapping[str, Any],
) -> Decimal:
    """Calculate billable USD cost without using ``total_tokens``."""
    _validate_attribution(usage, pricing)
    prompt = _token(usage, "prompt_tokens")
    cache_hit = _token(usage, "prompt_cache_hit_tokens")
    cache_miss = _token(usage, "prompt_cache_miss_tokens")
    completion = _token(usage, "completion_tokens")
    if cache_hit + cache_miss != prompt:
        raise ValueError(
            "Billable cache input tokens must equal prompt_tokens: "
            f"{cache_hit} + {cache_miss} != {prompt}."
        )
    rates = _mapping(pricing, PRICE_KEY)
    unit_tokens = _positive_integer(pricing, "unit_tokens")
    costs = (
        _segment_cost(cache_hit, rates, "input_cache_hit", unit_tokens=unit_tokens),
        _segment_cost(cache_miss, rates, "input_cache_miss", unit_tokens=unit_tokens),
        _segment_cost(completion, rates, "output", unit_tokens=unit_tokens),
    )
    return sum(costs, start=Decimal("0"))


def _validate_attribution(
    usage: Mapping[str, Any],
    pricing: Mapping[str, Any],
) -> None:
    for key in ("provider", "model"):
        usage_value = _nonempty_text(usage, key)
        pricing_value = _nonempty_text(pricing, key)
        if usage_value != pricing_value:
            raise ValueError(
                f"Usage {key} {usage_value!r} does not match pricing {key} {pricing_value!r}."
            )


def _segment_cost(
    tokens: int,
    rates: Mapping[str, Any],
    rate_key: str,
    *,
    unit_tokens: int,
) -> Decimal:
    rate = _decimal(rates, rate_key)
    return Decimal(tokens) * rate / Decimal(unit_tokens)


def _token(values: Mapping[str, Any], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer, got {value!r}.")
    return value


def _positive_integer(values: Mapping[str, Any], key: str) -> int:
    value = _token(values, key)
    if value == 0:
        raise ValueError(f"{key} must be positive.")
    return value


def _decimal(values: Mapping[str, Any], key: str) -> Decimal:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{key} must be a non-negative decimal, got {value!r}.")
    parsed = Decimal(str(value))
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"{key} must be a non-negative finite decimal, got {value!r}.")
    return parsed


def _mapping(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _nonempty_text(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string, got {value!r}.")
    return value.strip()
