"""Canonical road-maintenance method terminology approved by the domain model."""

from __future__ import annotations

from types import MappingProxyType

CANONICAL_METHOD_NAMES = frozenset(
    {
        "薄层沥青混合料加铺",
        "无槽贴封式灌缝施工",
        "开槽贴封式灌缝施工",
        "清缝并灌缝",
        "扩缝并灌缝",
        "路面贴缝胶施工",
    }
)

METHOD_NAME_ALIASES = MappingProxyType(
    {
        "路面加热型密封胶无槽贴封式施工": "无槽贴封式灌缝施工",
        "路面加热型密封胶无槽贴封式进行施工": "无槽贴封式灌缝施工",
        "路面加热型密封胶开槽贴封式施工": "开槽贴封式灌缝施工",
        "路面加热型密封胶开槽贴封式进行施工": "开槽贴封式灌缝施工",
        "路面裂缝贴缝胶施工": "路面贴缝胶施工",
    }
)
_ACCEPTANCE_SCOPES = MappingProxyType(
    {
        "无槽贴封式灌缝施工": "路面加热型密封胶施工",
        "开槽贴封式灌缝施工": "路面加热型密封胶施工",
        "路面贴缝胶施工": "路面裂缝贴缝胶施工",
    }
)


def canonical_method_name(value: str) -> str:
    name = value.strip()
    return METHOD_NAME_ALIASES.get(name, name)


def method_search_terms(value: str) -> tuple[str, ...]:
    canonical = canonical_method_name(value)
    aliases = tuple(source for source, target in METHOD_NAME_ALIASES.items() if target == canonical)
    return (canonical, *aliases)


def acceptance_scope(value: str) -> str | None:
    return _ACCEPTANCE_SCOPES.get(canonical_method_name(value))
