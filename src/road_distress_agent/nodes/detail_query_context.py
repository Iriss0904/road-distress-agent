"""Deterministic context for construction-detail retrieval queries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from road_distress_agent.method_name_catalog import acceptance_scope, method_search_terms
from road_distress_agent.nodes.method_retrieval_context import method_context_from_state

_UNNAMED_METHOD_SUFFIX = "专属维修流程"


def distress_description(state: Mapping[str, Any]) -> str:
    return method_context_from_state(state).facts_text()


def detail_queries(method: str, distress: str) -> tuple[str, str]:
    return _procedure_query(method, distress), _acceptance_query(method, distress)


def _procedure_query(method: str, distress: str) -> str:
    context = f"；病害上下文：{distress}" if distress else ""
    if _is_unnamed_method(method):
        return f"病害「{distress}」的完整维修操作流程和施工步骤"
    return f"施工方法「{_method_terms(method)}」的完整操作流程和施工步骤{context}"


def _acceptance_query(method: str, distress: str) -> str:
    criteria = "质量检查与验收：检查项目、检查频度、规定值或允许偏差、检验方法"
    if _is_unnamed_method(method):
        return f"{distress}的养护{criteria}"
    context = f"；病害上下文：{distress}" if distress else ""
    scope = acceptance_scope(method)
    if scope:
        return f"{scope}{criteria}；适用方法：{_method_terms(method)}{context}"
    return f"施工方法「{_method_terms(method)}」的{criteria}{context}"


def _method_terms(method: str) -> str:
    canonical, *aliases = method_search_terms(method)
    if not aliases:
        return canonical
    return f"{canonical}；规范表述：{'、'.join(aliases)}"


def _is_unnamed_method(method: str) -> bool:
    return method.endswith(_UNNAMED_METHOD_SUFFIX)
