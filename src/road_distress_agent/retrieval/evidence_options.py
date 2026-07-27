"""Shared retrieval option types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

DEFAULT_RECALL_TOP_K = 20
DEFAULT_LOCAL_TOP_K = 10
DEFAULT_FINAL_TOP_K = 3
DEFAULT_PROTECTED_EXACT_TOP_K = 2


@dataclass(frozen=True)
class EvidenceRetrievalOptions:
    node_name: str
    title: str
    query: str
    stage_goal: str
    explicit_clause_ids: tuple[str, ...] = ()
    filters: Mapping[str, Any] = field(default_factory=dict)
    recall_top_k: int = DEFAULT_RECALL_TOP_K
    bm25_top_k: int = DEFAULT_RECALL_TOP_K
    local_top_k: int = DEFAULT_LOCAL_TOP_K
    final_top_k: int = DEFAULT_FINAL_TOP_K
    protected_exact_top_k: int = DEFAULT_PROTECTED_EXACT_TOP_K
    comparison_query: bool = False
    preserve_semantic_role_filter: bool = False
    split_numbered_rerank_passages: bool = False
