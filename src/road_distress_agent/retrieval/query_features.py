"""Query feature detection for channel-aware retrieval."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SHORT_LOOKUP_MAX_CHARS = 24
MARKED_CLAUSE_ID_RE = re.compile(
    r"(?<![\d.])(?:第\s*(\d+(?:\.\d+){1,5})(?:\s*(?:条|节|款))?|"
    r"(\d+(?:\.\d+){1,5})\s*(?:条|节|款))(?![\d.a-zA-Z%℃])"
)
BARE_CLAUSE_ID_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+){2,5})(?![\d.a-zA-Z%℃])")
NUMERIC_CONDITION_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:mm|cm|m|%|℃|度)")
STANDARD_REF_RE = re.compile(r"\b[A-Z]{2,}\s*[A-Z]?\s*\d+(?:[-—]\d+)?\b")
QUOTED_PHRASE_RE = re.compile(r"[“\"']([^”\"']{2,})[”\"']")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
EXACT_TERM_RE = re.compile(
    r"(条款|章节|标准号|原文|表格|表\d|允许偏差|压实度|饱满度|检验|验收|"
    r"适用条件|适用范围|施工方法|施工步骤|维修流程|质量控制|冷补料|灌缝|"
    r"反射裂缝|网裂|块裂|龟裂|唧浆|坑槽|露基层|基层外露|裂缝)"
)
KB_RETRIEVAL_NODE_NAMES = frozenset({"kb_retriever", "kb_hop_retriever"})
DIAGNOSTIC_DEFAULT_BM25_NODE_NAMES = frozenset({"detail_retriever_v2"})
DEFAULT_BM25_NODE_NAMES = KB_RETRIEVAL_NODE_NAMES | DIAGNOSTIC_DEFAULT_BM25_NODE_NAMES


@dataclass(frozen=True)
class QueryFeatures:
    clause_ids: tuple[str, ...]
    has_standard_ref: bool
    has_exact_term: bool
    has_numeric_condition: bool
    is_short_lookup: bool

    @property
    def needs_clause_lookup(self) -> bool:
        return bool(self.clause_ids)

    @property
    def has_exact_signal(self) -> bool:
        return (
            self.needs_clause_lookup
            or self.has_standard_ref
            or self.has_exact_term
            or self.has_numeric_condition
            or self.is_short_lookup
        )


def detect_query_features(query: str, filters: Mapping[str, Any]) -> QueryFeatures:
    normalized = unicodedata.normalize("NFKC", query)
    clause_ids = _unique([*_filter_clause_ids(filters), *_query_clause_ids(normalized)])
    return QueryFeatures(
        clause_ids=tuple(clause_ids),
        has_standard_ref=bool(STANDARD_REF_RE.search(normalized.upper())),
        has_exact_term=bool(EXACT_TERM_RE.search(normalized)),
        has_numeric_condition=bool(NUMERIC_CONDITION_RE.search(normalized.lower())),
        is_short_lookup=_is_short_lookup(normalized),
    )


def should_run_bm25(features: QueryFeatures, node_name: str) -> bool:
    return node_name in DEFAULT_BM25_NODE_NAMES or features.has_exact_signal


def _filter_clause_ids(filters: Mapping[str, Any]) -> list[str]:
    raw = filters.get("clause_id")
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(item) for item in raw if str(item).strip()]
    return [str(raw)]


def _query_clause_ids(query: str) -> list[str]:
    marked = [match.group(1) or match.group(2) for match in MARKED_CLAUSE_ID_RE.finditer(query)]
    bare = [match.group(1) for match in BARE_CLAUSE_ID_RE.finditer(query)]
    return _unique([*marked, *bare])


def _is_short_lookup(query: str) -> bool:
    compact = re.sub(r"\s+", "", query)
    if len(compact) > SHORT_LOOKUP_MAX_CHARS:
        return False
    return bool(CHINESE_RE.search(compact))


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        item = value.strip()
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique
