"""Independent BM25 and clause metadata retrieval over Qdrant payloads."""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any

from qdrant_client import QdrantClient

from road_distress_agent.errors import BoundaryError
from road_distress_agent.evidence_anchors import tokenize
from road_distress_agent.qdrant_errors import classify_qdrant_error
from road_distress_agent.state import Citation
from road_distress_agent.tools.bm25_filters import matches_filters
from road_distress_agent.tools.bm25_formatting import (
    empty_index_info,
    with_bm25_score,
    with_clause_score,
)
from road_distress_agent.tools.qdrant_rag import (
    payload_to_citation,
    qdrant_filter_from_filters,
)

_COLLECTION_DEFAULT = "road_distress_documents"
BM25_K1 = 1.5
BM25_B = 0.75
SCROLL_BATCH_SIZE = 256
RANK_START = 1
EMPTY_AVG_DOC_LENGTH = 1.0
MIN_POSITIVE_SCORE = 0.0
NO_RESULTS_LIMIT = 0


@dataclass(frozen=True)
class BM25Document:
    chunk: Citation
    frequencies: Counter[str]
    length: int


@dataclass(frozen=True)
class BM25Index:
    documents: tuple[BM25Document, ...]
    document_frequency: Mapping[str, int]
    average_length: float

    @classmethod
    def build(cls, chunks: Sequence[Citation]) -> BM25Index:
        documents = tuple(_document(chunk) for chunk in chunks)
        document_frequency = _document_frequency(documents)
        average = _average_length(documents)
        return cls(documents, document_frequency, average)

    def search(
        self,
        query: str,
        filters: Mapping[str, Any],
        top_k: int,
    ) -> list[Citation]:
        query_terms = Counter(tokenize(query))
        scored = [
            (_bm25_score(query_terms, doc, self), doc)
            for doc in self.documents
            if matches_filters(doc.chunk, filters)
        ]
        positive = [(score, doc) for score, doc in scored if score > MIN_POSITIVE_SCORE]
        ranked = sorted(positive, key=lambda item: item[0], reverse=True)
        return [
            with_bm25_score(doc.chunk, score, rank)
            for rank, (score, doc) in enumerate(ranked[:top_k], RANK_START)
        ]


class QdrantBM25RetrievalTool:
    """Build a real BM25 index from Qdrant payloads and query it locally."""

    def __init__(
        self,
        *,
        url: str | None = None,
        collection: str | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self._url = url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        self._client = client or QdrantClient(url=self._url)
        self._collection = collection or os.environ.get("QDRANT_COLLECTION", _COLLECTION_DEFAULT)

    @cached_property
    def _chunks(self) -> tuple[Citation, ...]:
        payloads = self._scroll_payloads({})
        return tuple(
            payload_to_citation(payload, score=0.0, rank=rank)
            for rank, payload in enumerate(payloads, RANK_START)
        )

    @cached_property
    def _index(self) -> BM25Index:
        if not self._chunks:
            raise BoundaryError(empty_index_info(self._collection))
        return BM25Index.build(self._chunks)

    def search_bm25(
        self,
        query: str,
        filters: Mapping[str, Any],
        top_k: int,
    ) -> list[Citation]:
        return self._index.search(query, filters, top_k)

    def fetch_by_clause_ids(
        self,
        clause_ids: Sequence[str],
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[Citation]:
        payloads = self._clause_payloads(clause_ids, filters, limit)
        return [
            with_clause_score(payload_to_citation(payload, score=0.0, rank=rank), rank)
            for rank, payload in enumerate(payloads, RANK_START)
        ]

    def _clause_payloads(
        self,
        clause_ids: Sequence[str],
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for clause_id in clause_ids:
            clause_filter = {**dict(filters), "clause_id": clause_id}
            payloads.extend(self._scroll_payloads(clause_filter, limit - len(payloads)))
            if len(payloads) >= limit:
                return payloads[:limit]
        return payloads

    def _scroll_payloads(
        self,
        filters: Mapping[str, Any],
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if limit is not None and limit <= NO_RESULTS_LIMIT:
            return []
        offset: Any | None = None
        payloads: list[dict[str, Any]] = []
        while True:
            try:
                batch, offset = self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=qdrant_filter_from_filters(dict(filters)),
                    limit=_batch_limit(limit, len(payloads)),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                raise self._qdrant_error(exc) from exc
            payloads.extend([record.payload or {} for record in batch])
            if offset is None or _limit_reached(payloads, limit):
                return payloads[:limit] if limit is not None else payloads

    def _qdrant_error(self, exc: Exception) -> BoundaryError:
        if isinstance(exc, BoundaryError):
            return exc
        info = classify_qdrant_error(
            exc,
            step="BM25_INDEX",
            url=self._url,
            collection=self._collection,
        )
        return BoundaryError(info, exc)


@lru_cache(maxsize=1)
def get_default_bm25_retrieval_tool() -> QdrantBM25RetrievalTool:
    return QdrantBM25RetrievalTool()


def _document(chunk: Citation) -> BM25Document:
    tokens = tokenize(_chunk_text(chunk))
    return BM25Document(chunk=chunk, frequencies=Counter(tokens), length=len(tokens))


def _chunk_text(chunk: Citation) -> str:
    annotations = chunk.annotations or {}
    heading = " ".join(chunk.heading_path or [])
    parts = [
        _first_text(chunk.source_doc, chunk.document_id),
        _first_text(chunk.clause_id),
        heading,
        _first_text(chunk.context_prefix),
        _first_text(annotations.get("step_title")),
        _first_text(chunk.text),
    ]
    return "\n".join(part for part in parts if part)


def _first_text(*values: object) -> str:
    return next((str(value) for value in values if value), "")


def _document_frequency(documents: Sequence[BM25Document]) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for doc in documents:
        for term in doc.frequencies:
            frequencies[term] = frequencies.get(term, 0) + 1
    return frequencies


def _average_length(documents: Sequence[BM25Document]) -> float:
    if not documents:
        return EMPTY_AVG_DOC_LENGTH
    return sum(doc.length for doc in documents) / len(documents)


def _bm25_score(query_terms: Counter[str], doc: BM25Document, index: BM25Index) -> float:
    score = 0.0
    for term, query_count in query_terms.items():
        if term not in doc.frequencies:
            continue
        score += query_count * _term_score(term, doc, index)
    return score


def _term_score(term: str, doc: BM25Document, index: BM25Index) -> float:
    frequency = doc.frequencies[term]
    idf = _idf(term, index)
    denominator = frequency + BM25_K1 * _length_norm(doc, index)
    return idf * (frequency * (BM25_K1 + 1.0)) / denominator


def _idf(term: str, index: BM25Index) -> float:
    total = len(index.documents)
    frequency = index.document_frequency.get(term, 0)
    return math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))


def _length_norm(doc: BM25Document, index: BM25Index) -> float:
    return 1.0 - BM25_B + BM25_B * (doc.length / index.average_length)


def _batch_limit(limit: int | None, loaded_count: int) -> int:
    if limit is None:
        return SCROLL_BATCH_SIZE
    return max(min(SCROLL_BATCH_SIZE, limit - loaded_count), RANK_START)


def _limit_reached(payloads: Sequence[dict[str, Any]], limit: int | None) -> bool:
    return limit is not None and len(payloads) >= limit
