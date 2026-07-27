"""Qdrant + BGE-M3 hybrid-search RAG tool.

Uses BGE-M3's dense (1024-dim) + sparse vectors with Qdrant RRF fusion.
BGE-M3 is lazy-loaded on first query call and cached per process.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Collection, Mapping
from functools import lru_cache
from threading import Lock
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    Prefetch,
    SparseVector,
)

from road_distress_agent.error_classifiers import classify_model_load_error
from road_distress_agent.errors import BoundaryError, ErrorCategory, make_error_info, raw_exception
from road_distress_agent.qdrant_errors import classify_qdrant_error
from road_distress_agent.state import Citation
from road_distress_agent.tools.locked_lazy import LockedLazy

_COLLECTION_DEFAULT = "road_distress_documents"
_Encoder = Callable[[str], tuple[list[float], SparseVector]]
_BGEM3_MODEL = LockedLazy[Any]()
_BGEM3_ENCODER_LOCK = Lock()


def _load_bgem3_model() -> Any:
    """Load BGE-M3 once per Python process."""
    return _BGEM3_MODEL.get(_create_bgem3_model)


def _create_bgem3_model() -> Any:
    try:
        from FlagEmbedding import BGEM3FlagModel  # noqa: PLC0415

        return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    except Exception as exc:
        raise BoundaryError(classify_model_load_error(exc), exc) from exc


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """Convert a string chunk_id to a deterministic UUID for Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def payload_to_citation(payload: dict[str, Any], score: float, rank: int) -> Citation:
    """Map a Qdrant point payload dict to a Citation."""
    return Citation(
        chunk_id=payload.get("chunk_id"),
        citation_id=payload.get("chunk_id"),
        clause_id=payload.get("clause_id"),
        source_clause=payload.get("clause_id"),
        source_doc=payload.get("source_doc_id"),
        document_id=payload.get("source_doc_id"),
        source_pages=payload.get("source_pages"),
        title=str(payload.get("source_doc_id") or ""),
        snippet=payload.get("rawtext", ""),
        text=payload.get("rawtext", ""),
        context_prefix=payload.get("context_prefix"),
        heading_path=payload.get("heading_path") or [],
        cross_refs=payload.get("cross_refs") or [],
        sequential_group_id=payload.get("sequential_group_id"),
        step_index=payload.get("step_index"),
        score=score,
        similarity=score,
        rank=rank,
        annotations=payload,
    )


def qdrant_filter_from_filters(filters: dict[str, Any]) -> Filter | None:
    """Build the supported Qdrant payload filter from retrieval filters."""
    must_conditions = _filter_conditions(filters or {})
    return Filter(must=must_conditions) if must_conditions else None


def _filter_conditions(filters: dict[str, Any]) -> list[FieldCondition]:
    must_conditions: list[FieldCondition] = []
    if clause_id := filters.get("clause_id"):
        must_conditions.append(
            FieldCondition(key="clause_id", match=MatchValue(value=str(clause_id)))
        )
    elif group_id := filters.get("sequential_group_id"):
        must_conditions.append(
            FieldCondition(key="sequential_group_id", match=MatchValue(value=str(group_id)))
        )
    if role := filters.get("semantic_role"):
        must_conditions.append(
            FieldCondition(key="semantic_role", match=_semantic_role_match(role))
        )
    if source_doc_id := filters.get("source_doc_id"):
        must_conditions.append(
            FieldCondition(key="source_doc_id", match=MatchValue(value=str(source_doc_id)))
        )
    return must_conditions


def _semantic_role_match(role: Any) -> MatchAny | MatchValue:
    if isinstance(role, Collection) and not isinstance(role, (str, bytes, Mapping)):
        return MatchAny(any=[str(value) for value in role])
    return MatchValue(value=str(role))


_payload_to_citation = payload_to_citation


class QdrantRAGTool:
    """Hybrid-search RAG tool: Qdrant (dense + sparse) + BGE-M3 embeddings.

    Parameters
    ----------
    url:        Qdrant server URL. Defaults to $QDRANT_URL or http://localhost:6333.
    collection: Collection name. Defaults to $QDRANT_COLLECTION or
                "road_distress_documents".
    top_k:      Number of results to return. Defaults to $QDRANT_TOP_K or 5.
    client:     Inject a QdrantClient (used in tests with in-memory client).
    _encoder:   Inject an encoder callable (used in tests to avoid loading BGE-M3).
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        collection: str | None = None,
        top_k: int | None = None,
        client: QdrantClient | None = None,
        _encoder: _Encoder | None = None,
    ) -> None:
        self._url = url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        self._client = client or QdrantClient(url=self._url)
        self._collection = collection or os.environ.get("QDRANT_COLLECTION", _COLLECTION_DEFAULT)
        self._top_k = top_k if top_k is not None else int(os.environ.get("QDRANT_TOP_K", "5"))
        self._encoder_override = _encoder

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _get_encoder(self) -> _Encoder:
        """Return the encoder, lazy-loading BGE-M3 on first call."""
        if self._encoder_override is not None:
            return self._encoder_override

        model = _load_bgem3_model()

        def _bgem3_encode(text: str) -> tuple[list[float], SparseVector]:
            try:
                with _BGEM3_ENCODER_LOCK:
                    output = model.encode(
                        [text],
                        batch_size=1,
                        max_length=512,
                        return_dense=True,
                        return_sparse=True,
                        return_colbert_vecs=False,
                    )
                dense: list[float] = output["dense_vecs"][0].tolist()
                lw: dict[str, float] = output["lexical_weights"][0]
                sparse = SparseVector(
                    indices=[int(k) for k in lw.keys()],
                    values=[float(v) for v in lw.values()],
                )
                return dense, sparse
            except Exception as exc:
                raise BoundaryError(_encode_error_info(exc), exc) from exc

        return _bgem3_encode

    # ------------------------------------------------------------------
    # RAGTool protocol
    # ------------------------------------------------------------------

    def search_documents(self, query: str, filters: dict[str, Any]) -> list[Citation]:
        """Hybrid search: dense + sparse prefetch, RRF fusion, optional filter.

        Supported filter keys (evaluated in priority order; first match wins):
          - clause_id:            exact match on clause_id field
          - sequential_group_id:  exact match on sequential_group_id field
          - semantic_role:        exact match on semantic_role field
                                  (values: disease_definition, method_selection,
                                   construction_step, acceptance_criteria, general_info)
          - source_doc_id:        exact match on source_doc_id field

        Multiple keys can coexist only for semantic_role + (clause_id or
        sequential_group_id) + source_doc_id: in that case conditions are ANDed.
        """
        dense_vec, sparse_vec = self._encode_query(query)

        q_filter = qdrant_filter_from_filters(filters or {})

        try:
            results = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    Prefetch(query=dense_vec, using="dense", limit=50, filter=q_filter),
                    Prefetch(query=sparse_vec, using="sparse", limit=50, filter=q_filter),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=self._top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise self._qdrant_error(exc, "SEARCH") from exc

        citations = [
            payload_to_citation(hit.payload or {}, hit.score, rank)
            for rank, hit in enumerate(results.points, 1)
        ]
        return _dedupe(citations)

    def fetch_by_sequential_group(self, group_id: str) -> list[Citation]:
        """Scroll ALL chunks that belong to a sequential_group_id without vector search.

        Used by detail_retriever_v2 to guarantee complete step sequences even when
        the group has more steps than top_k.
        """
        q_filter = Filter(
            must=[FieldCondition(key="sequential_group_id", match=MatchValue(value=group_id))]
        )
        try:
            results, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=q_filter,
                limit=50,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise self._qdrant_error(exc, "SCROLL") from exc
        return [
            payload_to_citation(hit.payload or {}, score=0.0, rank=rank)
            for rank, hit in enumerate(results, 1)
        ]

    def _encode_query(self, query: str) -> tuple[list[float], SparseVector]:
        try:
            return self._get_encoder()(query)
        except BoundaryError:
            raise
        except Exception as exc:
            raise BoundaryError(_encode_error_info(exc), exc) from exc

    def _qdrant_error(self, exc: Exception, step: str) -> BoundaryError:
        if isinstance(exc, BoundaryError):
            return exc
        info = classify_qdrant_error(
            exc,
            step=step,
            url=self._url,
            collection=self._collection,
        )
        return BoundaryError(info, exc)


@lru_cache(maxsize=1)
def get_default_qdrant_rag_tool() -> QdrantRAGTool:
    """Return the process-level default RAG tool for query-time nodes."""
    return QdrantRAGTool()


@lru_cache(maxsize=8)
def get_qdrant_rag_tool_for_top_k(top_k: int) -> QdrantRAGTool:
    """Return a cached RAG tool configured for a specific result count."""
    return QdrantRAGTool(top_k=top_k)


def _dedupe(citations: list[Citation]) -> list[Citation]:
    seen: set[str] = set()
    unique: list[Citation] = []
    for c in citations:
        key = c.chunk_id or f"{c.source_doc}:{c.clause_id}:{c.rank}"
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _encode_error_info(exc: Exception):
    return make_error_info(
        domain="EMBED",
        step="ENCODE",
        category=ErrorCategory.INTERNAL,
        responsibility="文本编码失败",
        reason=raw_exception(exc),
        hint="查看 raw，确认输入文本、BGE 模型和 FlagEmbedding 运行环境。",
        raw=raw_exception(exc),
        retriable=False,
    )
