"""Exact Qdrant payload lookup by chunk_id."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from road_distress_agent.errors import BoundaryError
from road_distress_agent.qdrant_errors import classify_qdrant_error
from road_distress_agent.state import Citation
from road_distress_agent.tools.qdrant_rag import payload_to_citation

SCROLL_ONE = 1
RANK_START = 1


class QdrantChunkLookupTool:
    """Fetch already-indexed chunks without vector search."""

    def __init__(self, *, client: QdrantClient, collection: str, url: str | None = None) -> None:
        self._client = client
        self._collection = collection
        self._url = url

    def fetch_by_chunk_ids(
        self,
        chunk_ids: Sequence[str],
        limit: int,
    ) -> list[Citation]:
        payloads: list[dict[str, Any]] = []
        for chunk_id in _unique_ids(chunk_ids):
            if len(payloads) >= limit:
                break
            payload = self._payload_for_chunk(chunk_id)
            if payload:
                payloads.append(payload)
        return [
            payload_to_citation(payload, score=0.0, rank=rank)
            for rank, payload in enumerate(payloads, RANK_START)
        ]

    def _payload_for_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        try:
            records, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=_chunk_filter(chunk_id),
                limit=SCROLL_ONE,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise self._qdrant_error(exc) from exc
        return (records[0].payload or {}) if records else None

    def _qdrant_error(self, exc: Exception) -> BoundaryError:
        if isinstance(exc, BoundaryError):
            return exc
        info = classify_qdrant_error(
            exc,
            step="CHUNK_LOOKUP",
            url=self._url,
            collection=self._collection,
        )
        return BoundaryError(info, exc)


def _chunk_filter(chunk_id: str) -> Filter:
    return Filter(must=[FieldCondition(key="chunk_id", match=MatchValue(value=str(chunk_id)))])


def _unique_ids(chunk_ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for chunk_id in chunk_ids:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        unique.append(chunk_id)
    return unique
