"""Read-only knowledge-source metadata and PDF serving for evidence tracing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from road_distress_agent.api.paths import data_dir
from road_distress_agent.errors import BoundaryError, error_info_payload
from road_distress_agent.qdrant_errors import classify_qdrant_error

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

DEFAULT_DOC_DIR = "raw"
COLLECTION_DEFAULT = "road_distress_documents"
PDF_GLOB = "*.pdf"
EMPTY_COUNT = 0
PREVIEW_LIMIT = 1
QDRANT_HTTP_ERROR = 502
PREVIEW_PAYLOAD_FIELDS = (
    "chunk_id",
    "rawtext",
    "context_prefix",
    "heading_path",
    "source_pages",
    "semantic_role",
    "clause_id",
)
INDEXED_STATUS = "indexed"
RAW_ONLY_STATUS = "raw_only"


@dataclass(frozen=True)
class QdrantKnowledgeContext:
    client: QdrantClient
    collection: str


@router.get("/summary")
def get_summary() -> dict[str, Any]:
    base = _existing_doc_dir()
    context = _qdrant_context()
    try:
        return _summary_payload(base, context)
    except BoundaryError as exc:
        raise _qdrant_http_error(exc) from exc


@router.get("/preview")
def get_preview(doc_id: str) -> dict[str, Any]:
    base = _existing_doc_dir()
    pdf = _find_doc(base, doc_id)
    context = _qdrant_context()
    try:
        return _preview_payload(context, pdf.stem)
    except BoundaryError as exc:
        raise _qdrant_http_error(exc) from exc


@router.get("/doc")
def get_doc(doc_id: str) -> FileResponse:
    pdf = _find_doc(_existing_doc_dir(), doc_id)
    return FileResponse(pdf.resolve(), media_type="application/pdf", filename=pdf.name)


def _summary_payload(
    base: Path,
    context: QdrantKnowledgeContext,
) -> dict[str, Any]:
    info = _collection_info(context, "KNOWLEDGE_SUMMARY")
    documents = [
        _document_payload(pdf, _chunk_count(context, pdf.stem, "KNOWLEDGE_SUMMARY"))
        for pdf in _pdfs(base)
    ]
    return {
        "collection": context.collection,
        "qdrant": _qdrant_payload(info),
        "documents": documents,
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }


def _preview_payload(
    context: QdrantKnowledgeContext,
    doc_id: str,
) -> dict[str, Any]:
    count = _chunk_count(context, doc_id, "KNOWLEDGE_PREVIEW")
    if count <= EMPTY_COUNT:
        raise HTTPException(status_code=404, detail="该文档未入 Qdrant，无法预览命中片段。")
    records, _ = _preview_records(context, doc_id)
    if not records:
        raise HTTPException(status_code=404, detail="该文档没有可预览的命中片段。")
    return {"doc_id": doc_id, "snippet": _snippet_payload(records[0].payload or {})}


def _collection_info(context: QdrantKnowledgeContext, step: str) -> Any:
    try:
        return context.client.get_collection(collection_name=context.collection)
    except Exception as exc:
        raise _qdrant_error(exc, step, context.collection) from exc


def _chunk_count(context: QdrantKnowledgeContext, doc_id: str, step: str) -> int:
    try:
        result = context.client.count(
            collection_name=context.collection,
            count_filter=_source_doc_filter(doc_id),
            exact=True,
        )
        return int(result.count or EMPTY_COUNT)
    except Exception as exc:
        raise _qdrant_error(exc, step, context.collection) from exc


def _preview_records(
    context: QdrantKnowledgeContext,
    doc_id: str,
) -> tuple[list[Any], Any]:
    try:
        return context.client.scroll(
            collection_name=context.collection,
            scroll_filter=_source_doc_filter(doc_id),
            limit=PREVIEW_LIMIT,
            with_payload=list(PREVIEW_PAYLOAD_FIELDS),
            with_vectors=False,
        )
    except Exception as exc:
        raise _qdrant_error(exc, "KNOWLEDGE_PREVIEW", context.collection) from exc


def _doc_dir() -> Path:
    raw = Path(os.environ.get("ROAD_DISTRESS_DOC_DIR", DEFAULT_DOC_DIR))
    return (raw if raw.is_absolute() else data_dir() / raw).resolve()


def _existing_doc_dir() -> Path:
    base = _doc_dir()
    if not base.exists() or not base.is_dir():
        raise HTTPException(status_code=500, detail=f"文档目录不存在：{base}")
    return base


def _collection_name() -> str:
    return os.environ.get("QDRANT_COLLECTION", COLLECTION_DEFAULT)


def _qdrant_context() -> QdrantKnowledgeContext:
    return QdrantKnowledgeContext(client=_qdrant_client(), collection=_collection_name())


def _qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    return QdrantClient(
        url=url,
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        check_compatibility=False,
    )


def _pdfs(base: Path) -> list[Path]:
    return sorted(base.glob(PDF_GLOB), key=lambda pdf: pdf.stem)


def _find_doc(base: Path, doc_id: str) -> Path:
    if _is_pathlike(doc_id):
        raise HTTPException(status_code=404, detail="未找到该文档。")
    for pdf in _pdfs(base):
        target = pdf.resolve()
        if _matches_doc(pdf, doc_id) and _inside_base(base, target):
            return pdf
    raise HTTPException(status_code=404, detail="未找到该文档。")


def _is_pathlike(doc_id: str) -> bool:
    return "/" in doc_id or "\\" in doc_id or ".." in doc_id


def _matches_doc(pdf: Path, doc_id: str) -> bool:
    return pdf.stem == doc_id or pdf.name == doc_id


def _inside_base(base: Path, target: Path) -> bool:
    return target == base or base in target.parents


def _source_doc_filter(doc_id: str) -> Filter:
    return Filter(must=[FieldCondition(key="source_doc_id", match=MatchValue(value=doc_id))])


def _document_payload(pdf: Path, chunk_count: int) -> dict[str, Any]:
    return {
        "doc_id": pdf.stem,
        "title": pdf.stem,
        "filename": pdf.name,
        "size_bytes": pdf.stat().st_size,
        "lane": "用户资料",
        "status": INDEXED_STATUS if chunk_count > EMPTY_COUNT else RAW_ONLY_STATUS,
        "chunk_count": chunk_count,
    }


def _qdrant_payload(info: Any) -> dict[str, Any]:
    return {
        "status": _value(info.status),
        "points_count": int(info.points_count or EMPTY_COUNT),
        "indexed_vectors_count": int(info.indexed_vectors_count or EMPTY_COUNT),
    }


def _snippet_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {field: payload.get(field) for field in PREVIEW_PAYLOAD_FIELDS}


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


def _qdrant_error(exc: Exception, step: str, collection: str) -> BoundaryError:
    if isinstance(exc, BoundaryError):
        return exc
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    info = classify_qdrant_error(exc, step=step, url=url, collection=collection)
    return BoundaryError(info, exc)


def _qdrant_http_error(exc: BoundaryError) -> HTTPException:
    return HTTPException(status_code=QDRANT_HTTP_ERROR, detail=error_info_payload(exc.info))
