"""Shared data models for the raw-standard ingestion middle layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ParserBackend = Literal["mineru"]


def _compact_text(value: str) -> str:
    return " ".join(value.replace("\u3000", " ").split())


@dataclass(slots=True)
class ParsedBlock:
    block_id: str
    source_doc_id: str
    source_path: str
    page_number: int
    order: int
    text: str
    layout_type: str = "text"
    bbox: list[float] | None = None
    positions: list[list[float]] = field(default_factory=list)
    table_raw: Any = None
    table_path: str | None = None
    image_paths: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def normalized_text(self) -> str:
        return _compact_text(self.text)


@dataclass(slots=True)
class ParsedLine:
    line_id: str
    block_id: str
    source_doc_id: str
    source_path: str
    page_number: int
    block_order: int
    line_order: int
    text: str
    layout_type: str = "text"
    bbox: list[float] | None = None
    positions: list[list[float]] = field(default_factory=list)

    @property
    def order_key(self) -> tuple[int, int, int]:
        return (self.page_number, self.block_order, self.line_order)

    def normalized_text(self) -> str:
        return _compact_text(self.text)


@dataclass(slots=True)
class ParsedDocument:
    source_doc_id: str
    source_path: str
    parser_backend: ParserBackend
    page_count: int | None
    blocks: list[ParsedBlock] = field(default_factory=list)
    lines: list[ParsedLine] = field(default_factory=list)
    outlines: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def source_name(self) -> str:
        return Path(self.source_path).name


@dataclass(slots=True)
class HeadingCandidate:
    line_id: str
    block_id: str
    source_doc_id: str
    page_number: int
    order_key: tuple[int, int, int]
    raw_clause_id: str
    canonical_clause_id: str
    clause_aliases: list[str]
    level: int
    title: str
    raw_text: str
    anomaly_type: str | None = None
    anomaly_reason: str | None = None
    parent_clause_id: str | None = None
    heading_path: list[str] = field(default_factory=list)

    @property
    def display_text(self) -> str:
        if self.title:
            return f"{self.canonical_clause_id} {self.title}".strip()
        return self.canonical_clause_id


@dataclass(slots=True)
class HeadingNode:
    clause_id: str
    title: str
    level: int
    page_number: int
    raw_clause_id: str | None = None
    aliases: list[str] = field(default_factory=list)
    anomaly_type: str | None = None
    anomaly_reason: str | None = None
    children: list[HeadingNode] = field(default_factory=list)


@dataclass(slots=True)
class HeadingTopology:
    source_doc_id: str
    headings: list[HeadingCandidate]
    roots: list[HeadingNode]
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    profile: str = "generic"


@dataclass(slots=True)
class RawIngestionChunk:
    chunk_id: str
    source_doc_id: str
    source_path: str
    source_pages: str
    text: str
    raw_text: str
    heading_path: list[str]
    raw_clause_id: str | None = None
    canonical_clause_id: str | None = None
    clause_aliases: list[str] = field(default_factory=list)
    cross_refs: list[str] = field(default_factory=list)
    resolved_cross_refs: list[dict[str, Any]] = field(default_factory=list)
    sequential_group_id: str | None = None
    step_index: int | None = None
    step_title: str | None = None
    parent_chunk_id: str | None = None
    table_raw: Any = None
    table_paths: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Pass0Artifacts:
    source_doc_id: str
    clause_index: dict[str, list[str]]
    chunk_index: dict[str, dict[str, Any]]
    cross_refs: dict[str, list[dict[str, Any]]]
    sequential_groups: dict[str, dict[str, Any]]
    anomalies: list[dict[str, Any]]


@dataclass(slots=True)
class IngestionMiddleLayer:
    parsed_document: ParsedDocument
    topology: HeadingTopology
    chunks: list[RawIngestionChunk]
    pass0: Pass0Artifacts


def dataclass_to_dict(value: Any) -> Any:
    """Return a JSON-friendly dict/list tree for dataclass-based models."""
    return asdict(value)
