"""Build and aggregate optional logical-subclause rerank passages."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from road_distress_agent.state import RetrievedChunk

_NUMBERED_CLAUSE_START_RE = re.compile(r"(?m)^[ \t]*(?=\d+(?:\.\d+){2,}(?:[^\d.]|$))")


@dataclass(frozen=True)
class ChunkRerankScore:
    score: float
    passage_count: int
    best_passage_index: int


@dataclass(frozen=True)
class RerankPassageBatch:
    passages: tuple[str, ...]
    chunk_indices: tuple[int, ...]
    chunk_count: int

    def aggregate(self, scores: Sequence[float]) -> list[ChunkRerankScore]:
        if len(scores) != len(self.passages):
            raise ValueError("Rerank passage score count does not match the passage batch.")
        best_scores = [float("-inf")] * self.chunk_count
        best_indices = [0] * self.chunk_count
        counts = [0] * self.chunk_count
        for score, chunk_index in zip(scores, self.chunk_indices, strict=True):
            passage_index = counts[chunk_index]
            counts[chunk_index] += 1
            if score > best_scores[chunk_index]:
                best_scores[chunk_index] = float(score)
                best_indices[chunk_index] = passage_index
        return [
            ChunkRerankScore(score, counts[index], best_indices[index])
            for index, score in enumerate(best_scores)
        ]


def build_rerank_passage_batch(
    chunks: Sequence[RetrievedChunk],
    *,
    split_numbered_clauses: bool,
) -> RerankPassageBatch:
    passages: list[str] = []
    chunk_indices: list[int] = []
    for chunk_index, chunk in enumerate(chunks):
        chunk_passages = _chunk_passages(chunk, split_numbered_clauses)
        passages.extend(chunk_passages)
        chunk_indices.extend([chunk_index] * len(chunk_passages))
    return RerankPassageBatch(tuple(passages), tuple(chunk_indices), len(chunks))


def _chunk_passages(chunk: RetrievedChunk, split_numbered: bool) -> tuple[str, ...]:
    annotations = chunk.annotations or {}
    payload_text = annotations.get("embed_text")
    if isinstance(payload_text, str) and payload_text.strip():
        return (payload_text,)
    segments = _numbered_segments(chunk.text or "") if split_numbered else ()
    texts = segments or (chunk.text or "",)
    return tuple(_format_passage(chunk, text) for text in texts)


def _numbered_segments(text: str) -> tuple[str, ...]:
    starts = [match.start() for match in _NUMBERED_CLAUSE_START_RE.finditer(text)]
    if len(starts) <= 1:
        return ()
    boundaries = [0, *starts[1:], len(text)]
    return tuple(
        text[start:end].strip()
        for start, end in zip(boundaries, boundaries[1:], strict=False)
        if text[start:end].strip()
    )


def _format_passage(chunk: RetrievedChunk, text: str) -> str:
    prefix = chunk.context_prefix or ""
    heading = " → ".join(chunk.heading_path or [])
    return f"{prefix}\n\n【{heading}】\n{text}".strip()
