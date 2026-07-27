"""Channel-aware retrieval fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from road_distress_agent.state import RetrievedChunk

RRF_K = 60.0
RANK_START = 1
MAX_QUALITY_SCORE = 1.0
MIN_QUALITY_SCORE = 0.0
EXACT_HIT_QUALITY = 1.0


@dataclass(frozen=True)
class ChannelResults:
    name: str
    chunks: Sequence[RetrievedChunk]
    weight: float
    protected: bool = False


@dataclass
class _FusionEntry:
    chunk: RetrievedChunk
    score: float = 0.0
    channels: list[str] = field(default_factory=list)
    channel_scores: dict[str, float] = field(default_factory=dict)
    protected: bool = False


def fuse_channel_results(
    channels: Sequence[ChannelResults],
    limit: int,
) -> list[RetrievedChunk]:
    entries: dict[str, _FusionEntry] = {}
    for channel in channels:
        _merge_channel(entries, channel)
    ranked = sorted(entries.values(), key=_entry_rank_key)
    return [
        _with_fusion_annotations(entry, rank)
        for rank, entry in enumerate(ranked[:limit], RANK_START)
    ]


def _merge_channel(entries: dict[str, _FusionEntry], channel: ChannelResults) -> None:
    for rank, chunk in enumerate(channel.chunks, RANK_START):
        key = _chunk_key(chunk)
        entry = entries.setdefault(key, _FusionEntry(chunk=chunk))
        entry.score += channel.weight / (RRF_K + rank)
        entry.protected = entry.protected or channel.protected
        _record_channel(entry, channel.name, chunk)


def _record_channel(entry: _FusionEntry, channel: str, chunk: RetrievedChunk) -> None:
    if channel not in entry.channels:
        entry.channels.append(channel)
    entry.channel_scores[channel] = _score(chunk)
    if _score(chunk) > _score(entry.chunk):
        entry.chunk = chunk


def _entry_rank_key(entry: _FusionEntry) -> tuple[int, float]:
    protected_rank = 0 if entry.protected else 1
    return (protected_rank, -entry.score)


def _with_fusion_annotations(entry: _FusionEntry, rank: int) -> RetrievedChunk:
    annotations = {
        **(entry.chunk.annotations or {}),
        "retrieval_channels": entry.channels,
        "channel_scores": entry.channel_scores,
        "fusion_score": entry.score,
        "protected_exact_hit": entry.protected,
        "evidence_quality": _evidence_quality(entry),
    }
    _copy_named_scores(annotations, entry.channel_scores)
    return entry.chunk.model_copy(
        update={
            "rank": rank,
            "score": entry.score,
            "similarity": entry.score,
            "annotations": annotations,
        }
    )


def _copy_named_scores(annotations: dict[str, object], scores: dict[str, float]) -> None:
    if "qdrant" in scores:
        annotations.setdefault("qdrant_score", scores["qdrant"])
    if "bm25" in scores:
        annotations.setdefault("bm25_score", scores["bm25"])
    if "clause" in scores:
        annotations.setdefault("clause_lookup_score", scores["clause"])


def _evidence_quality(entry: _FusionEntry) -> float | None:
    if entry.protected or "clause" in entry.channel_scores:
        return EXACT_HIT_QUALITY
    semantic_score = entry.channel_scores.get("qdrant")
    if semantic_score is not None:
        return _bounded_quality(semantic_score)
    bm25_score = entry.channel_scores.get("bm25")
    if bm25_score is not None:
        return _bounded_quality(bm25_score)
    return None


def _bounded_quality(value: float) -> float:
    return max(MIN_QUALITY_SCORE, min(MAX_QUALITY_SCORE, float(value)))


def _chunk_key(chunk: RetrievedChunk) -> str:
    if chunk.chunk_id or chunk.citation_id:
        return str(chunk.chunk_id or chunk.citation_id)
    return f"{chunk.source_doc}:{chunk.clause_id}:{chunk.rank}"


def _score(chunk: RetrievedChunk) -> float:
    return chunk.score if chunk.score is not None else chunk.similarity or 0.0
