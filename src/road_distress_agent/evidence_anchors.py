"""Shared lexical anchors for BM25 retrieval and deterministic evidence checks."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from road_distress_agent.state import RetrievedChunk

CHINESE_NGRAM_MIN = 2
CHINESE_NGRAM_MAX = 3
TOKEN_RE = re.compile(r"\d+(?:\.\d+)*(?:mm|cm|m|%|℃)?|[a-zA-Z][a-zA-Z0-9_-]*")
CHINESE_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
QUOTED_PHRASE_RE = re.compile(r'["“”「」『』]([^"“”「」『』]+)["“”「」『』]')
CHINESE_QUESTION_PHRASE_RE = re.compile(r"请问|什么是|是什么|如何|怎么|怎样|为何")

QUESTION_TOKENS = frozenset(
    {
        "a",
        "an",
        "are",
        "how",
        "is",
        "please",
        "the",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "什么",
        "为何",
        "如何",
        "怎么",
        "怎样",
        "请问",
    }
)


def normalize_text(text: str) -> str:
    """Normalize compatibility characters and case without altering content."""
    return unicodedata.normalize("NFKC", text).lower()


def tokenize(text: str) -> list[str]:
    """Tokenize English, numbers, units, and Chinese 2-3 grams."""
    normalized = normalize_text(text)
    tokens = [match.group(0) for match in TOKEN_RE.finditer(normalized)]
    return [*tokens, *_chinese_ngrams(normalized)]


def query_anchors(query: str) -> tuple[str, ...]:
    """Return stable, unique content anchors for a user query."""
    normalized = normalize_text(query)
    phrases = tuple(match.group(1).strip() for match in QUOTED_PHRASE_RE.finditer(normalized))
    content_text = CHINESE_QUESTION_PHRASE_RE.sub(" ", normalized)
    tokens = (token for token in tokenize(content_text) if token not in QUESTION_TOKENS)
    return _unique((*phrases, *tokens))


def query_anchor_coverage(query: str, chunks: Sequence[RetrievedChunk]) -> float:
    """Measure query-anchor coverage over the union of all supplied chunks."""
    anchors = query_anchors(query)
    if not anchors:
        raise ValueError("query must contain at least one evidence anchor")
    texts = tuple(_chunk_text(chunk) for chunk in chunks)
    chunk_tokens = frozenset(token for text in texts for token in tokenize(text))
    normalized_corpus = "\n".join(normalize_text(text) for text in texts)
    covered = sum(_anchor_present(anchor, chunk_tokens, normalized_corpus) for anchor in anchors)
    return covered / len(anchors)


def _anchor_present(anchor: str, tokens: frozenset[str], corpus: str) -> bool:
    if anchor in tokens:
        return True
    return any(character.isspace() for character in anchor) and anchor in corpus


def _chunk_text(chunk: RetrievedChunk) -> str:
    values = (
        chunk.clause_id,
        *chunk.heading_path,
        chunk.context_prefix,
        chunk.text,
    )
    return "\n".join(str(value) for value in values if value)


def _chinese_ngrams(text: str) -> list[str]:
    return [gram for match in CHINESE_RUN_RE.finditer(text) for gram in _ngrams(match.group(0))]


def _ngrams(text: str) -> list[str]:
    return [
        text[index : index + size]
        for size in range(CHINESE_NGRAM_MIN, CHINESE_NGRAM_MAX + 1)
        for index in range(len(text) - size + 1)
    ]


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))
