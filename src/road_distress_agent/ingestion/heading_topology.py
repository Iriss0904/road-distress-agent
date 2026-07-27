"""Build a heading tree from parsed PDF lines.

The builder is intentionally conservative: it uses numeric clause markers to
find candidate headings, but it does not let a single malformed marker rewrite
the document context.  This is important for PDFs like the Shenzhen guide,
where a section under ``2.3.4`` is printed as ``3.3.4.1``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from road_distress_agent.ingestion.appendix_headings import appendix_candidate_from_line
from road_distress_agent.ingestion.content_window import content_lines
from road_distress_agent.ingestion.heading_tree_builder import build_tree
from road_distress_agent.ingestion.models import (
    HeadingCandidate,
    HeadingTopology,
    ParsedDocument,
    ParsedLine,
)
from road_distress_agent.ingestion.national_standard import (
    NATIONAL_STANDARD_PROFILE,
    detect_document_profile,
    national_heading_title_is_valid,
    normalize_clause_spacing,
)

CLAUSE_RE = re.compile(r"^\s*(?P<clause>\d+(?:\.\d+){0,5})\s*(?P<title>[^\d\s].{0,90})?\s*$")
CLAUSE_LINE_RE = re.compile(r"^\s*(?P<clause>\d+(?:\.\d+){0,5})(?:\s+|$)(?P<title>.*)$")

TOP_LEVEL_TITLES = {
    "总则",
    "术语",
    "基本规定",
    "材料",
    "设计",
    "施工",
    "检查与验收",
    "验收标准",
    "沥青路面养护",
    "水泥混凝土路面养护",
    "人行道养护",
}


def _parts(clause_id: str) -> tuple[int, ...]:
    return tuple(int(part) for part in clause_id.split(".") if part)


def _clause_id(parts: Iterable[int]) -> str:
    return ".".join(str(part) for part in parts)


def _clean_title(title: str | None) -> str:
    if not title:
        return ""
    title = " ".join(title.replace("\u3000", " ").split())
    return title.strip(" ：:。；;，,")


def _looks_like_toc_or_page_noise(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return True
    if re.fullmatch(r"[IVXLC]+", compact):
        return True
    if re.fullmatch(r"\d{1,3}", compact):
        return True
    if "........" in compact:
        return True
    return False


def _single_level_is_heading(
    *,
    clause: str,
    title: str,
    line: ParsedLine,
    stack: dict[int, HeadingCandidate],
) -> bool:
    if "." in clause:
        return True
    if int(clause) > 20:
        return False
    # MinerU's structural signal is authoritative: if it tagged this line as a
    # title, trust it over the numeric heuristics below. Otherwise a real chapter
    # such as "2 术语和符号" gets vetoed by the deep-stack rule and its children
    # are mis-attached to the previous chapter.
    if line.layout_type == "title":
        return True
    if title in TOP_LEVEL_TITLES:
        return True
    # A lone number inside a deep section (with no title signal) is much more
    # likely to be a list item than a new chapter heading.
    if stack and max(stack) >= 2 and int(clause) <= 20:
        return False
    if title.endswith(("养护", "验收", "规定", "标准")) and len(title) <= 16:
        return True
    return len(title) <= 16 and bool(re.search(r"[\u4e00-\u9fff]", title))


def _candidate_from_line(
    line: ParsedLine,
    stack: dict[int, HeadingCandidate],
    profile: str,
) -> HeadingCandidate | None:
    if line.layout_type == "table":
        return None
    text = normalize_clause_spacing(line.normalized_text())
    if _looks_like_toc_or_page_noise(text):
        return None
    appendix = appendix_candidate_from_line(line, _clean_title)
    if appendix:
        return appendix
    match = _numeric_match(text, profile)
    if not match:
        return None
    return _numeric_candidate_from_match(line, stack, match, profile)


def _numeric_match(text: str, profile: str) -> re.Match[str] | None:
    if profile == NATIONAL_STANDARD_PROFILE:
        return CLAUSE_LINE_RE.match(text)
    return CLAUSE_RE.match(text)


def _numeric_candidate_from_match(
    line: ParsedLine,
    stack: dict[int, HeadingCandidate],
    match: re.Match[str],
    profile: str,
) -> HeadingCandidate | None:
    raw_clause_id = match.group("clause")
    raw_parts = _parts(raw_clause_id)
    title = _numeric_title(match.group("title"), raw_parts, profile)
    level = _tree_level(raw_parts, profile)
    if not raw_parts or raw_parts[0] == 0:
        return None
    if _looks_like_table_suffix(match.group("title") or ""):
        return None
    if profile == NATIONAL_STANDARD_PROFILE and not national_heading_title_is_valid(
        raw_parts, title
    ):
        return None
    if not title and "." not in raw_clause_id:
        return None
    if not _single_level_is_heading(
        clause=raw_clause_id,
        title=title,
        line=line,
        stack=stack,
    ):
        return None
    canonical_parts, anomaly_type, anomaly_reason, parent_clause_id = _canonical_numeric_parts(
        raw_parts,
        stack,
    )
    canonical_clause_id = _clause_id(canonical_parts)
    aliases = [canonical_clause_id]
    if raw_clause_id != canonical_clause_id:
        aliases.append(raw_clause_id)

    return HeadingCandidate(
        line_id=line.line_id,
        block_id=line.block_id,
        source_doc_id=line.source_doc_id,
        page_number=line.page_number,
        order_key=line.order_key,
        raw_clause_id=raw_clause_id,
        canonical_clause_id=canonical_clause_id,
        clause_aliases=aliases,
        level=level,
        title=title,
        raw_text=line.normalized_text(),
        anomaly_type=anomaly_type,
        anomaly_reason=anomaly_reason,
        parent_clause_id=parent_clause_id,
    )


def _numeric_title(raw_title: str | None, raw_parts: tuple[int, ...], profile: str) -> str:
    title = _clean_title(raw_title)
    if profile == NATIONAL_STANDARD_PROFILE and len(raw_parts) >= 3:
        return ""
    return title


def _tree_level(raw_parts: tuple[int, ...], profile: str) -> int:
    if profile != NATIONAL_STANDARD_PROFILE or len(raw_parts) < 3:
        return len(raw_parts)
    if len(raw_parts) >= 3 and raw_parts[1] == 0:
        return 2
    return 3


def _looks_like_table_suffix(title: str) -> bool:
    return title.lstrip().startswith(("-", "－"))


def _canonical_numeric_parts(
    raw_parts: tuple[int, ...],
    stack: dict[int, HeadingCandidate],
) -> tuple[tuple[int, ...], str | None, str | None, str | None]:
    if len(raw_parts) <= 1:
        return raw_parts, None, None, None
    parent = stack.get(len(raw_parts) - 1)
    if not parent:
        return raw_parts, None, None, None
    raw_parent = raw_parts[:-1]
    parent_parts = _parts(parent.canonical_clause_id)
    if raw_parent == parent_parts or not _looks_like_shifted_child(raw_parent, parent_parts):
        return raw_parts, None, None, parent.canonical_clause_id
    anomaly_reason = (
        f"Raw parent {_clause_id(raw_parent)} does not match current parent "
        f"{parent.canonical_clause_id}; suffix matches, so this is treated as a "
        "local numbering anomaly."
    )
    return (
        (*parent_parts, raw_parts[-1]),
        "shifted_chapter_prefix",
        anomaly_reason,
        parent.canonical_clause_id,
    )


def _looks_like_shifted_child(
    raw_parent: tuple[int, ...],
    current_parent: tuple[int, ...],
) -> bool:
    if len(raw_parent) != len(current_parent):
        return False
    # A single-segment parent has an empty suffix, so the suffix comparison below
    # would be a vacuous () == () match. Requiring depth >= 2 keeps the check
    # meaningful and stops top-level chapters (e.g. 2.1 under a stale parent 1)
    # from being rewritten to 1.1.
    if len(raw_parent) < 2:
        return False
    if raw_parent == current_parent:
        return False
    # Shenzhen-style typo: 3.3.4.1 appears below 2.3.4.  Preserve the raw id
    # as an alias but attach the child under the current semantic parent.
    return raw_parent[1:] == current_parent[1:]


def build_heading_topology(document: ParsedDocument) -> HeadingTopology:
    profile = detect_document_profile(document)
    stack: dict[int, HeadingCandidate] = {}
    headings: list[HeadingCandidate] = []

    for line in content_lines(document.lines):
        candidate = _candidate_from_line(line, stack, profile)
        if candidate is None:
            continue
        candidate.heading_path = _heading_path_for(candidate, stack)
        headings.append(candidate)

        for level in list(stack):
            if level >= candidate.level:
                del stack[level]
        stack[candidate.level] = candidate

    roots = build_tree(headings)
    anomalies = [
        {
            "raw_clause_id": heading.raw_clause_id,
            "canonical_clause_id": heading.canonical_clause_id,
            "title": heading.title,
            "page_number": heading.page_number,
            "anomaly_type": heading.anomaly_type,
            "anomaly_reason": heading.anomaly_reason,
            "heading_path": heading.heading_path,
        }
        for heading in headings
        if heading.anomaly_type
    ]
    return HeadingTopology(
        source_doc_id=document.source_doc_id,
        headings=headings,
        roots=roots,
        anomalies=anomalies,
        profile=profile,
    )


def _heading_path_for(
    candidate: HeadingCandidate,
    stack: dict[int, HeadingCandidate],
) -> list[str]:
    path: list[str] = []
    for level in sorted(level for level in stack if level < candidate.level):
        path.append(stack[level].display_text)
    if candidate.raw_clause_id != candidate.canonical_clause_id:
        raw_suffix = f"原文 {candidate.raw_clause_id}"
        if candidate.title:
            path.append(f"{candidate.canonical_clause_id} {candidate.title} ({raw_suffix})")
        else:
            path.append(f"{candidate.canonical_clause_id} ({raw_suffix})")
    else:
        path.append(candidate.display_text)
    return path
