"""Build procedure-step chunks under structure-aware parent chunks."""

from __future__ import annotations

import re
from typing import Any

from road_distress_agent.ingestion.ids import stable_id
from road_distress_agent.ingestion.models import RawIngestionChunk

METHOD_GROUP_RE = re.compile(r"^\s*[（(]?\s*(?P<index>\d{1,2})\s*[)）]\s*(?P<title>.{2,40})$")
STEP_RE = re.compile(r"^\s*(?P<index>\d{1,2})\s+(?P<title>[^\d\s].{1,40})$")
STEP_TITLE_STOPWORDS = {"注意事项", "序号", "病害类型", "病害描述", "病害图例"}


def build_step_chunks(chunks: list[RawIngestionChunk]) -> list[RawIngestionChunk]:
    step_chunks: list[RawIngestionChunk] = []
    for parent in chunks:
        step_chunks.extend(_step_chunks_from_parent(parent))
    return step_chunks


def _step_chunks_from_parent(parent: RawIngestionChunk) -> list[RawIngestionChunk]:
    if not _may_have_steps(parent):
        return []
    lines = [line.strip() for line in parent.raw_text.splitlines() if line.strip()]
    groups = _group_step_events(_extract_step_events(lines))
    return _chunks_from_groups(parent, lines, groups)


def _may_have_steps(parent: RawIngestionChunk) -> bool:
    path_text = " ".join(parent.heading_path)
    if "典型病害" in path_text or "病害表" in parent.raw_text:
        return False
    signal = path_text + parent.raw_text[:200]
    return bool(re.search(r"(维修操作流程|操作流程|施工|处治|修补|灌缝|挖补|铣刨)", signal))


def _extract_step_events(lines: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    method_title: str | None = None
    method_index: int | None = None
    for line_index, line in enumerate(lines):
        method = METHOD_GROUP_RE.match(line)
        if method and not STEP_RE.match(line):
            method_index = int(method.group("index"))
            method_title = _clean_step_title(method.group("title"))
            continue
        parsed_step = _parse_step_event(lines, line_index)
        if parsed_step and _valid_step_title(parsed_step[1]):
            events.append(_step_event(line_index, parsed_step, method_index, method_title))
    return events


def _step_event(
    line_index: int,
    parsed_step: tuple[int, str],
    method_index: int | None,
    method_title: str | None,
) -> dict[str, Any]:
    step_index, step_title = parsed_step
    return {
        "line_index": line_index,
        "step_index": step_index,
        "step_title": step_title,
        "method_index": method_index,
        "method_title": method_title,
    }


def _group_step_events(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_step = 0
    previous_method = (None, None)
    for event in events:
        method_key = (event["method_index"], event["method_title"])
        if _starts_new_group(current, event, previous_step, previous_method):
            if _valid_step_group(current):
                groups.append(current)
            current = [event]
        else:
            current.append(event)
        previous_step = int(event["step_index"])
        previous_method = method_key
    if _valid_step_group(current):
        groups.append(current)
    return groups


def _starts_new_group(
    current: list[dict[str, Any]],
    event: dict[str, Any],
    previous_step: int,
    previous_method: tuple[object, object],
) -> bool:
    method_key = (event["method_index"], event["method_title"])
    return not current or event["step_index"] < previous_step or method_key != previous_method


def _chunks_from_groups(
    parent: RawIngestionChunk,
    lines: list[str],
    groups: list[list[dict[str, Any]]],
) -> list[RawIngestionChunk]:
    chunks: list[RawIngestionChunk] = []
    for group in groups:
        method_title = str(group[0].get("method_title") or parent.heading_path[-1])
        group_id = _step_group_id(parent, method_title)
        for index, _event in enumerate(group):
            chunks.append(_step_chunk(parent, lines, group, index, method_title, group_id))
    return chunks


def _step_group_id(parent: RawIngestionChunk, method_title: str) -> str:
    return (
        f"{parent.source_doc_id}:steps:"
        f"{stable_id(parent.chunk_id, method_title, parent.canonical_clause_id)}"
    )


def _step_chunk(
    parent: RawIngestionChunk,
    lines: list[str],
    group: list[dict[str, Any]],
    index: int,
    method_title: str,
    group_id: str,
) -> RawIngestionChunk:
    event = group[index]
    raw_text = _step_text(lines, group, index)
    step_index = index + 1
    step_title = str(event["step_title"])
    heading_path = [*parent.heading_path, method_title, f"步骤 {step_index} {step_title}"]
    return RawIngestionChunk(
        chunk_id=f"{group_id}:step:{step_index:02d}:{stable_id(raw_text)}",
        source_doc_id=parent.source_doc_id,
        source_path=parent.source_path,
        source_pages=parent.source_pages,
        text=raw_text,
        raw_text=raw_text,
        heading_path=heading_path,
        raw_clause_id=parent.raw_clause_id,
        canonical_clause_id=parent.canonical_clause_id,
        clause_aliases=parent.clause_aliases,
        sequential_group_id=group_id,
        step_index=step_index,
        step_title=step_title,
        parent_chunk_id=parent.chunk_id,
        table_raw=parent.table_raw,
        table_paths=parent.table_paths,
        image_paths=parent.image_paths,
        metadata=_step_metadata(parent, event, step_index, step_title, group_id, heading_path),
    )


def _step_text(lines: list[str], group: list[dict[str, Any]], index: int) -> str:
    start = int(group[index]["line_index"])
    end = int(group[index + 1]["line_index"]) if index + 1 < len(group) else len(lines)
    return "\n".join(lines[start:end]).strip()


def _step_metadata(
    parent: RawIngestionChunk,
    event: dict[str, Any],
    step_index: int,
    step_title: str,
    group_id: str,
    heading_path: list[str],
) -> dict[str, Any]:
    return {
        **parent.metadata,
        "sequential_group_id": group_id,
        "step_index": step_index,
        "raw_step_index": int(event["step_index"]),
        "step_title": step_title,
        "parent_chunk_id": parent.chunk_id,
        "heading_path": heading_path,
    }


def _parse_step_event(lines: list[str], line_index: int) -> tuple[int, str] | None:
    line = lines[line_index]
    step = STEP_RE.match(line)
    if step:
        return int(step.group("index")), _clean_step_title(step.group("title"))
    if not re.fullmatch(r"\d{1,2}", line):
        return None
    if line_index + 1 >= len(lines):
        return None
    return int(line), _clean_step_title(lines[line_index + 1])


def _clean_step_title(title: str) -> str:
    return " ".join(title.replace("\u3000", " ").split()).strip(" ：:。；;，,")


def _valid_step_title(title: str) -> bool:
    if not title or title in STEP_TITLE_STOPWORDS:
        return False
    if len(title) > 24:
        return False
    if re.search(r"(mm|cm|m2|℃|L/|[0-9])", title):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", title))


def _valid_step_group(events: list[dict[str, Any]]) -> bool:
    if len(events) < 2:
        return False
    indexes = [int(event["step_index"]) for event in events]
    if indexes[0] != 1:
        return False
    return all(0 <= right - left <= 2 for left, right in zip(indexes, indexes[1:], strict=False))
