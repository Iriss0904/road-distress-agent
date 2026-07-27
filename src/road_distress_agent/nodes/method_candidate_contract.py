"""Evidence-backed naming invariants for method candidates."""

from __future__ import annotations

import re

from road_distress_agent.method_name_catalog import canonical_method_name
from road_distress_agent.nodes.discriminator_tool_schemas import MAX_CANDIDATES
from road_distress_agent.state import MethodDiscriminatorOutput, MethodOption, RetrievedChunk

_HEADING_PREFIX = re.compile(r"^(?:(?:表)?\d+(?:\.\d+)*(?:[、.)）]\s*|\s+)|[（(]\d+[）)]\s*)")
_ENGLISH_TAIL = re.compile(r"\s+[A-Za-z][A-Za-z\s/()-]*$")
_RULE_TAIL = re.compile(r"(?:应符合下列规定|维修操作流程|质量检查.*|检验方法.*)$")
_INLINE_METHOD = re.compile(r"(?:可|应|宜)(?:优先|推荐)?采用(?P<name>[^，。；]+)")
_METHOD_SEPARATOR = re.compile(r"或|、|(?<!拌)和")
_INLINE_TAIL = re.compile(r"(?:修复|处治|维修)?(?:技术|工艺)$|时$")
_METHOD_SUFFIXES = (
    "施工",
    "加铺",
    "灌缝",
    "返铺",
    "重铺",
    "再生",
    "填充",
    "修补",
    "微表处",
    "维修流程",
)
_GENERIC_HEADINGS = frozenset({"施工", "维修", "养护", "维修流程"})


def normalize_method_candidate_names(
    output: MethodDiscriminatorOutput,
    chunks: list[RetrievedChunk],
) -> MethodDiscriminatorOutput:
    if not output.candidates:
        return output
    evidence_groups = _method_name_groups(chunks)
    evidence_names = _flatten_unique(evidence_groups)
    normalized = [_canonical_candidate(candidate) for candidate in output.candidates]
    candidates = _evidence_grounded_candidates(normalized, evidence_names, evidence_groups)
    _ensure_unique_names(candidates)
    return output.model_copy(update={"candidates": candidates})


def explicit_method_names(chunks: list[RetrievedChunk]) -> list[str]:
    return _flatten_unique(_method_name_groups(chunks))


def _method_name_groups(chunks: list[RetrievedChunk]) -> list[list[str]]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for index, chunk in enumerate(chunks):
        names = grouped.setdefault(_evidence_parent_key(chunk, index), [])
        sources = [*(chunk.heading_path[-1:] or []), *(chunk.text or "").splitlines()]
        for source in sources:
            _extend_unique(names, _method_names(source))
    return list(grouped.values())


def _evidence_parent_key(chunk: RetrievedChunk, index: int) -> tuple[str, str]:
    source = str(chunk.source_doc or chunk.document_id or "")
    parent = chunk.clause_id or chunk.chunk_id or chunk.citation_id or f"chunk-{index}"
    return source, str(parent)


def _flatten_unique(groups: list[list[str]]) -> list[str]:
    names: list[str] = []
    for group in groups:
        _extend_unique(names, group)
    return names


def explicit_method_headings(chunks: list[RetrievedChunk]) -> list[str]:
    names: list[str] = []
    for chunk in chunks:
        for heading in chunk.heading_path[-1:]:
            _extend_unique(names, _method_names(heading))
    return names


def _extend_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _method_names(heading: str) -> list[str]:
    text = _HEADING_PREFIX.sub("", heading.strip())
    names = [
        name
        for match in _INLINE_METHOD.finditer(text)
        for name in _inline_names(match.group("name"))
    ]
    bare = _bare_method_name(text)
    return [*names, *([bare] if bare and bare not in names else [])]


def _inline_names(value: str) -> list[str]:
    names = [_clean_inline_name(part) for part in _METHOD_SEPARATOR.split(value)]
    return [name for name in names if _is_method_name(name)]


def _clean_inline_name(value: str) -> str:
    name = _clean_name(value)
    return _INLINE_TAIL.sub("", name).strip()


def _bare_method_name(heading: str) -> str | None:
    name = _clean_name(heading)
    name = _ENGLISH_TAIL.sub("", name)
    name = _RULE_TAIL.sub("", name).strip(" ：:，,。；;")
    if name in _GENERIC_HEADINGS:
        return None
    if any(marker in name for marker in ("，", "；", "可采用", "应采用", "宜采用")):
        return None
    return name if _is_method_name(name) else None


def _canonical_candidate(candidate: MethodOption) -> MethodOption:
    return candidate.model_copy(update={"name": canonical_method_name(candidate.name)})


def _evidence_grounded_candidates(
    candidates: list[MethodOption],
    evidence_names: list[str],
    evidence_groups: list[list[str]],
) -> list[MethodOption]:
    coverage = _coverage_candidates(evidence_groups, candidates)
    grounded = [
        candidate
        for name in evidence_names
        if (candidate := _candidate_for_evidence(name, candidates)) is not None
    ]
    return _fill_candidate_slots(coverage, [grounded, candidates])


def _coverage_candidates(
    evidence_groups: list[list[str]],
    candidates: list[MethodOption],
) -> list[MethodOption]:
    covered: list[MethodOption] = []
    for names in evidence_groups:
        candidate = _first_unique_group_candidate(names, candidates, covered)
        if candidate:
            covered.append(candidate)
        if len(covered) >= MAX_CANDIDATES:
            break
    return covered


def _first_unique_group_candidate(
    names: list[str],
    candidates: list[MethodOption],
    covered: list[MethodOption],
) -> MethodOption | None:
    for name in names:
        candidate = _candidate_for_evidence(name, candidates)
        if candidate and not any(_same_name(candidate.name, item.name) for item in covered):
            return candidate
    return None


def _candidate_for_evidence(
    evidence_name: str,
    candidates: list[MethodOption],
) -> MethodOption | None:
    direct = next(
        (candidate for candidate in candidates if _same_name(candidate.name, evidence_name)),
        None,
    )
    if direct:
        return direct.model_copy(update={"name": evidence_name})
    referenced = next(
        (candidate for candidate in candidates if _reason_references(candidate, evidence_name)),
        None,
    )
    if not referenced:
        return None
    reason = f"所选证据及原判别理由明确支持{evidence_name}。"
    return referenced.model_copy(update={"name": evidence_name, "reason": reason})


def _reason_references(candidate: MethodOption, evidence_name: str) -> bool:
    reason = _normalize(candidate.reason)
    return _normalize(evidence_name) in reason or evidence_name in _method_names(candidate.reason)


def _fill_candidate_slots(
    seed: list[MethodOption],
    candidate_sets: list[list[MethodOption]],
) -> list[MethodOption]:
    combined = list(seed)
    for candidates in candidate_sets:
        for candidate in candidates:
            if len(combined) >= MAX_CANDIDATES:
                return combined
            if not any(_same_name(candidate.name, item.name) for item in combined):
                combined.append(candidate)
    return combined


def _same_name(left: str, right: str) -> bool:
    return _normalize(left) == _normalize(right)


def _ensure_unique_names(candidates: list[MethodOption]) -> None:
    names = [_normalize(candidate.name) for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError(
            "Method candidate normalization produced duplicate canonical names; "
            "return one independent candidate per evidence method."
        )


def _is_method_name(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in _METHOD_SUFFIXES)


def _clean_name(value: str) -> str:
    return canonical_method_name(value.strip(" ：:，,。；;"))


def _normalize(value: str) -> str:
    return "".join(value.split()).strip(" ：:，,。；;")
