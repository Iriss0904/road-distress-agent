"""National-standard specific numbering helpers."""

from __future__ import annotations

import re

from road_distress_agent.ingestion.content_window import content_lines
from road_distress_agent.ingestion.models import ParsedDocument

GENERIC_PROFILE = "generic"
NATIONAL_STANDARD_PROFILE = "national_standard"

_STANDARD_NAME_RE = re.compile(r"(规程|规范|标准|JTG|GB|CJJ)", re.IGNORECASE)
_ZERO_SECTION_RE = re.compile(r"^\s*\d+\.0\.\d+\s+")
_CLAUSE_SPACING_RE = re.compile(r"(?<=\d)\s*\.\s*(?=\d)")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def detect_document_profile(document: ParsedDocument) -> str:
    if _looks_like_national_standard(document):
        return NATIONAL_STANDARD_PROFILE
    return GENERIC_PROFILE


def normalize_clause_spacing(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    return _CLAUSE_SPACING_RE.sub(".", cleaned)


def national_heading_title_is_valid(raw_parts: tuple[int, ...], title: str) -> bool:
    if len(raw_parts) >= 3:
        return True
    return bool(_CJK_RE.search(title))


def _looks_like_national_standard(document: ParsedDocument) -> bool:
    name_signal = _STANDARD_NAME_RE.search(document.source_doc_id)
    path_signal = _STANDARD_NAME_RE.search(document.source_path)
    if not name_signal and not path_signal:
        return False
    return any(
        _ZERO_SECTION_RE.match(normalize_clause_spacing(line.normalized_text()))
        for line in content_lines(document.lines)
    )
