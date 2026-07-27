"""Table caption parsing helpers for ingestion chunks."""

from __future__ import annotations

import re

from road_distress_agent.ingestion.national_standard import normalize_clause_spacing

TABLE_LAYOUT = "table"
TABLE_CAPTION_RE = re.compile(r"^\s*(?:续)?表\s*[0-9A-ZＡ-Ｚ].{0,90}$")
TABLE_LABEL_RE = re.compile(
    r"^\s*(?:续)?表\s*(?P<label>\d+(?:\.\d+){0,5}(?:-\d+)?|[A-ZＡ-Ｚ](?:[.-]\d+)?)(?=\s|$|[\u4e00-\u9fff])"
)
BARE_TABLE_LABEL_RE = re.compile(r"^\s*(?:续)?(?P<label>\d+(?:\.\d+){1,5}-\d+)(?=\s|$)")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def is_table_caption(text: str) -> bool:
    caption = normalise_table_caption(text)
    if TABLE_CAPTION_RE.match(caption):
        return True
    match = BARE_TABLE_LABEL_RE.match(caption)
    return bool(match and _CJK_RE.search(caption[match.end() :]))


def is_continued_caption(caption: str) -> bool:
    return caption.lstrip().startswith("续表")


def primary_caption(captions: list[str]) -> str:
    for caption in captions:
        if caption and not is_continued_caption(caption):
            return caption
    return captions[0] if captions else ""


def table_label(caption: str) -> str | None:
    normalised = normalise_table_caption(caption)
    match = TABLE_LABEL_RE.match(normalised) or BARE_TABLE_LABEL_RE.match(normalised)
    if not match:
        return None
    return f"表{normalise_table_label(match.group('label'))}"


def table_clause_id(caption: str) -> str | None:
    label = table_label(caption)
    if not label:
        return None
    return label.removeprefix("表")


def normalise_table_label(label: str) -> str:
    compact = normalise_table_caption(label)
    compact = compact.replace(" ", "").replace("．", ".").replace("－", "-")
    appendix = re.fullmatch(r"(?P<letter>[A-ZＡ-Ｚ])[.-](?P<number>\d+)", compact)
    if appendix:
        return f"{_normalise_latin(appendix.group('letter'))}-{appendix.group('number')}"
    return compact


def normalise_table_caption(caption: str) -> str:
    compact = normalize_clause_spacing(caption)
    compact = re.sub(r"(?<=\d)\s*[－-]\s*(?=\d)", "-", compact)
    return re.sub(r"(?<=[A-ZＡ-Ｚ])\s*[－-]\s*(?=\d)", "-", compact)


def _normalise_latin(value: str) -> str:
    char = value.strip().upper()
    if "Ａ" <= char <= "Ｚ":
        return chr(ord(char) - ord("Ａ") + ord("A"))
    return char
