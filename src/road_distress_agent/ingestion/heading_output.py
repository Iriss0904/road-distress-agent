"""Serialization helpers for heading topology inspection."""

from __future__ import annotations

from dataclasses import asdict

from road_distress_agent.ingestion.models import HeadingCandidate, HeadingNode


def heading_tree_to_text(nodes: list[HeadingNode]) -> str:
    lines: list[str] = []
    _append_nodes(lines, nodes, depth=0)
    return "\n".join(lines)


def _append_nodes(lines: list[str], nodes: list[HeadingNode], depth: int) -> None:
    for node in nodes:
        lines.append(_node_line(node, depth))
        _append_nodes(lines, node.children, depth + 1)


def _node_line(node: HeadingNode, depth: int) -> str:
    marker = "  " * depth + "- "
    raw = "" if node.raw_clause_id == node.clause_id else f" [raw {node.raw_clause_id}]"
    anomaly = "" if not node.anomaly_type else f" !{node.anomaly_type}"
    title = f" {node.title}" if node.title else ""
    return f"{marker}{node.clause_id}{title} (PDF p.{node.page_number}){raw}{anomaly}"


def headings_as_dicts(headings: list[HeadingCandidate]) -> list[dict[str, object]]:
    return [asdict(heading) for heading in headings]
