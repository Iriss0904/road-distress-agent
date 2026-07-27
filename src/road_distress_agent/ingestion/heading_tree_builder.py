"""Build heading tree nodes from ordered heading candidates."""

from __future__ import annotations

from road_distress_agent.ingestion.models import HeadingCandidate, HeadingNode


def build_tree(headings: list[HeadingCandidate]) -> list[HeadingNode]:
    roots: list[HeadingNode] = []
    stack: dict[int, HeadingNode] = {}
    for heading in headings:
        node = _node_from_heading(heading)
        for level in list(stack):
            if level >= node.level:
                del stack[level]
        parent = stack.get(node.level - 1)
        if parent:
            parent.children.append(node)
        else:
            roots.append(node)
        stack[node.level] = node
    return roots


def _node_from_heading(heading: HeadingCandidate) -> HeadingNode:
    return HeadingNode(
        clause_id=heading.canonical_clause_id,
        title=heading.title,
        level=heading.level,
        page_number=heading.page_number,
        raw_clause_id=heading.raw_clause_id,
        aliases=heading.clause_aliases,
        anomaly_type=heading.anomaly_type,
        anomaly_reason=heading.anomaly_reason,
    )
