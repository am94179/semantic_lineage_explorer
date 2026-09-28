"""Deterministic traversal of explicit column-level lineage."""

from __future__ import annotations

from collections import deque

from semantic_lineage.models import Catalog, LineageEdge


def get_upstream_lineage(
    catalog: Catalog, target_column_id: str, max_depth: int = 5
) -> list[LineageEdge]:
    """Return explicit lineage edges that contribute to a target column."""
    return _traverse(catalog, target_column_id, max_depth, direction="upstream")


def get_downstream_lineage(
    catalog: Catalog, source_column_id: str, max_depth: int = 5
) -> list[LineageEdge]:
    """Return explicit lineage edges that depend on a source column."""
    return _traverse(catalog, source_column_id, max_depth, direction="downstream")


def _traverse(
    catalog: Catalog, start_column_id: str, max_depth: int, direction: str
) -> list[LineageEdge]:
    if max_depth < 1:
        return []
    if direction == "upstream":

        def matching(edge: LineageEdge, column_id: str) -> bool:
            return edge.target_column_id == column_id

        def next_column(edge: LineageEdge) -> str:
            return edge.source_column_id
    else:

        def matching(edge: LineageEdge, column_id: str) -> bool:
            return edge.source_column_id == column_id

        def next_column(edge: LineageEdge) -> str:
            return edge.target_column_id

    result: list[LineageEdge] = []
    visited_edges: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(start_column_id, 0)])
    while queue:
        column_id, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for edge in catalog.lineage_edges:
            if matching(edge, column_id) and edge.id not in visited_edges:
                visited_edges.add(edge.id)
                result.append(edge)
                queue.append((next_column(edge), depth + 1))
    return result
