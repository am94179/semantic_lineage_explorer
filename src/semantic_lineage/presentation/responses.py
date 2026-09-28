"""Render deterministic catalog exploration responses for the Phase 4 CLI."""

from __future__ import annotations

from semantic_lineage.exploration import DiscoveryResponse, LineageResponse


def render_discovery_response(response: DiscoveryResponse) -> str:
    """Render semantic discovery evidence without LLM synthesis."""
    lines = ["Semantic discovery", f"Question: {response.question}", "", "Relevant data assets:"]
    lines.extend(_render_assets(response.assets))
    lines.extend(["", "Related relationships:"])
    lines.extend(_render_relationships(response.relationships))
    lines.extend(["", "Retrieved semantic evidence:"])
    lines.extend(_render_evidence(response.evidence))
    return "\n".join(lines)


def render_lineage_response(response: LineageResponse) -> str:
    """Render explicit lineage edges and the semantic evidence used to resolve the target."""
    lines = ["Data lineage", f"Request: {response.request}"]
    if response.target_metric is not None:
        lines.extend(
            [
                f"Metric: {response.target_metric.name}",
                f"Definition: {response.target_metric.definition}",
            ]
        )
    lines.extend([f"Target column: {response.target_column_id}", "", "Explicit upstream lineage:"])
    if response.upstream_edges:
        lines.extend(
            f"- `{edge.source_column_id}` → `{edge.target_column_id}` ({edge.operation})"
            for edge in response.upstream_edges
        )
    else:
        lines.append("- No upstream lineage edges are recorded for this column.")
    lines.extend(["", "Data assets involved:"])
    lines.extend(_render_assets(response.assets))
    lines.extend(["", "Related relationships:"])
    lines.extend(_render_relationships(response.relationships))
    lines.extend(["", "Retrieved semantic evidence:"])
    lines.extend(_render_evidence(response.evidence))
    return "\n".join(lines)


def _render_assets(assets: list[object]) -> list[str]:
    if not assets:
        return ["- No catalog assets were resolved from the retrieved evidence."]
    return [
        f"- `{asset.qualified_name}` — {asset.description} (grain: {asset.grain})"
        for asset in assets
    ]


def _render_relationships(relationships: list[object]) -> list[str]:
    if not relationships:
        return ["- No related catalog relationships were resolved."]
    return [
        f"- `{relationship.id}`: {'; '.join(relationship.join_conditions)}"
        for relationship in relationships
    ]


def _render_evidence(evidence: list[object]) -> list[str]:
    if not evidence:
        return ["- No semantic documents were retrieved."]
    return [
        f"- [{result.score:.3f}] `{result.document_type}` `{result.document_id}`"
        for result in evidence
    ]
