"""Generate retrieval-ready semantic documents from the canonical catalog."""

from __future__ import annotations

from dataclasses import dataclass

from semantic_lineage.models import Catalog


@dataclass(frozen=True, slots=True)
class SemanticDocument:
    id: str
    document_type: str
    content: str
    payload: dict[str, str | list[str]]


def build_semantic_documents(catalog: Catalog) -> list[SemanticDocument]:
    """Create deterministic documents suitable for a future Qdrant indexing phase."""
    assets = {asset.id: asset for asset in catalog.assets}
    documents: list[SemanticDocument] = []
    for asset in catalog.assets:
        asset_columns = [column.name for column in catalog.columns if column.asset_id == asset.id]
        documents.append(
            SemanticDocument(
                id=f"asset:{asset.id}",
                document_type="asset",
                content=(
                    f"Dataset: {asset.schema_name}.{asset.name}\nDomain: {asset.domain}\n"
                    f"Description: {asset.description}\nGrain: {asset.grain}\n"
                    f"Concepts: {', '.join(asset.concepts)}\nColumns: {', '.join(asset_columns)}"
                ),
                payload={
                    "asset_id": asset.id,
                    "schema_name": asset.schema_name,
                    "table_name": asset.name,
                    "domain": asset.domain,
                    "tags": asset.tags,
                },
            )
        )
    for column in catalog.columns:
        asset = assets[column.asset_id]
        documents.append(
            SemanticDocument(
                id=f"column:{column.id}",
                document_type="column",
                content=(
                    f"Column: {asset.schema_name}.{asset.name}.{column.name}\n"
                    f"Description: {column.description}\nSemantic type: {column.semantic_type}\n"
                    f"Business concepts: {', '.join(column.concepts)}\n"
                    f"Synonyms: {', '.join(column.synonyms)}"
                ),
                payload={
                    "asset_id": asset.id,
                    "schema_name": asset.schema_name,
                    "table_name": asset.name,
                    "column_name": column.name,
                    "domain": asset.domain,
                    "tags": asset.tags,
                },
            )
        )
    for relationship in catalog.relationships:
        documents.append(
            SemanticDocument(
                id=f"relationship:{relationship.id}",
                document_type="relationship",
                content=(
                    f"Relationship: {relationship.id}\n"
                    f"Join: {'; '.join(relationship.join_conditions)}\n"
                    f"Cardinality: {relationship.cardinality}\nEvidence: {relationship.evidence}"
                ),
                payload={
                    "source_asset_id": relationship.source_asset_id,
                    "target_asset_id": relationship.target_asset_id,
                    "relationship_type": str(relationship.relationship_type),
                },
            )
        )
    for transformation in catalog.transformations:
        documents.append(
            SemanticDocument(
                id=f"transformation:{transformation.id}",
                document_type="transformation",
                content=(
                    f"Transformation: {transformation.id}\nTarget: {transformation.target_asset_id}\n"
                    f"Inputs: {', '.join(transformation.input_asset_ids)}\n"
                    f"Description: {transformation.description}\nOutput grain: {transformation.output_grain}"
                ),
                payload={
                    "target_asset_id": transformation.target_asset_id,
                    "input_asset_ids": transformation.input_asset_ids,
                    "sql_path": transformation.sql_path or "",
                },
            )
        )
    for edge in catalog.lineage_edges:
        documents.append(
            SemanticDocument(
                id=f"lineage:{edge.id}",
                document_type="lineage",
                content=(
                    f"Lineage: {edge.source_column_id} -> {edge.target_column_id}\n"
                    f"Transformation: {edge.transformation_id}\nOperation: {edge.operation}"
                ),
                payload={
                    "source_column_id": edge.source_column_id,
                    "target_column_id": edge.target_column_id,
                    "transformation_id": edge.transformation_id,
                },
            )
        )
    for metric in catalog.metrics:
        documents.append(
            SemanticDocument(
                id=f"metric:{metric.id}",
                document_type="metric",
                content=(
                    f"Metric: {metric.name}\nDescription: {metric.description}\n"
                    f"Definition: {metric.definition}\nGrain: {metric.grain}\n"
                    f"Concepts: {', '.join(metric.concepts)}\nSources: {', '.join(metric.source_asset_ids)}"
                ),
                payload={
                    "metric_id": metric.id,
                    "output_column_id": metric.output_column_id,
                    "source_asset_ids": metric.source_asset_ids,
                    "concepts": metric.concepts,
                },
            )
        )
    return documents
