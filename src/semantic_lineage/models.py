"""Typed, source-controlled semantic catalog contracts.

These models are the canonical metadata boundary. Qdrant documents will be derived from these
records in a later phase; they are not the source of truth themselves.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Reject misspelled fields in catalog files instead of silently ignoring them."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, str_strip_whitespace=True)


class AssetKind(StrEnum):
    SOURCE = "source"
    DERIVED = "derived"


class ColumnRole(StrEnum):
    KEY = "key"
    DIMENSION = "dimension"
    MEASURE = "measure"


class RelationshipType(StrEnum):
    FOREIGN_KEY = "foreign_key"
    SEMANTIC = "semantic"


class Cardinality(StrEnum):
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class DataAsset(StrictModel):
    id: str = Field(min_length=1)
    schema_name: str = Field(min_length=1, alias="schema", serialization_alias="schema")
    name: str = Field(min_length=1)
    kind: AssetKind
    domain: str = Field(min_length=1)
    description: str = Field(min_length=1)
    grain: str = Field(min_length=1)
    primary_key: list[str] = Field(min_length=1)
    concepts: list[str] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class Column(StrictModel):
    id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    physical_type: str = Field(min_length=1)
    semantic_type: str = Field(min_length=1)
    role: ColumnRole
    description: str = Field(min_length=1)
    concepts: list[str] = Field(min_length=1)
    synonyms: list[str] = Field(default_factory=list)


class Relationship(StrictModel):
    id: str = Field(min_length=1)
    source_asset_id: str = Field(min_length=1)
    target_asset_id: str = Field(min_length=1)
    join_conditions: list[str] = Field(min_length=1)
    cardinality: Cardinality
    relationship_type: RelationshipType
    evidence: str = Field(min_length=1)


class Transformation(StrictModel):
    id: str = Field(min_length=1)
    target_asset_id: str = Field(min_length=1)
    input_asset_ids: list[str] = Field(min_length=1)
    sql_path: str | None = None
    sql: str | None = None
    description: str = Field(min_length=1)
    output_grain: str = Field(min_length=1)

    @model_validator(mode="after")
    def has_exactly_one_sql_definition(self) -> Transformation:
        if (self.sql_path is None) == (self.sql is None):
            raise ValueError("exactly one of sql_path or sql must be provided")
        return self


class LineageEdge(StrictModel):
    id: str = Field(min_length=1)
    source_column_id: str = Field(min_length=1)
    target_column_id: str = Field(min_length=1)
    transformation_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)


class Metric(StrictModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    output_column_id: str = Field(min_length=1)
    grain: str = Field(min_length=1)
    source_asset_ids: list[str] = Field(min_length=1)
    concepts: list[str] = Field(min_length=1)


class Catalog(StrictModel):
    """A complete catalog snapshot that can be validated before indexing or use."""

    assets: list[DataAsset] = Field(default_factory=list)
    columns: list[Column] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    transformations: list[Transformation] = Field(default_factory=list)
    lineage_edges: list[LineageEdge] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)

    @model_validator(mode="after")
    def validates_references_and_ids(self) -> Catalog:
        self._ensure_unique_ids()

        asset_ids = {asset.id for asset in self.assets}
        column_ids = {column.id for column in self.columns}
        transformation_ids = {transformation.id for transformation in self.transformations}

        for column in self.columns:
            self._require_reference(column.asset_id, asset_ids, f"column '{column.id}' asset_id")
        for relationship in self.relationships:
            self._require_reference(
                relationship.source_asset_id,
                asset_ids,
                f"relationship '{relationship.id}' source_asset_id",
            )
            self._require_reference(
                relationship.target_asset_id,
                asset_ids,
                f"relationship '{relationship.id}' target_asset_id",
            )
        for transformation in self.transformations:
            self._require_reference(
                transformation.target_asset_id,
                asset_ids,
                f"transformation '{transformation.id}' target_asset_id",
            )
            for input_asset_id in transformation.input_asset_ids:
                self._require_reference(
                    input_asset_id,
                    asset_ids,
                    f"transformation '{transformation.id}' input_asset_ids",
                )
        for edge in self.lineage_edges:
            self._require_reference(
                edge.source_column_id, column_ids, f"lineage edge '{edge.id}' source_column_id"
            )
            self._require_reference(
                edge.target_column_id, column_ids, f"lineage edge '{edge.id}' target_column_id"
            )
            self._require_reference(
                edge.transformation_id,
                transformation_ids,
                f"lineage edge '{edge.id}' transformation_id",
            )
        for metric in self.metrics:
            self._require_reference(
                metric.output_column_id, column_ids, f"metric '{metric.id}' output_column_id"
            )
            for source_asset_id in metric.source_asset_ids:
                self._require_reference(
                    source_asset_id, asset_ids, f"metric '{metric.id}' source_asset_ids"
                )
        return self

    def _ensure_unique_ids(self) -> None:
        entity_groups = {
            "asset": [asset.id for asset in self.assets],
            "column": [column.id for column in self.columns],
            "relationship": [relationship.id for relationship in self.relationships],
            "transformation": [transformation.id for transformation in self.transformations],
            "lineage edge": [edge.id for edge in self.lineage_edges],
            "metric": [metric.id for metric in self.metrics],
        }
        for entity_name, ids in entity_groups.items():
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {entity_name} IDs are not allowed")

    @staticmethod
    def _require_reference(reference: str, valid_ids: set[str], field_name: str) -> None:
        if reference not in valid_ids:
            raise ValueError(f"{field_name} references unknown ID '{reference}'")
