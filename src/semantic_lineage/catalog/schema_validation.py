"""Validate canonical catalog physical references against a DuckDB warehouse."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import duckdb

from semantic_lineage.models import Catalog


class CatalogSchemaValidationError(ValueError):
    """Raised when catalog metadata disagrees with the physical warehouse."""


@dataclass(frozen=True, slots=True)
class CatalogSchemaValidationSummary:
    assets_validated: int
    columns_validated: int
    join_conditions_validated: int
    transformations_validated: int


_REFERENCE_PATTERN = re.compile(
    r"^(?P<schema>[A-Za-z_][A-Za-z0-9_]*)\.(?P<table>[A-Za-z_][A-Za-z0-9_]*)\."
    r"(?P<column>[A-Za-z_][A-Za-z0-9_]*)$"
)


def validate_catalog_against_duckdb(
    catalog: Catalog, database_path: str | Path, project_root: str | Path = "."
) -> CatalogSchemaValidationSummary:
    """Validate catalog assets, columns, joins, and SQL paths against DuckDB and disk."""
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        _validate_assets(connection, catalog)
        _validate_columns(connection, catalog)
        join_count = _validate_join_conditions(connection, catalog)
    finally:
        connection.close()
    transformation_count = _validate_transformation_paths(catalog, Path(project_root))
    return CatalogSchemaValidationSummary(
        assets_validated=len(catalog.assets),
        columns_validated=len(catalog.columns),
        join_conditions_validated=join_count,
        transformations_validated=transformation_count,
    )


def _validate_assets(connection: duckdb.DuckDBPyConnection, catalog: Catalog) -> None:
    for asset in catalog.assets:
        exists = connection.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = ? AND table_name = ?
            """,
            [asset.schema_name, asset.name],
        ).fetchone()
        if exists is None:
            raise CatalogSchemaValidationError(
                f"catalog asset '{asset.id}' does not exist as {asset.schema_name}.{asset.name}"
            )


def _validate_columns(connection: duckdb.DuckDBPyConnection, catalog: Catalog) -> None:
    assets_by_id = {asset.id: asset for asset in catalog.assets}
    for column in catalog.columns:
        asset = assets_by_id[column.asset_id]
        actual_types = {
            row[0]: _normalise_type(row[1])
            for row in connection.execute(
                f"DESCRIBE {_quote_identifier(asset.schema_name)}.{_quote_identifier(asset.name)}"
            ).fetchall()
        }
        actual_type = actual_types.get(column.name)
        if actual_type is None:
            raise CatalogSchemaValidationError(
                f"catalog column '{column.id}' does not exist physically"
            )
        expected_type = _normalise_type(column.physical_type)
        if actual_type != expected_type:
            raise CatalogSchemaValidationError(
                f"catalog column '{column.id}' type is {expected_type}, but DuckDB has {actual_type}"
            )


def _validate_join_conditions(connection: duckdb.DuckDBPyConnection, catalog: Catalog) -> int:
    count = 0
    for relationship in catalog.relationships:
        for condition in relationship.join_conditions:
            left, separator, right = condition.partition("=")
            if not separator:
                raise CatalogSchemaValidationError(
                    f"relationship '{relationship.id}' has invalid join condition '{condition}'"
                )
            _validate_physical_reference(connection, left.strip(), relationship.id)
            _validate_physical_reference(connection, right.strip(), relationship.id)
            count += 1
    return count


def _validate_physical_reference(
    connection: duckdb.DuckDBPyConnection, reference: str, relationship_id: str
) -> None:
    matched = _REFERENCE_PATTERN.fullmatch(reference)
    if matched is None:
        raise CatalogSchemaValidationError(
            f"relationship '{relationship_id}' has unsupported reference '{reference}'"
        )
    schema, table, column = matched.group("schema", "table", "column")
    columns = {
        row[0]
        for row in connection.execute(
            f"DESCRIBE {_quote_identifier(schema)}.{_quote_identifier(table)}"
        ).fetchall()
    }
    if column not in columns:
        raise CatalogSchemaValidationError(
            f"relationship '{relationship_id}' references missing physical column '{reference}'"
        )


def _validate_transformation_paths(catalog: Catalog, project_root: Path) -> int:
    for transformation in catalog.transformations:
        if (
            transformation.sql_path is not None
            and not (project_root / transformation.sql_path).is_file()
        ):
            raise CatalogSchemaValidationError(
                f"transformation '{transformation.id}' SQL file does not exist: {transformation.sql_path}"
            )
    return len(catalog.transformations)


def _normalise_type(type_name: str) -> str:
    return re.sub(r"\s+", "", type_name).upper()


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
