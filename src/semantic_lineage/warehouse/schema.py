"""Physical-schema inspection restricted to catalog-approved DuckDB tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from semantic_lineage.models import Catalog


class SchemaInspectionError(ValueError):
    """Raised when a requested table is not a catalog-approved physical table."""


@dataclass(frozen=True, slots=True)
class SchemaColumn:
    name: str
    physical_type: str


@dataclass(frozen=True, slots=True)
class TableSchema:
    qualified_name: str
    columns: list[SchemaColumn]


def catalog_table_names(catalog: Catalog, asset_ids: list[str] | None = None) -> list[str]:
    """Return approved physical tables, optionally limited to selected catalog assets."""
    selected = set(asset_ids) if asset_ids else None
    return [
        f"{asset.schema_name}.{asset.name}"
        for asset in catalog.assets
        if selected is None or asset.id in selected
    ]


def inspect_schema(
    database_path: str | Path, catalog: Catalog, asset_ids: list[str] | None = None
) -> list[TableSchema]:
    """Inspect catalog-approved tables from DuckDB in read-only mode."""
    tables = catalog_table_names(catalog, asset_ids)
    if not tables:
        raise SchemaInspectionError("no catalog-approved tables were selected for inspection")
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        snapshots: list[TableSchema] = []
        for qualified_name in tables:
            schema_name, table_name = qualified_name.split(".", maxsplit=1)
            rows = connection.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = ? AND table_name = ?
                ORDER BY ordinal_position
                """,
                [schema_name, table_name],
            ).fetchall()
            if not rows:
                raise SchemaInspectionError(
                    f"catalog table is absent from DuckDB: {qualified_name}"
                )
            snapshots.append(
                TableSchema(
                    qualified_name=qualified_name,
                    columns=[
                        SchemaColumn(name=name, physical_type=data_type) for name, data_type in rows
                    ],
                )
            )
        return snapshots
    finally:
        connection.close()
