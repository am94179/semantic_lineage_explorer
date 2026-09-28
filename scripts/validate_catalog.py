"""Validate the canonical catalog against a local DuckDB warehouse."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from semantic_lineage.catalog.loader import CatalogLoadError, load_catalog
from semantic_lineage.catalog.schema_validation import (
    CatalogSchemaValidationError,
    validate_catalog_against_duckdb,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("metadata/tpcds_revenue_catalog.yaml"))
    parser.add_argument("--database", type=Path, default=Path("data/semantic_lineage.duckdb"))
    arguments = parser.parse_args()
    try:
        catalog = load_catalog(arguments.catalog)
        summary = validate_catalog_against_duckdb(catalog, arguments.database)
    except (CatalogLoadError, CatalogSchemaValidationError, OSError, duckdb.Error) as exc:
        print(f"Catalog validation failed: {exc}")
        return 1
    print(
        "Catalog validation passed: "
        f"{summary.assets_validated} assets, "
        f"{summary.columns_validated} columns, "
        f"{summary.join_conditions_validated} join conditions, "
        f"{summary.transformations_validated} transformations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
