from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from semantic_lineage.catalog.loader import load_catalog
from semantic_lineage.catalog.schema_validation import (
    CatalogSchemaValidationError,
    validate_catalog_against_duckdb,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / "metadata" / "tpcds_revenue_catalog.yaml"
WAREHOUSE_PATH = PROJECT_ROOT / "data" / "semantic_lineage.duckdb"


def test_catalog_validates_against_the_phase_one_warehouse() -> None:
    if not WAREHOUSE_PATH.is_file():
        pytest.skip("Phase 1 warehouse has not been generated")

    summary = validate_catalog_against_duckdb(
        load_catalog(CATALOG_PATH), WAREHOUSE_PATH, PROJECT_ROOT
    )

    assert summary.assets_validated == 6
    assert summary.columns_validated == 22
    assert summary.join_conditions_validated == 7
    assert summary.transformations_validated == 1


def test_schema_validation_rejects_a_missing_physical_column(tmp_path: Path) -> None:
    database_path = tmp_path / "warehouse.duckdb"
    connection = duckdb.connect(str(database_path))
    connection.execute("CREATE SCHEMA raw")
    connection.execute("CREATE TABLE raw.customer (c_customer_sk BIGINT)")
    connection.close()

    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        """
assets:
  - id: raw.customer
    schema: raw
    name: customer
    kind: source
    domain: customer
    description: Customer records.
    grain: One row per customer.
    primary_key: [c_customer_sk]
    concepts: [customer]
columns:
  - id: raw.customer.missing_column
    asset_id: raw.customer
    name: missing_column
    physical_type: BIGINT
    semantic_type: customer identifier
    role: key
    description: Missing customer key.
    concepts: [customer]
""",
        encoding="utf-8",
    )

    with pytest.raises(CatalogSchemaValidationError, match="does not exist physically"):
        validate_catalog_against_duckdb(load_catalog(catalog_path), database_path, tmp_path)
