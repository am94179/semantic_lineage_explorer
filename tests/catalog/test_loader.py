from __future__ import annotations

from pathlib import Path

import pytest

from semantic_lineage.catalog.loader import CatalogLoadError, load_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_loads_valid_example_catalog() -> None:
    catalog = load_catalog(PROJECT_ROOT / "metadata" / "example_catalog.yaml")

    assert len(catalog.assets) == 2
    assert catalog.metrics[0].output_column_id == "analytics.customer_revenue_yearly.total_net_paid"


def test_rejects_unknown_asset_reference(tmp_path: Path) -> None:
    invalid_catalog = tmp_path / "invalid_catalog.yaml"
    invalid_catalog.write_text(
        """
columns:
  - id: example.customer.customer_id
    asset_id: absent.customer
    name: customer_id
    physical_type: BIGINT
    semantic_type: customer identifier
    role: key
    description: Customer identifier.
    concepts: [customer]
""",
        encoding="utf-8",
    )

    with pytest.raises(CatalogLoadError, match="unknown ID 'absent.customer'"):
        load_catalog(invalid_catalog)


def test_rejects_transformation_with_both_sql_forms(tmp_path: Path) -> None:
    invalid_catalog = tmp_path / "invalid_catalog.yaml"
    invalid_catalog.write_text(
        """
assets:
  - id: example.asset
    schema: example
    name: asset
    kind: source
    domain: example
    description: Example asset.
    grain: One row per entity.
    primary_key: [id]
    concepts: [entity]
transformations:
  - id: invalid_transform
    target_asset_id: example.asset
    input_asset_ids: [example.asset]
    sql_path: sql/example.sql
    sql: SELECT 1
    description: Invalid example.
    output_grain: One row per entity.
""",
        encoding="utf-8",
    )

    with pytest.raises(CatalogLoadError, match="exactly one of sql_path or sql"):
        load_catalog(invalid_catalog)
