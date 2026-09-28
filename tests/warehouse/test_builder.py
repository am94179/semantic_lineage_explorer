from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from semantic_lineage.warehouse.builder import (
    WarehouseBuildError,
    materialize_customer_revenue_yearly,
    validate_customer_revenue_yearly,
    validate_source_schema,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSFORMATION_SQL = PROJECT_ROOT / "sql" / "customer_revenue_yearly.sql"


@pytest.fixture
def warehouse_connection() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    connection.execute("CREATE SCHEMA raw")
    connection.execute("CREATE SCHEMA analytics")
    connection.execute("CREATE TABLE raw.customer (c_customer_sk BIGINT)")
    connection.execute("CREATE TABLE raw.date_dim (d_date_sk BIGINT, d_year INTEGER)")
    connection.execute(
        "CREATE TABLE raw.store_sales (ss_sold_date_sk BIGINT, ss_customer_sk BIGINT, ss_net_paid DECIMAL(10,2))"
    )
    connection.execute(
        "CREATE TABLE raw.catalog_sales (cs_sold_date_sk BIGINT, cs_bill_customer_sk BIGINT, cs_net_paid DECIMAL(10,2))"
    )
    connection.execute(
        "CREATE TABLE raw.web_sales (ws_sold_date_sk BIGINT, ws_bill_customer_sk BIGINT, ws_net_paid DECIMAL(10,2))"
    )
    connection.execute("INSERT INTO raw.customer VALUES (1), (2)")
    connection.execute("INSERT INTO raw.date_dim VALUES (10, 2001), (20, 2002)")
    connection.execute("INSERT INTO raw.store_sales VALUES (10, 1, 100.00), (20, 2, 50.00)")
    connection.execute("INSERT INTO raw.catalog_sales VALUES (10, 1, 25.00)")
    connection.execute("INSERT INTO raw.web_sales VALUES (10, 1, 10.00), (20, NULL, 20.00)")
    yield connection
    connection.close()


def test_materialized_customer_revenue_reconciles(
    warehouse_connection: duckdb.DuckDBPyConnection,
) -> None:
    validate_source_schema(warehouse_connection)
    materialize_customer_revenue_yearly(warehouse_connection, TRANSFORMATION_SQL)
    validate_customer_revenue_yearly(warehouse_connection)
    rows = warehouse_connection.execute(
        "SELECT customer_sk, calendar_year, store_net_paid, catalog_net_paid, web_net_paid, total_net_paid FROM analytics.customer_revenue_yearly ORDER BY customer_sk, calendar_year"
    ).fetchall()
    assert rows == [(1, 2001, 100, 25, 10, 135), (2, 2002, 50, 0, 0, 50)]


def test_source_schema_validation_rejects_missing_column(
    warehouse_connection: duckdb.DuckDBPyConnection,
) -> None:
    warehouse_connection.execute("ALTER TABLE raw.web_sales DROP ws_net_paid")
    with pytest.raises(WarehouseBuildError, match="raw.web_sales is missing required columns"):
        validate_source_schema(warehouse_connection)


def test_revenue_validation_rejects_unreconciled_channel_totals(
    warehouse_connection: duckdb.DuckDBPyConnection,
) -> None:
    materialize_customer_revenue_yearly(warehouse_connection, TRANSFORMATION_SQL)
    warehouse_connection.execute(
        "UPDATE analytics.customer_revenue_yearly SET total_net_paid = 0 WHERE customer_sk = 1"
    )
    with pytest.raises(WarehouseBuildError, match="channel totals do not reconcile"):
        validate_customer_revenue_yearly(warehouse_connection)
