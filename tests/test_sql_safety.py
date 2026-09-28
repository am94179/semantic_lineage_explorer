from __future__ import annotations

import pytest

from semantic_lineage.warehouse.sql_safety import SQLSafetyError, validate_read_only_sql

ALLOWED_TABLES = ["raw.customer", "analytics.customer_revenue_yearly"]
ALLOWED_COLUMNS = {
    "raw.customer": {"c_customer_sk", "c_customer_id", "c_first_name", "c_last_name", "shared_id"},
    "analytics.customer_revenue_yearly": {
        "customer_sk",
        "calendar_year",
        "total_net_paid",
        "shared_id",
    },
}


def validate(sql: str, *, max_rows: int = 100):
    return validate_read_only_sql(
        sql, ALLOWED_TABLES, allowed_columns=ALLOWED_COLUMNS, max_rows=max_rows
    )


def test_validator_adds_a_bounded_limit_to_an_allowed_select() -> None:
    query = validate("SELECT c_customer_id FROM raw.customer")

    assert query.referenced_tables == ["raw.customer"]
    assert query.sql.endswith("LIMIT 100")


def test_validator_accepts_aliases_ctes_set_operations_and_literal_offset() -> None:
    query = validate(
        "WITH customers AS (SELECT c_customer_id AS customer_id FROM raw.customer) "
        "SELECT customer_id FROM customers UNION "
        "SELECT c_customer_id FROM raw.customer ORDER BY customer_id LIMIT 5 OFFSET 1"
    )

    assert query.referenced_tables == ["raw.customer"]
    assert "WITH customers AS" in query.sql
    assert "LIMIT 5 OFFSET 1" in query.sql


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("DELETE FROM raw.customer", "only SELECT"),
        ("SELECT * FROM raw.customer; SELECT * FROM raw.customer", "exactly one"),
        ("SELECT * FROM raw.store_sales", "not allowed"),
        ("SELECT * FROM customer", "schema-qualified"),
        ("SELECT * FROM read_csv_auto('outside.csv')", "table-valued functions"),
        ("SELECT * FROM range(10)", "table-valued functions"),
        (
            "WITH RECURSIVE c AS (SELECT c_customer_id FROM raw.customer) SELECT * FROM c",
            "recursive CTEs",
        ),
    ],
)
def test_validator_rejects_unsafe_or_unapproved_sources(sql: str, message: str) -> None:
    with pytest.raises(SQLSafetyError, match=message):
        validate(sql)


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("SELECT unknown_column FROM raw.customer", "not present"),
        ("SELECT c.unknown_column FROM raw.customer AS c", "raw.customer.unknown_column"),
        (
            "SELECT shared_id FROM raw.customer JOIN analytics.customer_revenue_yearly "
            "ON raw.customer.shared_id = analytics.customer_revenue_yearly.shared_id",
            "ambiguous",
        ),
        ("SELECT missing_column FROM raw.customer", "not present"),
    ],
)
def test_validator_rejects_unknown_or_ambiguous_columns(sql: str, message: str) -> None:
    with pytest.raises(SQLSafetyError, match=message):
        validate(sql)


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("SELECT * FROM raw.customer LIMIT ?", "integer literal"),
        ("SELECT * FROM raw.customer LIMIT -1", "must not be negative"),
        ("SELECT * FROM raw.customer LIMIT 1.5", "integer literal"),
        ("SELECT * FROM raw.customer OFFSET -1", "must not be negative"),
    ],
)
def test_validator_rejects_malformed_bounds(sql: str, message: str) -> None:
    with pytest.raises(SQLSafetyError, match=message):
        validate(sql)


def test_validator_clamps_an_excessive_limit() -> None:
    query = validate("SELECT * FROM raw.customer LIMIT 1000", max_rows=25)

    assert query.sql.endswith("LIMIT 25")


def test_validator_requires_schema_metadata_for_each_allowed_table() -> None:
    with pytest.raises(SQLSafetyError, match="missing column metadata"):
        validate_read_only_sql(
            "SELECT c_customer_id FROM raw.customer",
            ALLOWED_TABLES,
            allowed_columns={"raw.customer": {"c_customer_id"}},
        )
