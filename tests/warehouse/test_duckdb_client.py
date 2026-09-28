from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from semantic_lineage.warehouse.duckdb_client import (
    QueryExecutionError,
    QueryTimeoutError,
    execute_validated_query,
)
from semantic_lineage.warehouse.sql_safety import ValidatedSQL


def create_database(database_path: Path) -> Path:
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute("CREATE SCHEMA raw")
        connection.execute("CREATE TABLE raw.numbers (value BIGINT)")
        connection.execute("INSERT INTO raw.numbers VALUES (1), (2), (3)")
    finally:
        connection.close()
    return database_path


def test_execute_validated_query_returns_bounded_rows_from_worker_process(tmp_path: Path) -> None:
    database_path = create_database(tmp_path / "warehouse.duckdb")
    query = ValidatedSQL("SELECT value FROM raw.numbers ORDER BY value", ["raw.numbers"], 2)

    result = execute_validated_query(database_path, query, timeout_seconds=5)

    assert result.columns == ["value"]
    assert result.rows == [(1,), (2,)]
    assert result.row_limit == 2


def test_execute_validated_query_terminates_an_expensive_query_at_deadline(tmp_path: Path) -> None:
    database_path = create_database(tmp_path / "warehouse.duckdb")
    query = ValidatedSQL(
        "SELECT SUM(sin(a.value::DOUBLE + b.value)) FROM raw.numbers AS a CROSS JOIN range(1000000000) AS b(value)",
        ["raw.numbers"],
        1,
    )

    with pytest.raises(QueryTimeoutError, match="execution limit"):
        execute_validated_query(database_path, query, timeout_seconds=0.1)


def test_execute_validated_query_reports_duckdb_errors_without_worker_traceback(
    tmp_path: Path,
) -> None:
    database_path = create_database(tmp_path / "warehouse.duckdb")
    query = ValidatedSQL("SELECT missing_column FROM raw.numbers", ["raw.numbers"], 1)

    with pytest.raises(QueryExecutionError, match="DuckDB query failed"):
        execute_validated_query(database_path, query, timeout_seconds=5)


def test_execute_validated_query_rejects_non_positive_timeout(tmp_path: Path) -> None:
    database_path = create_database(tmp_path / "warehouse.duckdb")
    query = ValidatedSQL("SELECT value FROM raw.numbers", ["raw.numbers"], 1)

    with pytest.raises(ValueError, match="greater than zero"):
        execute_validated_query(database_path, query, timeout_seconds=0)
