"""Build the reproducible TPC-DS warehouse used by initial implementation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb

DEFAULT_SCALE_FACTOR = 0.03
DEFAULT_DATABASE_PATH = Path("data/semantic_lineage.duckdb")
DEFAULT_MANIFEST_PATH = Path("data/manifests/tpcds_generation.json")
TRANSFORMATION_SQL_PATH = Path("sql/customer_revenue_yearly.sql")

REQUIRED_SOURCE_COLUMNS: dict[str, set[str]] = {
    "customer": {"c_customer_sk"},
    "date_dim": {"d_date_sk", "d_year"},
    "store_sales": {"ss_sold_date_sk", "ss_customer_sk", "ss_net_paid"},
    "catalog_sales": {"cs_sold_date_sk", "cs_bill_customer_sk", "cs_net_paid"},
    "web_sales": {"ws_sold_date_sk", "ws_bill_customer_sk", "ws_net_paid"},
}
REVENUE_COLUMNS = {
    "customer_sk",
    "calendar_year",
    "store_net_paid",
    "catalog_net_paid",
    "web_net_paid",
    "total_net_paid",
}


class WarehouseBuildError(RuntimeError):
    """Raised when the generated warehouse does not meet the initial implementation contract."""


def build_warehouse(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    transformation_sql_path: str | Path = TRANSFORMATION_SQL_PATH,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Generate TPC-DS data, materialize yearly revenue, and write a provenance manifest."""
    if scale_factor <= 0:
        raise WarehouseBuildError("scale_factor must be greater than zero")
    database, manifest, transformation = (
        Path(database_path),
        Path(manifest_path),
        Path(transformation_sql_path),
    )
    if database.exists() and not overwrite:
        raise WarehouseBuildError(
            f"database already exists at '{database}'. Re-run with --force to replace it."
        )
    if not transformation.is_file():
        raise WarehouseBuildError(f"transformation SQL file does not exist: '{transformation}'")
    database.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if database.exists():
        database.unlink()
    connection = duckdb.connect(str(database))
    try:
        connection.execute("INSTALL tpcds")
        connection.execute("LOAD tpcds")
        connection.execute("CREATE SCHEMA raw")
        connection.execute("CREATE SCHEMA analytics")
        connection.execute("CALL dsdgen(schema = 'raw', sf = ?, keys = true)", [scale_factor])
        validate_source_schema(connection)
        materialize_customer_revenue_yearly(connection, transformation)
        validate_customer_revenue_yearly(connection)
        table_row_counts = collect_table_row_counts(connection)
    finally:
        connection.close()
    result = create_generation_manifest(
        database_path=database,
        manifest_path=manifest,
        scale_factor=scale_factor,
        transformation_sql_path=transformation,
        table_row_counts=table_row_counts,
    )
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def validate_source_schema(connection: duckdb.DuckDBPyConnection) -> None:
    """Confirm the extension produced the source fields needed by the transformation."""
    for table_name, expected_columns in REQUIRED_SOURCE_COLUMNS.items():
        actual_columns = {
            row[0]
            for row in connection.execute(
                f"DESCRIBE raw.{_quote_identifier(table_name)}"
            ).fetchall()
        }
        missing_columns = expected_columns - actual_columns
        if missing_columns:
            raise WarehouseBuildError(
                f"raw.{table_name} is missing required columns: {', '.join(sorted(missing_columns))}"
            )


def materialize_customer_revenue_yearly(
    connection: duckdb.DuckDBPyConnection, transformation_sql_path: str | Path
) -> None:
    """Create the Phase 1 derived asset from the version-controlled transformation."""
    connection.execute(Path(transformation_sql_path).read_text(encoding="utf-8"))


def validate_customer_revenue_yearly(connection: duckdb.DuckDBPyConnection) -> None:
    """Verify the derived table has expected fields and reconciles across sales channels."""
    actual_columns = {
        row[0]
        for row in connection.execute("DESCRIBE analytics.customer_revenue_yearly").fetchall()
    }
    missing_columns = REVENUE_COLUMNS - actual_columns
    if missing_columns:
        raise WarehouseBuildError(
            "analytics.customer_revenue_yearly is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    row_count = connection.execute(
        "SELECT COUNT(*) FROM analytics.customer_revenue_yearly"
    ).fetchone()[0]
    if row_count == 0:
        raise WarehouseBuildError("analytics.customer_revenue_yearly must contain at least one row")
    failures = connection.execute(
        "SELECT COUNT(*) FROM analytics.customer_revenue_yearly WHERE total_net_paid <> store_net_paid + catalog_net_paid + web_net_paid"
    ).fetchone()[0]
    if failures:
        raise WarehouseBuildError(
            f"customer revenue channel totals do not reconcile for {failures} row(s)"
        )


def collect_table_row_counts(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Collect row counts for all source tables and the derived table."""
    source_tables = connection.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'raw' AND table_type = 'BASE TABLE' ORDER BY table_name"
    ).fetchall()
    row_counts = {
        f"raw.{table_name}": connection.execute(
            f"SELECT COUNT(*) FROM raw.{_quote_identifier(table_name)}"
        ).fetchone()[0]
        for (table_name,) in source_tables
    }
    row_counts["analytics.customer_revenue_yearly"] = connection.execute(
        "SELECT COUNT(*) FROM analytics.customer_revenue_yearly"
    ).fetchone()[0]
    return row_counts


def create_generation_manifest(
    *,
    database_path: Path,
    manifest_path: Path,
    scale_factor: float,
    transformation_sql_path: Path,
    table_row_counts: dict[str, int],
) -> dict[str, object]:
    """Produce tracked provenance for the otherwise ignored database artifact."""
    return {
        "database_file": str(database_path),
        "database_sha256": _sha256_file(database_path),
        "duckdb_version": duckdb.__version__,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "manifest_file": str(manifest_path),
        "scale_factor": scale_factor,
        "table_row_counts": table_row_counts,
        "transformation_sql_file": str(transformation_sql_path),
        "transformation_sql_sha256": _sha256_file(transformation_sql_path),
    }


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
