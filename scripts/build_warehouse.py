"""Generate the local TPC-DS DuckDB warehouse for this project."""

from __future__ import annotations

import argparse
from pathlib import Path

from semantic_lineage.warehouse.builder import (
    DEFAULT_DATABASE_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SCALE_FACTOR,
    TRANSFORMATION_SQL_PATH,
    WarehouseBuildError,
    build_warehouse,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--scale-factor", type=float, default=DEFAULT_SCALE_FACTOR)
    parser.add_argument("--transformation-sql", type=Path, default=TRANSFORMATION_SQL_PATH)
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing database at --database."
    )
    arguments = parser.parse_args()
    try:
        manifest = build_warehouse(
            database_path=arguments.database,
            manifest_path=arguments.manifest,
            scale_factor=arguments.scale_factor,
            transformation_sql_path=arguments.transformation_sql,
            overwrite=arguments.force,
        )
    except WarehouseBuildError as exc:
        print(f"Warehouse build failed: {exc}")
        return 1
    print(
        f"Warehouse build completed: {manifest['database_file']} (DuckDB {manifest['duckdb_version']}, SF {manifest['scale_factor']})."
    )
    print(f"Generation manifest: {arguments.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
