"""DuckDB warehouse construction and validation."""

from semantic_lineage.warehouse.builder import WarehouseBuildError, build_warehouse

__all__ = ["WarehouseBuildError", "build_warehouse"]
