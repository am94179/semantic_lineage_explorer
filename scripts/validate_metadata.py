"""Validate a canonical semantic catalog file without requiring external services."""

from __future__ import annotations

import argparse
from pathlib import Path

from semantic_lineage.catalog.loader import CatalogLoadError, load_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path, help="Path to a complete catalog YAML file")
    arguments = parser.parse_args()

    try:
        catalog = load_catalog(arguments.catalog)
    except CatalogLoadError as exc:
        print(f"Metadata validation failed: {exc}")
        return 1

    print(
        "Metadata validation passed: "
        f"{len(catalog.assets)} assets, "
        f"{len(catalog.columns)} columns, "
        f"{len(catalog.relationships)} relationships, "
        f"{len(catalog.transformations)} transformations, "
        f"{len(catalog.lineage_edges)} lineage edges, "
        f"{len(catalog.metrics)} metrics."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
