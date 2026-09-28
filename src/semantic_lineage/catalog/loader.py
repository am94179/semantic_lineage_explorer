"""Load the canonical semantic catalog from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from semantic_lineage.models import Catalog


class CatalogLoadError(ValueError):
    """Raised when a YAML catalog cannot be parsed into the canonical model."""


def load_catalog(path: str | Path) -> Catalog:
    """Parse and validate a complete catalog YAML document."""
    catalog_path = Path(path)
    try:
        raw_catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CatalogLoadError(f"could not read catalog '{catalog_path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise CatalogLoadError(f"catalog '{catalog_path}' is not valid YAML: {exc}") from exc

    if raw_catalog is None:
        raw_catalog = {}
    if not isinstance(raw_catalog, dict):
        raise CatalogLoadError(f"catalog '{catalog_path}' must contain a YAML mapping")

    try:
        return Catalog.model_validate(raw_catalog)
    except ValidationError as exc:
        raise CatalogLoadError(f"catalog '{catalog_path}' failed validation:\n{exc}") from exc
