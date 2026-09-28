from __future__ import annotations

import os
from pathlib import Path

import pytest

from semantic_lineage.e2e import run_offline_e2e

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.e2e


@pytest.mark.skipif(
    os.getenv("RUN_E2E") != "1",
    reason="requires Docker Qdrant and TPC-DS generation; run with RUN_E2E=1",
)
def test_offline_e2e_pipeline(tmp_path: Path) -> None:
    report = run_offline_e2e(
        qdrant_url=os.getenv("SEMANTIC_LINEAGE_QDRANT_URL", "http://localhost:6333"),
        workspace=tmp_path,
        project_root=PROJECT_ROOT,
        collection_name="semantic_catalog_e2e_pytest",
    )

    assert all(stage.status == "passed" for stage in report.stages)
    assert [stage.name for stage in report.stages] == [
        "qdrant_readiness",
        "warehouse_build",
        "catalog_validation",
        "metadata_indexing",
        "metadata_reindexing",
        "metadata_reconciliation",
        "semantic_discovery",
        "explicit_lineage",
        "sql_backed_analysis",
        "retrieval_evaluation",
    ]
    assert report.evaluation_metrics is not None
    assert all(0 <= value <= 1 for value in report.evaluation_metrics.values())
    assert (tmp_path / "retrieval_report.json").is_file()
