from __future__ import annotations

from semantic_lineage.e2e import E2EReport, E2EStageResult


def test_e2e_report_serializes_stage_and_metric_data() -> None:
    report = E2EReport(
        workspace="/tmp/semantic_lineage_e2e",
        collection="semantic_catalog_e2e",
        stages=[E2EStageResult("qdrant_readiness", "passed", 0.12, {"status": "reachable"})],
        evaluation_metrics={
            "recall_at_5": 0.75,
            "recall_at_10": 0.875,
            "mean_reciprocal_rank": 0.64,
        },
    )

    assert report.as_dict() == {
        "workspace": "/tmp/semantic_lineage_e2e",
        "collection": "semantic_catalog_e2e",
        "stages": [
            {
                "name": "qdrant_readiness",
                "status": "passed",
                "duration_seconds": 0.12,
                "details": {"status": "reachable"},
            }
        ],
        "evaluation_metrics": {
            "recall_at_5": 0.75,
            "recall_at_10": 0.875,
            "mean_reciprocal_rank": 0.64,
        },
    }
