"""Offline, isolated end-to-end verification for the isolated vertical slice."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import TypeVar

from qdrant_client import QdrantClient

from semantic_lineage.agents.planner import DeterministicPlanner
from semantic_lineage.catalog.documents import SemanticDocument, build_semantic_documents
from semantic_lineage.catalog.loader import load_catalog
from semantic_lineage.catalog.schema_validation import validate_catalog_against_duckdb
from semantic_lineage.evaluation import (
    RetrievalEvaluationReport,
    evaluate_retrieval,
    load_retrieval_cases,
)
from semantic_lineage.exploration import CatalogExplorer
from semantic_lineage.retrieval.embeddings import DeterministicHashEmbeddingProvider
from semantic_lineage.retrieval.qdrant_store import QdrantSemanticStore, assert_qdrant_ready
from semantic_lineage.retrieval.search import semantic_search
from semantic_lineage.warehouse.builder import build_warehouse
from semantic_lineage.workflow.graph import SemanticDataWorkflow

DEFAULT_E2E_COLLECTION = "semantic_catalog_e2e"
EXPECTED_DOCUMENT_COUNT = 47
EXPECTED_TOP_CUSTOMERS = [
    ("AAAAAAAAHCFAAAAA", "Luis", "Drew", Decimal("144587.55")),
    ("AAAAAAAAPMKAAAAA", "Joseph", "Bush", Decimal("140586.79")),
    ("AAAAAAAALIGAAAAA", "Herbert", "Ross", Decimal("139967.64")),
]


@dataclass(frozen=True, slots=True)
class E2EStageResult:
    name: str
    status: str
    duration_seconds: float
    details: dict[str, object]


@dataclass(slots=True)
class E2EReport:
    workspace: str
    collection: str
    stages: list[E2EStageResult] = field(default_factory=list)
    evaluation_metrics: dict[str, float] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "workspace": self.workspace,
            "collection": self.collection,
            "stages": [asdict(stage) for stage in self.stages],
            "evaluation_metrics": self.evaluation_metrics,
        }


class E2EError(RuntimeError):
    """Raised with a completed stage report when the offline flow cannot continue."""

    def __init__(self, stage: str, cause: Exception, report: E2EReport) -> None:
        super().__init__(f"E2E stage '{stage}' failed: {cause}")
        self.stage = stage
        self.report = report


T = TypeVar("T")


def run_offline_e2e(
    *,
    qdrant_url: str,
    workspace: str | Path,
    project_root: str | Path,
    collection_name: str = DEFAULT_E2E_COLLECTION,
) -> E2EReport:
    """Run the credential-free end-to-end flow against isolated warehouse artifacts and collection."""
    root = Path(project_root).resolve()
    workdir = Path(workspace).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    database_path = workdir / "semantic_lineage_e2e.duckdb"
    manifest_path = workdir / "tpcds_generation.json"
    evaluation_path = workdir / "retrieval_report.json"
    catalog_path = root / "metadata" / "tpcds_revenue_catalog.yaml"
    cases_path = root / "evaluation" / "retrieval_cases.yaml"
    report = E2EReport(workspace=str(workdir), collection=collection_name)
    client = QdrantClient(url=qdrant_url)
    provider = DeterministicHashEmbeddingProvider()
    store = QdrantSemanticStore(client, provider, collection_name)

    _run_stage(report, "qdrant_readiness", lambda: _ready_details(client))
    catalog = load_catalog(catalog_path)
    _run_stage(
        report,
        "warehouse_build",
        lambda: build_warehouse(
            database_path=database_path,
            manifest_path=manifest_path,
            transformation_sql_path=root / "sql" / "customer_revenue_yearly.sql",
            overwrite=True,
        ),
    )
    _run_stage(
        report,
        "catalog_validation",
        lambda: asdict(validate_catalog_against_duckdb(catalog, database_path, root)),
    )
    documents = build_semantic_documents(catalog)
    _run_stage(report, "metadata_indexing", lambda: _index_details(store, documents))
    _run_stage(report, "metadata_reindexing", lambda: _index_details(store, documents))
    _run_stage(report, "metadata_reconciliation", lambda: _reconcile_details(store, documents))

    explorer = CatalogExplorer(catalog, store)
    workflow = SemanticDataWorkflow(catalog, explorer, DeterministicPlanner(), database_path)
    _run_stage(report, "semantic_discovery", lambda: _assert_discovery(workflow))
    _run_stage(report, "explicit_lineage", lambda: _assert_lineage(workflow))
    _run_stage(report, "sql_backed_analysis", lambda: _assert_analysis(workflow))
    evaluation = _run_stage(
        report,
        "retrieval_evaluation",
        lambda: evaluate_retrieval(
            load_retrieval_cases(cases_path),
            lambda query, limit: semantic_search(store, query, limit=limit),
        ),
    )
    evaluation_path.write_text(_render_evaluation(evaluation), encoding="utf-8")
    metrics = evaluation.as_dict()["metrics"]
    assert isinstance(metrics, dict)
    report.evaluation_metrics = {key: float(value) for key, value in metrics.items()}
    report.stages[-1] = E2EStageResult(
        name="retrieval_evaluation",
        status="passed",
        duration_seconds=report.stages[-1].duration_seconds,
        details={
            "case_count": evaluation.case_count,
            "report_path": str(evaluation_path),
            **metrics,
        },
    )
    return report


def _run_stage(report: E2EReport, name: str, operation: Callable[[], T]) -> T:
    started = time.monotonic()
    try:
        value = operation()
    except Exception as exc:
        report.stages.append(
            E2EStageResult(
                name=name,
                status="failed",
                duration_seconds=round(time.monotonic() - started, 3),
                details={"error": str(exc)},
            )
        )
        raise E2EError(name, exc, report) from exc
    details = value if isinstance(value, dict) else {}
    report.stages.append(
        E2EStageResult(
            name=name,
            status="passed",
            duration_seconds=round(time.monotonic() - started, 3),
            details=details,
        )
    )
    return value


def _ready_details(client: QdrantClient) -> dict[str, object]:
    assert_qdrant_ready(client)
    return {"status": "reachable"}


def _index_details(
    store: QdrantSemanticStore, documents: list[SemanticDocument]
) -> dict[str, object]:
    result = store.index_documents(documents)
    count = store.point_count()
    if result.documents_indexed != EXPECTED_DOCUMENT_COUNT or count != EXPECTED_DOCUMENT_COUNT:
        raise AssertionError(
            f"expected {EXPECTED_DOCUMENT_COUNT} indexed documents, got {result.documents_indexed}/{count}"
        )
    return {"documents_indexed": result.documents_indexed, "point_count": count}


def _reconcile_details(
    store: QdrantSemanticStore, documents: list[SemanticDocument]
) -> dict[str, object]:
    reduced_result = store.index_documents(documents[:-1])
    if (
        reduced_result.documents_deleted != 1
        or reduced_result.collection_point_count != len(documents) - 1
    ):
        raise AssertionError("reconciliation did not remove exactly one stale document")
    restored_result = store.index_documents(documents)
    if restored_result.documents_deleted != 0 or restored_result.collection_point_count != len(
        documents
    ):
        raise AssertionError("reconciliation did not restore the canonical document set")
    return {
        "documents_deleted": reduced_result.documents_deleted,
        "reduced_point_count": reduced_result.collection_point_count,
        "restored_point_count": restored_result.collection_point_count,
    }


def _assert_discovery(workflow: SemanticDataWorkflow) -> dict[str, object]:
    response = workflow.ask("Where can I find customer spending information?")
    if response.intent != "discovery" or response.sql is not None:
        raise AssertionError("discovery must route without SQL")
    asset_ids = set(response.candidate_asset_ids)
    if "analytics.customer_revenue_yearly" not in asset_ids or not asset_ids.intersection(
        {"raw.store_sales", "raw.catalog_sales", "raw.web_sales"}
    ):
        raise AssertionError("discovery did not resolve yearly revenue and a sales-channel asset")
    if not response.retrieved_document_ids:
        raise AssertionError("discovery did not retain retrieved document IDs")
    return {
        "asset_ids": sorted(asset_ids),
        "retrieved_document_count": len(response.retrieved_document_ids),
    }


def _assert_lineage(workflow: SemanticDataWorkflow) -> dict[str, object]:
    response = workflow.ask("Where does yearly customer revenue come from?")
    required_columns = {
        "raw.store_sales.ss_net_paid",
        "raw.catalog_sales.cs_net_paid",
        "raw.web_sales.ws_net_paid",
    }
    if response.intent != "lineage" or response.sql is not None:
        raise AssertionError("lineage must route without SQL")
    if not response.lineage_context or not all(
        column in response.lineage_context for column in required_columns
    ):
        raise AssertionError("lineage did not include all documented net-paid inputs")
    if "Customer Yearly Net Revenue" not in response.answer:
        raise AssertionError("lineage did not resolve the documented metric")
    return {"required_source_columns": sorted(required_columns)}


def _assert_analysis(workflow: SemanticDataWorkflow) -> dict[str, object]:
    response = workflow.ask("Who were the top 3 customers by net paid revenue in 2001?")
    if response.intent != "analysis" or response.sql is None or response.result is None:
        raise AssertionError("analysis did not return SQL and query results")
    if (
        "analytics.customer_revenue_yearly" not in response.sql
        or "raw.customer" not in response.sql
    ):
        raise AssertionError("analysis SQL did not use the documented assets")
    if response.result.rows != EXPECTED_TOP_CUSTOMERS:
        raise AssertionError(
            f"analysis rows differ from expected top customers: {response.result.rows}"
        )
    if not response.lineage_context or "raw.web_sales.ws_net_paid" not in response.lineage_context:
        raise AssertionError("analysis did not retain revenue lineage context")
    return {"row_count": len(response.result.rows), "sql": response.sql}


def _render_evaluation(evaluation: RetrievalEvaluationReport) -> str:
    from json import dumps

    return dumps(evaluation.as_dict(), indent=2, sort_keys=True, default=str) + "\n"
