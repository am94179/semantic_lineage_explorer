from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import duckdb
import pytest
from qdrant_client import QdrantClient

from semantic_lineage.agents.planner import DeterministicPlanner, IntentDecision, SQLPlan
from semantic_lineage.catalog.documents import build_semantic_documents
from semantic_lineage.catalog.loader import load_catalog
from semantic_lineage.exploration import CatalogExplorer
from semantic_lineage.retrieval.embeddings import DeterministicHashEmbeddingProvider
from semantic_lineage.retrieval.qdrant_store import QdrantSemanticStore
from semantic_lineage.workflow.graph import SemanticDataWorkflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "metadata" / "tpcds_revenue_catalog.yaml"


def create_workflow(
    tmp_path: Path,
    planner: DeterministicPlanner | None = None,
    *,
    query_timeout_seconds: float = 10.0,
) -> SemanticDataWorkflow:
    database_path = _create_workflow_warehouse(tmp_path / "workflow.duckdb")
    catalog = load_catalog(CATALOG_PATH)
    store = QdrantSemanticStore(
        QdrantClient(":memory:"),
        DeterministicHashEmbeddingProvider(dimensions=128),
        "workflow_test",
    )
    store.index_documents(build_semantic_documents(catalog))
    return SemanticDataWorkflow(
        catalog,
        CatalogExplorer(catalog, store),
        planner or DeterministicPlanner(),
        database_path,
        query_timeout_seconds=query_timeout_seconds,
    )


def _create_workflow_warehouse(database_path: Path) -> Path:
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute("CREATE SCHEMA raw")
        connection.execute("CREATE SCHEMA analytics")
        connection.execute(
            "CREATE TABLE raw.customer (c_customer_sk BIGINT, c_customer_id VARCHAR, c_first_name VARCHAR, c_last_name VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE raw.date_dim (d_date_sk BIGINT, d_date DATE, d_year BIGINT)"
        )
        connection.execute(
            "CREATE TABLE raw.store_sales (ss_sold_date_sk BIGINT, ss_customer_sk BIGINT, ss_net_paid DECIMAL(7,2))"
        )
        connection.execute(
            "CREATE TABLE raw.catalog_sales (cs_sold_date_sk BIGINT, cs_bill_customer_sk BIGINT, cs_net_paid DECIMAL(7,2))"
        )
        connection.execute(
            "CREATE TABLE raw.web_sales (ws_sold_date_sk BIGINT, ws_bill_customer_sk BIGINT, ws_net_paid DECIMAL(7,2))"
        )
        connection.execute(
            "CREATE TABLE analytics.customer_revenue_yearly (customer_sk BIGINT, calendar_year BIGINT, store_net_paid DECIMAL(38,2), catalog_net_paid DECIMAL(38,2), web_net_paid DECIMAL(38,2), total_net_paid DECIMAL(38,2))"
        )
        connection.execute(
            "INSERT INTO raw.customer VALUES (1, 'C001', 'Luis', 'Drew'), (2, 'C002', 'Joseph', 'Bush'), (3, 'C003', 'Herbert', 'Ross')"
        )
        connection.execute(
            "INSERT INTO analytics.customer_revenue_yearly VALUES (1, 2001, 0, 0, 0, 300.00), (2, 2001, 0, 0, 0, 200.00), (3, 2001, 0, 0, 0, 100.00)"
        )
    finally:
        connection.close()
    return database_path


def test_discovery_route_does_not_generate_or_execute_sql(tmp_path: Path) -> None:
    response = create_workflow(tmp_path).ask("where can I find customer spending")

    assert response.intent == "discovery"
    assert response.sql is None
    assert "Relevant data assets:" in response.answer
    assert response.retrieved_document_ids


def test_lineage_route_uses_explicit_catalog_lineage_without_sql(tmp_path: Path) -> None:
    response = create_workflow(tmp_path).ask("where does yearly customer revenue come from")

    assert response.intent == "lineage"
    assert response.sql is None
    assert "Explicit upstream lineage:" in response.answer


def test_analysis_route_returns_expected_fixture_results_and_sql(tmp_path: Path) -> None:
    response = create_workflow(tmp_path).ask(
        "Who were the top 3 customers by net paid revenue in 2001?"
    )

    assert response.intent == "analysis"
    assert response.sql is not None
    assert "analytics.customer_revenue_yearly" in response.sql
    assert response.result is not None
    assert response.result.rows == [
        ("C001", "Luis", "Drew", Decimal("300.00")),
        ("C002", "Joseph", "Bush", Decimal("200.00")),
        ("C003", "Herbert", "Ross", Decimal("100.00")),
    ]
    assert "DuckDB returned 3 row(s)" in response.answer


class FixedIntentPlanner(DeterministicPlanner):
    def classify_intent(self, question: str, retrieval_context: str) -> IntentDecision:
        return IntentDecision("discovery", "mocked structured intent result")


def test_workflow_routing_uses_the_planner_structured_intent(tmp_path: Path) -> None:
    response = create_workflow(tmp_path, FixedIntentPlanner()).ask(
        "Who were the top customers in 2001?"
    )

    assert response.intent == "discovery"
    assert response.sql is None


def test_analysis_route_reports_an_actionable_query_timeout(tmp_path: Path) -> None:
    workflow = create_workflow(tmp_path, query_timeout_seconds=0.001)

    with pytest.raises(
        RuntimeError, match="DuckDB query exceeded the 0.001-second execution limit"
    ):
        workflow.ask("Who were the top 3 customers by net paid revenue in 2001?")


class UnknownColumnPlanner(DeterministicPlanner):
    def generate_sql(self, question, schemas, retrieval_context):
        return SQLPlan(
            "SELECT r.not_a_column FROM analytics.customer_revenue_yearly AS r",
            "intentionally invalid regression plan",
        )


def test_analysis_route_reports_schema_validation_failure_before_duckdb(tmp_path: Path) -> None:
    workflow = create_workflow(tmp_path, UnknownColumnPlanner())

    with pytest.raises(
        RuntimeError, match="SQL validation failed: column is not present in inspected schema"
    ):
        workflow.ask("Who were the top 3 customers by net paid revenue in 2001?")
