from __future__ import annotations

from pathlib import Path

from semantic_lineage.catalog.documents import build_semantic_documents
from semantic_lineage.catalog.lineage import get_downstream_lineage, get_upstream_lineage
from semantic_lineage.catalog.loader import load_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / "metadata" / "tpcds_revenue_catalog.yaml"


def test_generates_semantic_documents_from_the_canonical_catalog() -> None:
    catalog = load_catalog(CATALOG_PATH)
    documents = build_semantic_documents(catalog)
    document_by_id = {document.id: document for document in documents}

    assert len(documents) == 47
    revenue_column = document_by_id["column:analytics.customer_revenue_yearly.total_net_paid"]
    assert revenue_column.document_type == "column"
    assert "customer spending" in revenue_column.content
    assert revenue_column.payload["table_name"] == "customer_revenue_yearly"


def test_explicit_lineage_traversal_returns_all_revenue_inputs() -> None:
    catalog = load_catalog(CATALOG_PATH)

    upstream = get_upstream_lineage(catalog, "analytics.customer_revenue_yearly.total_net_paid")
    downstream = get_downstream_lineage(catalog, "raw.store_sales.ss_net_paid")

    assert {edge.source_column_id for edge in upstream} == {
        "raw.store_sales.ss_net_paid",
        "raw.catalog_sales.cs_net_paid",
        "raw.web_sales.ws_net_paid",
    }
    assert {edge.target_column_id for edge in downstream} == {
        "analytics.customer_revenue_yearly.store_net_paid",
        "analytics.customer_revenue_yearly.total_net_paid",
    }
