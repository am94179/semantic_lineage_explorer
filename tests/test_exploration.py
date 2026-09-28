from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

from semantic_lineage.catalog.documents import build_semantic_documents
from semantic_lineage.catalog.loader import load_catalog
from semantic_lineage.exploration import CatalogExplorer
from semantic_lineage.presentation.responses import (
    render_discovery_response,
    render_lineage_response,
)
from semantic_lineage.retrieval.embeddings import DeterministicHashEmbeddingProvider
from semantic_lineage.retrieval.qdrant_store import QdrantSemanticStore
from semantic_lineage.retrieval.search import SemanticSearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "metadata" / "tpcds_revenue_catalog.yaml"


def create_explorer() -> CatalogExplorer:
    catalog = load_catalog(CATALOG_PATH)
    store = QdrantSemanticStore(
        QdrantClient(":memory:"),
        DeterministicHashEmbeddingProvider(dimensions=128),
        "exploration_test",
    )
    store.index_documents(build_semantic_documents(catalog))
    return CatalogExplorer(catalog, store)


def test_discovery_response_includes_assets_relationships_and_evidence() -> None:
    response = create_explorer().discover("where can I find customer spending", limit=10)
    rendered = render_discovery_response(response)

    assert {asset.id for asset in response.assets} >= {
        "raw.store_sales",
        "raw.catalog_sales",
        "raw.web_sales",
        "analytics.customer_revenue_yearly",
    }
    assert response.relationships
    assert response.evidence
    assert "Relevant data assets:" in rendered
    assert "Retrieved semantic evidence:" in rendered


def test_natural_language_metric_name_takes_precedence_over_weak_vector_ranking() -> None:
    response = create_explorer().lineage("where does yearly customer revenue come from")

    assert response.target_metric is not None
    assert response.target_metric.id == "customer_yearly_net_revenue"


def test_transformation_evidence_resolves_declared_input_assets() -> None:
    explorer = create_explorer()
    response = explorer.discover_from_evidence(
        "transformation inputs",
        [
            SemanticSearchResult(
                "transformation:build_customer_revenue_yearly",
                "transformation",
                "",
                {
                    "target_asset_id": "analytics.customer_revenue_yearly",
                    "input_asset_ids": [
                        "raw.store_sales",
                        "raw.catalog_sales",
                        "raw.web_sales",
                        "raw.date_dim",
                    ],
                },
                1.0,
            )
        ],
    )

    assert {asset.id for asset in response.assets} >= {
        "raw.store_sales",
        "raw.catalog_sales",
        "raw.web_sales",
        "raw.date_dim",
    }


def test_lineage_response_uses_explicit_edges_without_an_llm() -> None:
    response = create_explorer().lineage("customer_yearly_net_revenue")
    rendered = render_lineage_response(response)

    assert response.target_metric is not None
    assert response.target_column_id == "analytics.customer_revenue_yearly.total_net_paid"
    assert {edge.source_column_id for edge in response.upstream_edges} == {
        "raw.store_sales.ss_net_paid",
        "raw.catalog_sales.cs_net_paid",
        "raw.web_sales.ws_net_paid",
    }
    assert "Explicit upstream lineage:" in rendered
    assert "returns are not subtracted" in rendered
