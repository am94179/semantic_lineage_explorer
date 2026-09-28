from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import QdrantClient

from semantic_lineage.catalog.documents import SemanticDocument, build_semantic_documents
from semantic_lineage.catalog.loader import load_catalog
from semantic_lineage.retrieval.embeddings import DeterministicHashEmbeddingProvider
from semantic_lineage.retrieval.qdrant_store import (
    QdrantSemanticStore,
    QdrantServiceError,
    assert_qdrant_ready,
)
from semantic_lineage.retrieval.search import semantic_search

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / "metadata" / "tpcds_revenue_catalog.yaml"


def create_test_store(
    client: QdrantClient | None = None, collection_name: str = "test_semantic_catalog"
) -> tuple[QdrantClient, QdrantSemanticStore]:
    client = client or QdrantClient(":memory:")
    store = QdrantSemanticStore(
        client, DeterministicHashEmbeddingProvider(dimensions=128), collection_name
    )
    return client, store


def documents() -> list[SemanticDocument]:
    return build_semantic_documents(load_catalog(CATALOG_PATH))


def point_id(document_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, document_id))


def test_indexing_is_idempotent() -> None:
    client, store = create_test_store()

    first_result = store.index_documents(documents())
    second_result = store.index_documents(documents())

    assert first_result.documents_indexed == 47
    assert first_result.documents_deleted == 0
    assert second_result.documents_indexed == 47
    assert second_result.documents_deleted == 0
    assert second_result.collection_point_count == 47
    assert client.count("test_semantic_catalog", exact=True).count == 47


def test_indexing_reconciles_a_removed_document() -> None:
    _, store = create_test_store()
    initial_documents = documents()
    removed = initial_documents[0]
    store.index_documents(initial_documents)

    result = store.index_documents(
        [document for document in initial_documents if document.id != removed.id]
    )

    assert result.documents_indexed == 46
    assert result.documents_deleted == 1
    assert result.collection_point_count == 46
    assert point_id(removed.id) not in store.point_ids()


def test_indexing_reconciles_a_renamed_document() -> None:
    _, store = create_test_store()
    initial_documents = documents()
    replaced = initial_documents[0]
    renamed = SemanticDocument(
        id="asset:raw.customer_renamed",
        document_type=replaced.document_type,
        content=replaced.content,
        payload=replaced.payload,
    )
    store.index_documents(initial_documents)

    result = store.index_documents(
        [renamed if document.id == replaced.id else document for document in initial_documents]
    )

    assert result.documents_deleted == 1
    assert result.collection_point_count == 47
    assert point_id(replaced.id) not in store.point_ids()
    assert point_id(renamed.id) in store.point_ids()


def test_indexing_empty_desired_state_removes_all_points() -> None:
    _, store = create_test_store()
    store.index_documents(documents())

    result = store.index_documents([])

    assert result.documents_indexed == 0
    assert result.documents_deleted == 47
    assert result.collection_point_count == 0
    assert store.point_ids() == set()


def test_reconciliation_does_not_modify_another_collection() -> None:
    client, store = create_test_store()
    _, other_store = create_test_store(client, "other_semantic_catalog")
    initial_documents = documents()
    store.index_documents(initial_documents)
    other_store.index_documents(initial_documents)

    store.index_documents(initial_documents[:-1])

    assert store.point_count() == 46
    assert other_store.point_count() == 47


def test_point_ids_scrolls_all_records_in_pages() -> None:
    _, store = create_test_store()
    indexed_documents = documents()
    store.index_documents(indexed_documents)

    assert store.point_ids(batch_size=7) == {
        point_id(document.id) for document in indexed_documents
    }


def test_search_returns_customer_spending_assets_and_respects_filters() -> None:
    _, store = create_test_store()
    store.index_documents(documents())

    results = semantic_search(store, "where can I find customer spending", limit=10)
    column_results = semantic_search(
        store,
        "customer spending",
        limit=10,
        document_types=["column"],
        domains=["analytics"],
        asset_ids=["analytics.customer_revenue_yearly"],
    )

    assert any("customer_revenue_yearly" in result.document_id for result in results)
    assert column_results
    assert all(result.document_type == "column" for result in column_results)
    assert all(
        result.payload["asset_id"] == "analytics.customer_revenue_yearly"
        for result in column_results
    )


def test_readiness_failure_has_an_actionable_message() -> None:
    client = MagicMock()
    client.get_collections.side_effect = RuntimeError("connection refused")

    with pytest.raises(QdrantServiceError, match="docker compose up -d"):
        assert_qdrant_ready(client)


def test_collection_summary_reports_dimensions_count_and_qdrant_version() -> None:
    _, store = create_test_store()
    store.index_documents(documents())

    summary = store.collection_summary()

    assert summary.collection_name == "test_semantic_catalog"
    assert summary.vector_dimensions == 128
    assert summary.point_count == 47
    assert summary.qdrant_version
