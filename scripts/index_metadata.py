"""Index canonical catalog documents into a running local Qdrant service."""

from __future__ import annotations

import argparse
from pathlib import Path

from semantic_lineage.catalog.documents import build_semantic_documents
from semantic_lineage.catalog.loader import CatalogLoadError, load_catalog
from semantic_lineage.config import Settings
from semantic_lineage.retrieval.embeddings import (
    DeterministicHashEmbeddingProvider,
    EmbeddingProviderError,
    create_embedding_provider,
)
from semantic_lineage.retrieval.qdrant_store import (
    QdrantSemanticStore,
    QdrantServiceError,
    create_qdrant_client,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("metadata/tpcds_revenue_catalog.yaml"))
    parser.add_argument("--collection")
    parser.add_argument(
        "--test-embedder",
        action="store_true",
        help="Use deterministic hash embeddings for local plumbing checks; not retrieval evaluation.",
    )
    arguments = parser.parse_args()
    try:
        catalog = load_catalog(arguments.catalog)
        settings = Settings.from_environment()
        provider = (
            DeterministicHashEmbeddingProvider()
            if arguments.test_embedder
            else create_embedding_provider(settings)
        )
        collection_name = arguments.collection or (
            "semantic_catalog_test" if arguments.test_embedder else "semantic_catalog"
        )
        store = QdrantSemanticStore(
            create_qdrant_client(settings.qdrant_url), provider, collection_name
        )
        result = store.index_documents(build_semantic_documents(catalog))
    except (CatalogLoadError, EmbeddingProviderError, QdrantServiceError, ValueError) as exc:
        print(f"Metadata indexing failed: {exc}")
        return 1
    print(
        f"Indexed {result.documents_indexed} documents; deleted {result.documents_deleted} stale document(s); collection '{result.collection_name}' now contains {result.collection_point_count} point(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
