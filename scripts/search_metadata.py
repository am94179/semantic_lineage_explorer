"""Search catalog metadata in a running local Qdrant service."""

from __future__ import annotations

import argparse

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
from semantic_lineage.retrieval.search import semantic_search


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--collection")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--document-type", action="append", dest="document_types")
    parser.add_argument("--domain", action="append", dest="domains")
    parser.add_argument("--asset-id", action="append", dest="asset_ids")
    parser.add_argument("--test-embedder", action="store_true")
    arguments = parser.parse_args()
    try:
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
        results = semantic_search(
            store,
            arguments.query,
            limit=arguments.limit,
            document_types=arguments.document_types,
            domains=arguments.domains,
            asset_ids=arguments.asset_ids,
        )
    except (EmbeddingProviderError, QdrantServiceError, ValueError) as exc:
        print(f"Semantic search failed: {exc}")
        return 1
    for result in results:
        print(f"[{result.score:.3f}] {result.document_type}: {result.document_id}")
        print(result.content)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
