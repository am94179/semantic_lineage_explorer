"""Explore semantic catalog discovery and explicit lineage through a local Qdrant index."""

from __future__ import annotations

import argparse
from pathlib import Path

from semantic_lineage.catalog.loader import CatalogLoadError, load_catalog
from semantic_lineage.config import Settings
from semantic_lineage.exploration import CatalogExplorer, ExplorationError
from semantic_lineage.presentation.responses import (
    render_discovery_response,
    render_lineage_response,
)
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
    parser.add_argument("--test-embedder", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    discover_parser = subparsers.add_parser("discover", help="Find relevant catalog assets.")
    discover_parser.add_argument("question")
    discover_parser.add_argument("--limit", type=int, default=8)
    lineage_parser = subparsers.add_parser("lineage", help="Show explicit upstream lineage.")
    lineage_parser.add_argument("target_or_question")
    lineage_parser.add_argument("--limit", type=int, default=8)
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
        explorer = CatalogExplorer(
            load_catalog(arguments.catalog),
            QdrantSemanticStore(
                create_qdrant_client(settings.qdrant_url), provider, collection_name
            ),
        )
        if arguments.command == "discover":
            print(render_discovery_response(explorer.discover(arguments.question, arguments.limit)))
        else:
            print(
                render_lineage_response(
                    explorer.lineage(arguments.target_or_question, arguments.limit)
                )
            )
    except (
        CatalogLoadError,
        EmbeddingProviderError,
        ExplorationError,
        QdrantServiceError,
        ValueError,
    ) as exc:
        print(f"Catalog exploration failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
