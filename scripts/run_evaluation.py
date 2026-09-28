"""Run the version-controlled retrieval benchmark against a Qdrant collection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from semantic_lineage.catalog.documents import build_semantic_documents
from semantic_lineage.catalog.loader import CatalogLoadError, load_catalog
from semantic_lineage.config import Settings
from semantic_lineage.evaluation import (
    EvaluationLoadError,
    EvaluationProvenance,
    build_evaluation_payload,
    evaluate_retrieval,
    file_sha256,
    load_retrieval_cases,
    utc_timestamp,
    validate_collection_compatibility,
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
from semantic_lineage.retrieval.search import semantic_search


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("metadata/tpcds_revenue_catalog.yaml"))
    parser.add_argument("--cases", type=Path, default=Path("evaluation/retrieval_cases.yaml"))
    parser.add_argument("--collection")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--test-embedder",
        action="store_true",
        help="Evaluate deterministic embeddings for CI/plumbing only, not semantic quality.",
    )
    parser.add_argument(
        "--real-embedding-baseline",
        action="store_true",
        help="Explicitly run an opt-in real-provider semantic retrieval baseline.",
    )
    arguments = parser.parse_args()
    if arguments.test_embedder and arguments.real_embedding_baseline:
        parser.error("--test-embedder and --real-embedding-baseline cannot be combined")
    if not arguments.test_embedder and not arguments.real_embedding_baseline:
        parser.error("use --test-embedder or explicitly opt in with --real-embedding-baseline")
    if arguments.real_embedding_baseline and not arguments.collection:
        parser.error("--real-embedding-baseline requires an explicit --collection name")

    started_at_utc = utc_timestamp()
    started_at = time.perf_counter()
    try:
        settings = Settings.from_environment()
        catalog = load_catalog(arguments.catalog)
        documents = build_semantic_documents(catalog)
        provider = (
            DeterministicHashEmbeddingProvider()
            if arguments.test_embedder
            else create_embedding_provider(settings)
        )
        collection = arguments.collection or "semantic_catalog_test"
        store = QdrantSemanticStore(create_qdrant_client(settings.qdrant_url), provider, collection)
        summary = store.collection_summary()
        validate_collection_compatibility(
            summary.vector_dimensions, summary.point_count, provider.dimensions, len(documents)
        )
        report = evaluate_retrieval(
            load_retrieval_cases(arguments.cases),
            lambda query, limit: semantic_search(store, query, limit=limit),
        )
    except (
        CatalogLoadError,
        EmbeddingProviderError,
        EvaluationLoadError,
        QdrantServiceError,
        ValueError,
    ) as exc:
        print(f"Retrieval evaluation failed: {exc}")
        return 1

    provenance = EvaluationProvenance(
        run_type="deterministic_test" if arguments.test_embedder else "real_embedding_baseline",
        embedding_provider=provider.provider_name,
        embedding_model=provider.model_name,
        embedding_dimensions=provider.dimensions,
        catalog_path=str(arguments.catalog),
        catalog_sha256=file_sha256(arguments.catalog),
        catalog_version="unversioned",
        benchmark_path=str(arguments.cases),
        benchmark_sha256=file_sha256(arguments.cases),
        collection=collection,
        document_count=summary.point_count,
        qdrant_version=summary.qdrant_version,
        qdrant_url=settings.qdrant_url,
        started_at_utc=started_at_utc,
        elapsed_seconds=round(time.perf_counter() - started_at, 6),
    )
    rendered = (
        json.dumps(build_evaluation_payload(report, provenance), indent=2, sort_keys=True) + "\n"
    )
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
