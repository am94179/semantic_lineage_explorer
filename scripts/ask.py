"""Ask the Phase 5 semantic-data workflow one question."""

from __future__ import annotations

import argparse
from pathlib import Path

from semantic_lineage.agents.planner import DeterministicPlanner, OpenAIPlanner, PlannerError
from semantic_lineage.catalog.loader import load_catalog
from semantic_lineage.config import Settings
from semantic_lineage.exploration import CatalogExplorer
from semantic_lineage.retrieval.embeddings import (
    DeterministicHashEmbeddingProvider,
    create_embedding_provider,
)
from semantic_lineage.retrieval.qdrant_store import (
    QdrantSemanticStore,
    QdrantServiceError,
    create_qdrant_client,
)
from semantic_lineage.workflow.graph import SemanticDataWorkflow, WorkflowResponse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--catalog", type=Path, default=Path("metadata/tpcds_revenue_catalog.yaml"))
    parser.add_argument("--database", type=Path, default=Path("data/semantic_lineage.duckdb"))
    parser.add_argument("--collection")
    parser.add_argument("--test-embedder", action="store_true")
    parser.add_argument("--test-planner", action="store_true")
    parser.add_argument("--query-timeout-seconds", type=float, default=10.0)
    arguments = parser.parse_args()
    try:
        settings = Settings.from_environment()
        provider = (
            DeterministicHashEmbeddingProvider()
            if arguments.test_embedder
            else create_embedding_provider(settings)
        )
        collection = arguments.collection or (
            "semantic_catalog_test" if arguments.test_embedder else "semantic_catalog"
        )
        explorer = CatalogExplorer(
            load_catalog(arguments.catalog),
            QdrantSemanticStore(create_qdrant_client(settings.qdrant_url), provider, collection),
        )
        workflow = SemanticDataWorkflow(
            load_catalog(arguments.catalog),
            explorer,
            DeterministicPlanner() if arguments.test_planner else OpenAIPlanner(settings),
            arguments.database,
            query_timeout_seconds=arguments.query_timeout_seconds,
        )
        _render(workflow.ask(arguments.question))
    except (PlannerError, QdrantServiceError, RuntimeError, ValueError) as exc:
        print(f"Workflow failed: {exc}")
        return 1
    return 0


def _render(response: WorkflowResponse) -> None:
    print(f"Intent: {response.intent}")
    print(response.answer)
    print("\nRetrieved document IDs:")
    print("\n".join(f"- {document_id}" for document_id in response.retrieved_document_ids))
    print("\nCandidate assets:")
    print("\n".join(f"- {asset_id}" for asset_id in response.candidate_asset_ids))
    if response.sql:
        print(f"\nExecuted SQL:\n{response.sql}")
    if response.lineage_context:
        print(f"\nLineage context:\n{response.lineage_context}")


if __name__ == "__main__":
    raise SystemExit(main())
